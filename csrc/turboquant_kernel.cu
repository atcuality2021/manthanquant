/**
 * turboquant_kernel.cu — TurboQuant: Near-optimal KV cache compression.
 *
 * Implements TurboQuant (Zandieh et al., 2025) Algorithm 1:
 *
 * ENCODE:
 *   1. Compute radius r = ||x|| and normalize: u = x / r
 *   2. Apply random rotation via SRHT: y = WHT(signs ⊙ u)
 *      After rotation, each coordinate follows Beta → N(0, 1/d)
 *   3. Quantize each coordinate independently using pre-computed
 *      optimal Lloyd-Max centroids for N(0, 1/d)
 *   4. Store: radius (float32) + b-bit index per coordinate
 *
 * DECODE:
 *   1. Look up centroids from indices
 *   2. Scale by radius
 *   3. Apply inverse rotation: WHT + sign flip (self-inverse)
 *
 * Key insight: After random rotation, coordinates are nearly independent
 * and identically distributed, so optimal scalar quantization per
 * coordinate achieves near-optimal vector quantization distortion.
 *
 * No per-vector normalization constants needed (unlike KIVI, QuaRot etc.)
 * — the codebook is universal and pre-computed from the analytical
 * distribution. This is the fundamental advantage of TurboQuant.
 *
 * Memory: D-dim bf16 vector = 2D bytes →
 *   b=2: radius(4B) + ceil(D/16)*4B packed = 4 + D/4 bytes
 *   b=3: radius(4B) + ceil(D/10)*4B packed = 4 + ~0.4D bytes
 *   For D=128, b=3: 4 + 52 = 56B  (4.6x compression)
 *   For D=256, b=3: 4 + 104 = 108B (4.7x compression)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include "packing.cuh"

namespace manthanquant {

// ══════════════════════════════════════════════════════════════════════════
// PRE-COMPUTED OPTIMAL LLOYD-MAX CENTROIDS FOR N(0,1)
// ══════════════════════════════════════════════════════════════════════════
//
// These are the optimal scalar quantizer centroids that minimize MSE
// for the standard normal distribution N(0,1).
//
// At runtime, we scale by 1/sqrt(D) since after SRHT rotation the
// coordinates follow N(0, 1/D) for unit-norm vectors.
//
// Computed via iterative Lloyd-Max algorithm (converges in ~20 iterations).
// Values match Table 1 in Max (1960) "Quantizing for minimum distortion".
// ══════════════════════════════════════════════════════════════════════════

// b=2: 4 centroids, 3 boundaries
__constant__ float C2[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};
__constant__ float B2[3] = {-0.9816f, 0.0f, 0.9816f};

// b=3: 8 centroids, 7 boundaries
__constant__ float C3[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f
};
__constant__ float B3[7] = {
    -1.7480f, -1.0500f, -0.5006f, 0.0f,
     0.5006f,  1.0500f,  1.7480f
};

// b=4: 16 centroids, 15 boundaries
__constant__ float C4[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};
__constant__ float B4[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7996f, -0.5224f, -0.2582f,  0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f
};

constexpr int MAX_HEAD_DIM = 512;

// ── Random sign-flip generation (deterministic from seed) ────────────────

__device__ __forceinline__
float random_sign(int dim_idx, uint32_t seed) {
    uint32_t h = seed ^ (dim_idx * 2654435761u);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    return (h & 1) ? 1.0f : -1.0f;
}

// ── Fast Walsh-Hadamard Transform (in-place, shared memory) ──────────────

__device__
void fast_wht_shared(float* s, int D, int tid) {
    for (int half = 1; half < D; half <<= 1) {
        __syncthreads();
        if (tid < D) {
            int block_size = half << 1;
            int block_id = tid / block_size;
            int local_id = tid % block_size;
            if (local_id < half) {
                int i = block_id * block_size + local_id;
                int j = i + half;
                float a = s[i];
                float b = s[j];
                s[i] = a + b;
                s[j] = a - b;
            }
        }
    }
    __syncthreads();
    if (tid < D) {
        s[tid] *= rsqrtf(static_cast<float>(D));
    }
    __syncthreads();
}

// ── Quantize: map scaled coordinate to nearest Lloyd-Max centroid ────────
//
// Input x is a coordinate of the rotated unit vector, distributed ~ N(0, 1/D).
// We work in standardized space: x_std = x * sqrt(D) ~ N(0, 1), then
// quantize against the N(0,1) boundaries.

__device__ __forceinline__
uint32_t quantize_2bit(float x_std) {
    // 3 boundaries → 4 levels. Binary search (2 comparisons).
    if (x_std < B2[1]) {  // B2[1] = 0
        return (x_std < B2[0]) ? 0u : 1u;
    } else {
        return (x_std < B2[2]) ? 2u : 3u;
    }
}

__device__ __forceinline__
uint32_t quantize_3bit(float x_std) {
    // 7 boundaries → 8 levels. Binary search (3 comparisons).
    if (x_std < B3[3]) {  // B3[3] = 0
        if (x_std < B3[1]) {
            return (x_std < B3[0]) ? 0u : 1u;
        } else {
            return (x_std < B3[2]) ? 2u : 3u;
        }
    } else {
        if (x_std < B3[5]) {
            return (x_std < B3[4]) ? 4u : 5u;
        } else {
            return (x_std < B3[6]) ? 6u : 7u;
        }
    }
}

__device__ __forceinline__
uint32_t quantize_4bit(float x_std) {
    // 15 boundaries → 16 levels. Binary search (4 comparisons).
    if (x_std < B4[7]) {  // B4[7] = 0
        if (x_std < B4[3]) {
            if (x_std < B4[1]) {
                return (x_std < B4[0]) ? 0u : 1u;
            } else {
                return (x_std < B4[2]) ? 2u : 3u;
            }
        } else {
            if (x_std < B4[5]) {
                return (x_std < B4[4]) ? 4u : 5u;
            } else {
                return (x_std < B4[6]) ? 6u : 7u;
            }
        }
    } else {
        if (x_std < B4[11]) {
            if (x_std < B4[9]) {
                return (x_std < B4[8]) ? 8u : 9u;
            } else {
                return (x_std < B4[10]) ? 10u : 11u;
            }
        } else {
            if (x_std < B4[13]) {
                return (x_std < B4[12]) ? 12u : 13u;
            } else {
                return (x_std < B4[14]) ? 14u : 15u;
            }
        }
    }
}

// Dispatch quantization by bit width
__device__ __forceinline__
uint32_t quantize_coord(float x_std, int bits) {
    if (bits == 2) return quantize_2bit(x_std);
    if (bits == 3) return quantize_3bit(x_std);
    return quantize_4bit(x_std);
}

// ── Dequantize: index → centroid value in original scale ────────────────

__device__ __forceinline__
float dequantize_coord(uint32_t idx, int bits, float inv_sqrt_d) {
    // Look up N(0,1) centroid, scale to N(0, 1/D)
    float c;
    if (bits == 2) c = C2[idx];
    else if (bits == 3) c = C3[idx];
    else c = C4[idx];
    return c * inv_sqrt_d;
}

// ══════════════════════════════════════════════════════════════════════════
// ENCODE KERNEL
// ══════════════════════════════════════════════════════════════════════════
//
// Input:  vectors [N, D] in float32 (pre-cast by wrapper)
// Output: radii   [N]    in float32 (L2 norm)
//         packed  [N, num_words] in uint32 (b-bit indices)
//
// One block per vector. D threads cooperate on WHT + quantization.
// ══════════════════════════════════════════════════════════════════════════

__global__
void tq_encode_kernel(
    const float* __restrict__ input,   // [N, D]
    float*       __restrict__ radii,   // [N]
    uint32_t*    __restrict__ packed,  // [N, num_words]
    int N, int D, int num_words, int bits,
    uint32_t seed
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* reduce_buf = smem + D;

    // 1. Load vector + apply sign flips
    if (tid < D) {
        smem[tid] = input[vec_idx * D + tid] * random_sign(tid, seed);
    }
    __syncthreads();

    // 2. Walsh-Hadamard Transform
    fast_wht_shared(smem, D, tid);

    // 3. Compute L2 norm (parallel reduction)
    float my_sq = (tid < D) ? smem[tid] * smem[tid] : 0.0f;
    reduce_buf[tid] = my_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }
    float radius = sqrtf(reduce_buf[0]);

    // 4. Normalize to unit vector
    float inv_radius = (radius > 1e-8f) ? (1.0f / radius) : 0.0f;
    if (tid < D) {
        smem[tid] *= inv_radius;
    }
    __syncthreads();

    // Store radius
    if (tid == 0) {
        radii[vec_idx] = radius;
    }

    // 5. Quantize each coordinate using optimal Lloyd-Max centroids
    //    Coordinate distribution: u_i ~ Beta ≈ N(0, 1/D) for unit vectors
    //    Standardize: x_std = u_i * sqrt(D) ~ N(0, 1)
    if (tid < D) {
        float sqrt_d = sqrtf(static_cast<float>(D));
        float x_std = smem[tid] * sqrt_d;

        uint32_t level = quantize_coord(x_std, bits);

        // Pack into output
        int vpw = vals_per_word(bits);
        int word_idx = tid / vpw;
        int pos = tid % vpw;
        int out_off = vec_idx * num_words + word_idx;

        atomicOr(&packed[out_off], (level & bit_mask(bits)) << (pos * bits));
    }
}

// ══════════════════════════════════════════════════════════════════════════
// DECODE KERNEL
// ══════════════════════════════════════════════════════════════════════════

__global__
void tq_decode_kernel(
    const float*    __restrict__ radii,   // [N]
    const uint32_t* __restrict__ packed,  // [N, num_words]
    float*          __restrict__ output,  // [N, D]
    int N, int D, int num_words, int bits,
    uint32_t seed
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    int tid = threadIdx.x;
    extern __shared__ float smem[];

    float radius = radii[vec_idx];
    float inv_sqrt_d = rsqrtf(static_cast<float>(D));

    // 1. Unpack indices → centroid values → scale by radius
    if (tid < D) {
        int vpw = vals_per_word(bits);
        int word_idx = tid / vpw;
        int pos = tid % vpw;
        uint32_t word = packed[vec_idx * num_words + word_idx];
        uint32_t level = (word >> (pos * bits)) & bit_mask(bits);

        float centroid = dequantize_coord(level, bits, inv_sqrt_d);
        smem[tid] = centroid * radius;
    }
    __syncthreads();

    // 2. Inverse WHT (self-inverse with normalization)
    fast_wht_shared(smem, D, tid);

    // 3. Undo sign flips
    if (tid < D) {
        output[vec_idx * D + tid] = smem[tid] * random_sign(tid, seed);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// TORCH C++ WRAPPERS
// ══════════════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor> tq_encode(
    torch::Tensor input,
    int64_t seed,
    int64_t bits
) {
    TORCH_CHECK(input.dim() == 2, "Input must be [N, D]");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(bits >= 2 && bits <= 4, "bits must be 2, 3, or 4, got ", bits);

    int N = input.size(0);
    int D = input.size(1);

    TORCH_CHECK((D & (D - 1)) == 0, "D must be a power of 2, got ", D);
    TORCH_CHECK(D <= MAX_HEAD_DIM, "D must be <= ", MAX_HEAD_DIM);

    auto input_f32 = input.to(torch::kFloat32).contiguous();
    int num_words = packed_word_count(D, bits);

    auto radii = torch::empty({N}, input_f32.options());
    auto packed = torch::zeros({N, num_words}, input.options().dtype(torch::kInt32));

    if (N == 0) return {radii, packed};

    int threads = D;
    int smem_bytes = 2 * D * sizeof(float);

    tq_encode_kernel<<<N, threads, smem_bytes>>>(
        input_f32.data_ptr<float>(),
        radii.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
        N, D, num_words, static_cast<int>(bits),
        static_cast<uint32_t>(seed)
    );

    return {radii, packed};
}

torch::Tensor tq_decode(
    torch::Tensor radii,
    torch::Tensor packed,
    int64_t D,
    int64_t seed,
    int64_t bits
) {
    TORCH_CHECK(radii.dim() == 1, "Radii must be [N]");
    TORCH_CHECK(packed.dim() == 2, "Packed must be [N, words]");
    TORCH_CHECK(radii.is_cuda() && packed.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(bits >= 2 && bits <= 4, "bits must be 2, 3, or 4");

    int N = radii.size(0);
    int num_words = packed_word_count(D, bits);
    TORCH_CHECK(packed.size(1) == num_words, "Packed size mismatch: got ",
                packed.size(1), " expected ", num_words);
    TORCH_CHECK((D & (D - 1)) == 0, "D must be a power of 2");

    auto output = torch::empty({N, D}, radii.options());
    if (N == 0) return output;

    int threads = D;
    int smem_bytes = D * sizeof(float);

    tq_decode_kernel<<<N, threads, smem_bytes>>>(
        radii.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
        output.data_ptr<float>(),
        N, D, num_words, static_cast<int>(bits),
        static_cast<uint32_t>(seed)
    );

    return output;
}

}  // namespace manthanquant
