/**
 * qjl_kernel.cu — Quantized Johnson-Lindenstrauss residual correction.
 *
 * After PolarQuant compresses a KV vector, the reconstruction error
 * (original - reconstructed) introduces bias in attention scores.
 * QJL corrects this bias using a 1-bit projection:
 *
 * ENCODE:
 *   1. Compute error e = v_original - v_reconstructed  (in rotated space)
 *   2. Project error with a random matrix: p = R_jl · e  (R_jl is D×M Rademacher)
 *   3. Store only the signs: s[i] = sign(p[i])  → 1 bit each
 *   4. Pack signs into uint32 words (32 signs per word)
 *
 * CORRECTION (at attention time):
 *   Given query q, key k (compressed), and QJL signs s_k:
 *   1. Project query: p_q = R_jl · q  → M floats
 *   2. Compute correction: Σ |p_q[i]| * s_k[i]  (using stored signs)
 *   3. Scale by sqrt(pi/(2*M)) — the JL estimator for E[sign(x)*|y|] ≈ <x,y>/||x||
 *   4. Add correction to the PolarQuant attention score
 *
 * Memory cost: M bits per vector. With M = D (typical), that's D/8 bytes.
 *   For D=128: 16 extra bytes per vector (still 4x+ total compression)
 *
 * The Rademacher matrix is never stored — generated on-the-fly from a seed,
 * same as PolarQuant's sign flips.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

namespace manthanquant {

// ── Random Rademacher entry from seed ────────────────────────────────────

/**
 * Generate a random +1/-1 value for the JL projection matrix.
 * R_jl[i, j] = rademacher(i, j, seed)
 * Uses a different hash than PolarQuant's sign flips to ensure independence.
 */
__device__ __forceinline__
float rademacher(int row, int col, uint32_t seed) {
    // Cantor pairing + mixing
    uint32_t h = seed ^ ((row * 0x9e3779b9u) + (col * 0x517cc1b7u));
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

// ══════════════════════════════════════════════════════════════════════════
// QJL ENCODE KERNEL
// ══════════════════════════════════════════════════════════════════════════
//
// Input:  errors [N, D]  — quantization error vectors (float32)
// Output: signs  [N, sign_words]  — packed sign bits (uint32)
//
// M = D (projection dimension = input dimension for maximum correction)
// One block per vector.
// ══════════════════════════════════════════════════════════════════════════

__global__
void qjl_encode_kernel(
    const float* __restrict__ errors,   // [N, D]
    uint32_t*    __restrict__ signs,    // [N, sign_words]
    int N,
    int D,
    int M,              // projection dim (typically = D)
    int sign_words,     // ceil(M / 32)
    uint32_t seed
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    int tid = threadIdx.x;

    // Each thread computes one or more projection outputs
    // p[m] = Σ_d R_jl[m, d] * error[d]
    for (int m = tid; m < M; m += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            float r = rademacher(m, d, seed);
            dot += r * errors[vec_idx * D + d];
        }

        // Store sign bit: 1 if positive, 0 if negative
        uint32_t sign_bit = (dot >= 0.0f) ? 1u : 0u;
        int word_idx = m / 32;
        int bit_pos  = m % 32;

        atomicOr(&signs[vec_idx * sign_words + word_idx],
                 sign_bit << bit_pos);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// QJL ATTENTION CORRECTION KERNEL
// ══════════════════════════════════════════════════════════════════════════
//
// For each (query, key) pair, compute the QJL correction term:
//   correction = (1/M) * Σ_m sign(R·e[m]) * (R·q[m])
//
// This is the unbiased estimator: E[sign(Re) · (Rq)] ≈ <e, q> * sqrt(2/(πM))
// but using the simpler (1/M) * Σ sign * proj form which is also unbiased
// when R is Rademacher (each row is ±1).
//
// where p_q = R_jl · query  (computed on-the-fly)
//
// Input:  queries [B, D]           — query vectors (float32)
//         signs   [N_keys, words]  — packed sign bits of key errors
// Output: corrections [B, N_keys] — additive correction to attention scores
//
// This kernel is for verification. The production path fuses this into
// the main attention computation.
// ══════════════════════════════════════════════════════════════════════════

__global__
void qjl_correction_kernel(
    const float*    __restrict__ queries,      // [B, D]
    const uint32_t* __restrict__ key_signs,    // [N_keys, sign_words]
    float*          __restrict__ corrections,  // [B, N_keys]
    int B,
    int N_keys,
    int D,
    int M,
    int sign_words,
    uint32_t seed
) {
    // One block per (query, key) pair — simplistic for correctness
    // Production version would tile this more efficiently
    int query_idx = blockIdx.x;
    int key_idx   = blockIdx.y;
    if (query_idx >= B || key_idx >= N_keys) return;

    int tid = threadIdx.x;

    // Shared memory for partial sums
    extern __shared__ float partial[];

    float my_sum = 0.0f;

    // Each thread handles a subset of projection dimensions
    for (int m = tid; m < M; m += blockDim.x) {
        // Compute p_q[m] = Σ_d R_jl[m, d] * query[d]
        float p_q = 0.0f;
        for (int d = 0; d < D; d++) {
            p_q += rademacher(m, d, seed) * queries[query_idx * D + d];
        }

        // Get sign of key error projection
        int word_idx = m / 32;
        int bit_pos  = m % 32;
        uint32_t word = key_signs[key_idx * sign_words + word_idx];
        float sign_k = ((word >> bit_pos) & 1u) ? 1.0f : -1.0f;

        // Accumulate sign(R·e) * (R·q) — unbiased estimator for <e, q>
        my_sum += sign_k * p_q;
    }

    partial[tid] = my_sum;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Scale: (1/M) normalizes the sum of M random projections.
        // For Rademacher R: E[(1/M) Σ sign(Re_m)(Rq_m)] = <e, q>
        // The sqrt(π/2) factor corrects for sign quantization bias.
        float scale = sqrtf(M_PI / 2.0f) / static_cast<float>(M);
        corrections[query_idx * N_keys + key_idx] = partial[0] * scale;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// TORCH C++ WRAPPERS
// ══════════════════════════════════════════════════════════════════════════

/**
 * Encode quantization errors into QJL sign bits.
 *
 * Args:
 *   errors: [N, D] float32 — error vectors (original - polar_decoded)
 *   M: projection dimension (default = D)
 *   seed: random seed for Rademacher matrix
 *
 * Returns: [N, sign_words] int32 (packed uint32 sign bits)
 */
torch::Tensor qjl_encode(
    torch::Tensor errors,
    int64_t M,
    int64_t seed
) {
    TORCH_CHECK(errors.dim() == 2, "Errors must be [N, D]");
    TORCH_CHECK(errors.is_cuda(), "Errors must be on CUDA");

    int N = errors.size(0);
    int D = errors.size(1);
    if (M <= 0) M = D;

    int sign_words = (M + 31) / 32;

    auto errors_f32 = errors.to(torch::kFloat32).contiguous();
    auto signs = torch::zeros({N, sign_words}, errors.options().dtype(torch::kInt32));

    if (N == 0) return signs;

    // Threads: min(M, 256)
    int threads = std::min(static_cast<int>(M), 256);

    qjl_encode_kernel<<<N, threads>>>(
        errors_f32.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(signs.data_ptr<int32_t>()),
        N, D, M, sign_words,
        static_cast<uint32_t>(seed)
    );

    return signs;
}

/**
 * Compute QJL attention score corrections.
 *
 * Args:
 *   queries:   [B, D] float32
 *   key_signs: [N_keys, sign_words] int32 (packed sign bits)
 *   D: original vector dimension
 *   M: projection dimension
 *   seed: same seed used in qjl_encode
 *
 * Returns: [B, N_keys] float32 correction terms
 */
torch::Tensor qjl_correction(
    torch::Tensor queries,
    torch::Tensor key_signs,
    int64_t D,
    int64_t M,
    int64_t seed
) {
    TORCH_CHECK(queries.dim() == 2, "Queries must be [B, D]");
    TORCH_CHECK(key_signs.dim() == 2, "Key signs must be [N_keys, words]");

    int B = queries.size(0);
    int N_keys = key_signs.size(0);
    if (M <= 0) M = D;

    auto queries_f32 = queries.to(torch::kFloat32).contiguous();
    auto corrections = torch::empty({B, N_keys}, queries_f32.options());

    if (B == 0 || N_keys == 0) return corrections;

    int threads = std::min(static_cast<int>(M), 256);
    dim3 grid(B, N_keys);

    qjl_correction_kernel<<<grid, threads, threads * sizeof(float)>>>(
        queries_f32.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(key_signs.data_ptr<int32_t>()),
        corrections.data_ptr<float>(),
        B, N_keys, D, M, (M + 31) / 32,
        static_cast<uint32_t>(seed)
    );

    return corrections;
}

}  // namespace manthanquant
