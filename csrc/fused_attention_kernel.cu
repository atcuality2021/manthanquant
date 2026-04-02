/**
 * fused_attention_kernel.cu — Fused decompress + attention for TurboQuant.
 *
 * Takes compressed KV cache + queries, produces attention output directly.
 * No intermediate decompression buffer — saves a full memory round-trip.
 *
 * Computes: output = softmax(Q · K_decomp^T / sqrt(d)) · V_decomp
 *
 * Decompression uses pre-computed Lloyd-Max centroids (no per-vector
 * normalization constants). K and V are decompressed on-the-fly in
 * rotated space; the final output is inverse-rotated once.
 *
 * Grid:  one block per (query_token, head)
 * Each block processes ALL seq_len keys for ONE query using online softmax.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include "packing.cuh"

namespace manthanquant {

// ── Duplicated helpers in fused_impl namespace (avoids ODR with turboquant_kernel.cu)

namespace fused_impl {

// Lloyd-Max centroids for N(0,1) — must match turboquant_kernel.cu
__constant__ float FC3[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f
};

__constant__ float FC2[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};

__device__ __forceinline__
float dequantize_coord(uint32_t idx, int bits, float inv_sqrt_d) {
    float c;
    if (bits == 3) c = FC3[idx];
    else c = FC2[idx];
    return c * inv_sqrt_d;
}

__device__ __forceinline__
float random_sign(int dim_idx, uint32_t seed) {
    uint32_t h = seed ^ (dim_idx * 2654435761u);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    return (h & 1) ? 1.0f : -1.0f;
}

__device__
void fast_wht_shared(float* s, int D, int tid) {
    for (int half_step = 1; half_step < D; half_step <<= 1) {
        __syncthreads();
        if (tid < D) {
            int block_size = half_step << 1;
            int block_id = tid / block_size;
            int local_id = tid % block_size;
            if (local_id < half_step) {
                int i = block_id * block_size + local_id;
                int j = i + half_step;
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

}  // namespace fused_impl

// ══════════════════════════════════════════════════════════════════════════
// FUSED COMPRESSED ATTENTION KERNEL
// ══════════════════════════════════════════════════════════════════════════
//
// Input:
//   queries:   [Q, H, D]  float32
//   k_radii:   [S, KH]    float32
//   k_packed:  [S, KH, W] int32  (b-bit packed indices)
//   v_radii:   [S, KH]    float32
//   v_packed:  [S, KH, W] int32
// Output:
//   output:    [Q, H, D]  float32
//
// No maxabs needed — centroids are universal (pre-computed from N(0,1)).
// ══════════════════════════════════════════════════════════════════════════

__global__
void fused_compressed_attention_kernel(
    const float*   __restrict__ queries,   // [Q, H, D]
    const float*   __restrict__ k_radii,   // [S, KH]
    const int32_t* __restrict__ k_packed,  // [S, KH, W]
    const float*   __restrict__ v_radii,   // [S, KH]
    const int32_t* __restrict__ v_packed,  // [S, KH, W]
    float*         __restrict__ output,    // [Q, H, D]
    int num_queries,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int num_words,
    int bits,
    float scale,     // 1/sqrt(head_dim)
    uint32_t seed
) {
    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    if (q_idx >= num_queries || h_idx >= num_heads) return;

    int tid = threadIdx.x;
    int kv_h = h_idx / (num_heads / num_kv_heads);

    float inv_sqrt_d = rsqrtf(static_cast<float>(head_dim));
    int vpw = vals_per_word(bits);

    // Shared memory: [0..D) = query, [D..2D) = workspace, [2D..3D) = unused
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* workspace = smem + head_dim;

    // Load query into shared memory
    if (tid < head_dim) {
        q_shared[tid] = queries[q_idx * num_heads * head_dim + h_idx * head_dim + tid];
    }
    __syncthreads();

    // Rotate query into WHT space: q_rot = WHT(signs ⊙ q)
    if (tid < head_dim) {
        q_shared[tid] *= fused_impl::random_sign(tid, seed);
    }
    __syncthreads();
    fused_impl::fast_wht_shared(q_shared, head_dim, tid);

    // Online softmax over all key positions
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float my_output = 0.0f;

    for (int s = 0; s < seq_len; s++) {
        float kr = k_radii[s * num_kv_heads + kv_h];

        // Decompress key[s] in rotated space: centroid * radius
        float k_val = 0.0f;
        if (tid < head_dim) {
            int w_idx = tid / vpw;
            int pos = tid % vpw;
            uint32_t word = static_cast<uint32_t>(
                k_packed[s * num_kv_heads * num_words + kv_h * num_words + w_idx]
            );
            uint32_t level = (word >> (pos * bits)) & bit_mask(bits);
            k_val = fused_impl::dequantize_coord(level, bits, inv_sqrt_d) * kr;
        }

        // Dot product q_rot · k_rot (parallel reduction)
        float partial = (tid < head_dim) ? q_shared[tid] * k_val : 0.0f;
        workspace[tid] = partial;
        __syncthreads();
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (tid < stride) workspace[tid] += workspace[tid + stride];
            __syncthreads();
        }
        float score = workspace[0] * scale;

        // Decompress value[s] in rotated space
        float vr = v_radii[s * num_kv_heads + kv_h];
        float v_val = 0.0f;
        if (tid < head_dim) {
            int w_idx = tid / vpw;
            int pos = tid % vpw;
            uint32_t word = static_cast<uint32_t>(
                v_packed[s * num_kv_heads * num_words + kv_h * num_words + w_idx]
            );
            uint32_t level = (word >> (pos * bits)) & bit_mask(bits);
            v_val = fused_impl::dequantize_coord(level, bits, inv_sqrt_d) * vr;
        }

        // Online softmax update
        float new_max = fmaxf(max_score, score);
        float old_scale_factor = expf(max_score - new_max);
        float new_weight = expf(score - new_max);

        if (tid < head_dim) {
            my_output = my_output * old_scale_factor + new_weight * v_val;
        }
        sum_exp = sum_exp * old_scale_factor + new_weight;
        max_score = new_max;
        __syncthreads();
    }

    // Normalize and inverse-rotate output back to original space
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    if (tid < head_dim) {
        smem[tid] = my_output * inv_sum;
    }
    __syncthreads();

    fused_impl::fast_wht_shared(smem, head_dim, tid);

    if (tid < head_dim) {
        float val = smem[tid] * fused_impl::random_sign(tid, seed);
        output[q_idx * num_heads * head_dim + h_idx * head_dim + tid] = val;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// TORCH C++ WRAPPER
// ══════════════════════════════════════════════════════════════════════════

torch::Tensor fused_compressed_attention(
    torch::Tensor queries,   // [Q, H, D]
    torch::Tensor k_radii,   // [S, KH]
    torch::Tensor k_packed,  // [S, KH, W]
    torch::Tensor v_radii,   // [S, KH]
    torch::Tensor v_packed,  // [S, KH, W]
    int64_t num_kv_heads,
    int64_t seed,
    int64_t bits
) {
    TORCH_CHECK(queries.dim() == 3, "Queries must be [Q, H, D]");
    TORCH_CHECK(queries.is_cuda(), "Queries must be on CUDA");
    TORCH_CHECK(bits >= 2 && bits <= 4, "bits must be 2, 3, or 4");

    int num_queries = queries.size(0);
    int num_heads = queries.size(1);
    int head_dim = queries.size(2);
    int seq_len = k_radii.size(0);
    int num_words = packed_word_count(head_dim, bits);

    TORCH_CHECK((head_dim & (head_dim - 1)) == 0, "head_dim must be power of 2");

    auto queries_f32 = queries.to(torch::kFloat32).contiguous();
    auto k_radii_c = k_radii.to(torch::kFloat32).contiguous();
    auto k_packed_c = k_packed.to(torch::kInt32).contiguous();
    auto v_radii_c = v_radii.to(torch::kFloat32).contiguous();
    auto v_packed_c = v_packed.to(torch::kInt32).contiguous();

    auto output = torch::empty({num_queries, num_heads, head_dim}, queries_f32.options());
    if (num_queries == 0 || seq_len == 0) return output;

    float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(num_queries, num_heads);
    int threads = head_dim;
    int smem_bytes = 2 * head_dim * sizeof(float);

    fused_compressed_attention_kernel<<<grid, threads, smem_bytes>>>(
        queries_f32.data_ptr<float>(),
        k_radii_c.data_ptr<float>(),
        k_packed_c.data_ptr<int32_t>(),
        v_radii_c.data_ptr<float>(),
        v_packed_c.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        num_queries, num_heads,
        static_cast<int>(num_kv_heads),
        seq_len, head_dim, num_words,
        static_cast<int>(bits), attn_scale,
        static_cast<uint32_t>(seed)
    );

    return output;
}

}  // namespace manthanquant
