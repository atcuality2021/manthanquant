/**
 * bindings.cpp — pybind11 / torch bindings for ManthanQuant CUDA ops.
 *
 * Exposes TurboQuant (encode/decode), QJL, and fused attention kernels.
 * Legacy polar_encode/polar_decode wrappers maintained for transition.
 */

#include <torch/extension.h>

namespace manthanquant {

// TurboQuant (turboquant_kernel.cu)
std::tuple<torch::Tensor, torch::Tensor> tq_encode(torch::Tensor input, int64_t seed, int64_t bits);
torch::Tensor tq_decode(torch::Tensor radii, torch::Tensor packed, int64_t D, int64_t seed, int64_t bits);

// QJL — disabled (glibc/CUDA math conflicts on Ubuntu 25.10)
// torch::Tensor qjl_encode(torch::Tensor errors, int64_t M, int64_t seed);
// torch::Tensor qjl_correction(torch::Tensor queries, torch::Tensor key_signs, int64_t D, int64_t M, int64_t seed);

// Fused attention — disabled on Ubuntu 25.10 (glibc/CUDA math conflicts)
// torch::Tensor fused_compressed_attention(
//     torch::Tensor queries, torch::Tensor k_radii, torch::Tensor k_packed,
//     torch::Tensor v_radii, torch::Tensor v_packed,
//     int64_t num_kv_heads, int64_t seed, int64_t bits);

}  // namespace manthanquant

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ManthanQuant — TurboQuant KV cache compression (Lloyd-Max + QJL)";

    // ── TurboQuant (primary API) ────────────────────────────────────────
    m.def("tq_encode", &manthanquant::tq_encode,
          "TurboQuant encode: vectors [N, D] → (radii [N], packed [N, words])",
          py::arg("input"), py::arg("seed") = 42, py::arg("bits") = 3);

    m.def("tq_decode", &manthanquant::tq_decode,
          "TurboQuant decode: (radii, packed) → vectors [N, D]",
          py::arg("radii"), py::arg("packed"),
          py::arg("D"), py::arg("seed") = 42, py::arg("bits") = 3);

    // ── QJL — disabled (glibc/CUDA math conflicts on Ubuntu 25.10) ──

    // ── Fused attention — disabled (glibc/CUDA math conflicts on Ubuntu 25.10) ──

}
