/**
 * packing.cuh — Generalized b-bit packing/unpacking for TurboQuant.
 *
 * Supports 2-bit, 3-bit, and 4-bit packing into uint32 words.
 *
 * Layout per word:
 *   b=2: 16 values × 2 bits = 32 bits (0 wasted)
 *   b=3: 10 values × 3 bits = 30 bits (2 wasted)
 *   b=4:  8 values × 4 bits = 32 bits (0 wasted)
 */

#pragma once
#include <cstdint>

namespace manthanquant {

// ── Compile-time packing parameters ─────────────────────────────────────

// Values per uint32 word for each bit width
__host__ __device__ __forceinline__
constexpr int vals_per_word(int bits) {
    return (bits == 2) ? 16 : (bits == 3) ? 10 : (bits == 4) ? 8 : 0;
}

// Mask for extracting one value
__host__ __device__ __forceinline__
constexpr uint32_t bit_mask(int bits) {
    return (1u << bits) - 1u;
}

// Number of uint32 words needed to pack n values at given bit width
__host__ __device__ __forceinline__
int packed_word_count(int n, int bits) {
    int vpw = vals_per_word(bits);
    return (n + vpw - 1) / vpw;
}

// ── Device pack/unpack ──────────────────────────────────────────────────

// Pack a single value at position idx within the word
__device__ __forceinline__
void pack_val(uint32_t& word, int idx, uint32_t val, int bits) {
    word |= (val & bit_mask(bits)) << (idx * bits);
}

// Extract a single value at position idx from a packed word
__device__ __forceinline__
uint32_t unpack_val(uint32_t word, int idx, int bits) {
    return (word >> (idx * bits)) & bit_mask(bits);
}

// Given a flat index into the value array, compute (word_idx, pos_within_word)
__device__ __forceinline__
void flat_to_packed(int flat_idx, int bits, int& word_idx, int& pos) {
    int vpw = vals_per_word(bits);
    word_idx = flat_idx / vpw;
    pos = flat_idx % vpw;
}

// ── Legacy 3-bit compatibility (used by existing code) ──────────────────

constexpr int PACK_WIDTH = 10;
constexpr uint32_t MASK_3BIT = 0x7u;

__host__ __device__ __forceinline__
int packed_words(int n) {
    return (n + PACK_WIDTH - 1) / PACK_WIDTH;
}

}  // namespace manthanquant
