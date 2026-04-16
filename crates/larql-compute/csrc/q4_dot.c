// Fused Q4_0 × Q8_0 dot product.
//
// ARM path: uses vdotq_s32 (dotprod extension) for ~0.95 ms per 105 MB matrix.
// Scalar path: portable C, works on x86_64 (Windows/Linux), RISC-V, etc.
//
// Called from Rust via FFI — no dependencies beyond stdint/stddef.

#include <stdint.h>
#include <stddef.h>
#include <string.h>   /* memcpy — standard replacement for __builtin_memcpy on MSVC */

// Helper: decode IEEE 754 f16 to f32.
// Defined unconditionally so both ARM and scalar paths can use it.
static inline float decode_f16(uint16_t h) {
    uint32_t sign     = (h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa =  h & 0x3FFu;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        exponent = 1;
        while (!(mantissa & 0x400u)) { mantissa <<= 1; exponent--; }
        mantissa &= 0x3FFu;
        exponent += 127u - 15u;
    } else if (exponent == 31u) {
        exponent = 255u;
    } else {
        exponent += 127u - 15u;
    }

    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// ARM aarch64 path — vdotq_s32 intrinsics
// ─────────────────────────────────────────────────────────────────────────────
#if defined(__aarch64__)
#include <arm_neon.h>

// Fused Q4_0 × Q8_0 dot product for one row.
//
// q4_row:    packed Q4_0 blocks (18 bytes each: 2B f16 scale + 16B nibbles)
// q8:        pre-quantized Q8_0 input vector (int8)
// q8_scales: per-block scales for the Q8 input (float32)
// blocks:    number of Q4_0 blocks (hidden_size / 32)
static float q4_q8_dot_neon(
    const uint8_t* q4_row,
    const int8_t*  q8,
    const float*   q8_scales,
    size_t         blocks
) {
    float acc = 0.0f;
    const int8x16_t offset  = vdupq_n_s8(8);
    const uint8x16_t mask_lo = vdupq_n_u8(0x0Fu);

    for (size_t b = 0; b < blocks; b++) {
        const uint8_t* block = q4_row + b * 18;

        uint16_t scale_bits  = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
        float combined_scale = decode_f16(scale_bits) * q8_scales[b];

        const uint8_t* quants = block + 2;
        const int8_t*  q8_ptr = q8 + b * 32;

        uint8x16_t raw = vld1q_u8(quants);

        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), offset);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)),    offset);

        // Interleave: [lo0,hi0,lo1,hi1,...] matches sequential Q8 layout
        int8x16_t q4_0 = vzip1q_s8(lo, hi);
        int8x16_t q4_1 = vzip2q_s8(lo, hi);

        int8x16_t q8_0 = vld1q_s8(q8_ptr);
        int8x16_t q8_1 = vld1q_s8(q8_ptr + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t isum = vdupq_n_s32(0);
        isum = vdotq_s32(isum, q4_0, q8_0);
        isum = vdotq_s32(isum, q4_1, q8_1);
        acc += (float)vaddvq_s32(isum) * combined_scale;
#else
        // Fallback for ARM without dotprod extension
        int16x8_t prod0 = vmull_s8(vget_low_s8(q4_0),  vget_low_s8(q8_0));
        int16x8_t prod1 = vmull_s8(vget_high_s8(q4_0), vget_high_s8(q8_0));
        int16x8_t prod2 = vmull_s8(vget_low_s8(q4_1),  vget_low_s8(q8_1));
        int16x8_t prod3 = vmull_s8(vget_high_s8(q4_1), vget_high_s8(q8_1));
        int16x8_t sum16 = vaddq_s16(vaddq_s16(prod0, prod1), vaddq_s16(prod2, prod3));
        int32x4_t sum32 = vpaddlq_s16(sum16);
        acc += (float)vaddvq_s32(sum32) * combined_scale;
#endif
    }
    return acc;
}

// Fused Q4_0 matvec: scores[num_rows] = Q4[num_rows, hidden] @ Q8_x[hidden]
void q4_0_matvec_c(
    const uint8_t* q4_data,
    const int8_t*  q8_x,
    const float*   q8_scales,
    float*         scores,
    size_t         num_rows,
    size_t         hidden
) {
    size_t blocks_per_row = hidden / 32;
    size_t bytes_per_row  = blocks_per_row * 18;
    for (size_t row = 0; row < num_rows; row++) {
        scores[row] = q4_q8_dot_neon(
            q4_data + row * bytes_per_row,
            q8_x, q8_scales, blocks_per_row);
    }
}

// Fused Q4_0 vecmat: out[hidden] = activation[intermediate] @ Q4[intermediate, hidden]
void q4_0_vecmat_c(
    const float*   activation,
    const uint8_t* q4_data,
    float*         out,
    size_t         intermediate,
    size_t         hidden
) {
    size_t blocks_per_row = hidden / 32;
    size_t bytes_per_row  = blocks_per_row * 18;

    for (size_t j = 0; j < hidden; j++) out[j] = 0.0f;

    for (size_t row = 0; row < intermediate; row++) {
        float act = activation[row];
        if (act > -1e-10f && act < 1e-10f) continue;

        const uint8_t* row_data = q4_data + row * bytes_per_row;

        for (size_t b = 0; b < blocks_per_row; b++) {
            const uint8_t* block = row_data + b * 18;
            uint16_t scale_bits  = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float scale          = decode_f16(scale_bits) * act;
            const uint8_t* quants = block + 2;
            float* o = out + b * 32;

            for (size_t j = 0; j < 16; j++) {
                uint8_t byte = quants[j];
                int lo_v = (int)(byte & 0x0Fu) - 8;
                int hi_v = (int)((byte >> 4) & 0x0Fu) - 8;
                o[j * 2]     += (float)lo_v * scale;
                o[j * 2 + 1] += (float)hi_v * scale;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar path — x86_64, x86, RISC-V, Windows, etc.
// ─────────────────────────────────────────────────────────────────────────────
#else

// Dot product for one Q4_0 row against a Q8_0 vector.
// Nibble layout: byte j → lo nibble = Q4 position 2*j, hi nibble = position 2*j+1.
static float q4_q8_dot_scalar(
    const uint8_t* q4_row,
    const int8_t*  q8,
    const float*   q8_scales,
    size_t         blocks
) {
    float acc = 0.0f;
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t* block = q4_row + b * 18;

        uint16_t scale_bits  = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
        float combined_scale = decode_f16(scale_bits) * q8_scales[b];

        const uint8_t* quants = block + 2;
        const int8_t*  q8_ptr = q8 + b * 32;

        int isum = 0;
        for (size_t j = 0; j < 16; j++) {
            uint8_t byte = quants[j];
            int lo = (int)(byte & 0x0Fu) - 8;
            int hi = (int)((byte >> 4) & 0x0Fu) - 8;
            isum += lo * (int)q8_ptr[j * 2];
            isum += hi * (int)q8_ptr[j * 2 + 1];
        }
        acc += (float)isum * combined_scale;
    }
    return acc;
}

// Fused Q4_0 matvec: scores[num_rows] = Q4[num_rows, hidden] @ Q8_x[hidden]
void q4_0_matvec_c(
    const uint8_t* q4_data,
    const int8_t*  q8_x,
    const float*   q8_scales,
    float*         scores,
    size_t         num_rows,
    size_t         hidden
) {
    size_t blocks_per_row = hidden / 32;
    size_t bytes_per_row  = blocks_per_row * 18;
    for (size_t row = 0; row < num_rows; row++) {
        scores[row] = q4_q8_dot_scalar(
            q4_data + row * bytes_per_row,
            q8_x, q8_scales, blocks_per_row);
    }
}

// Fused Q4_0 vecmat: out[hidden] = activation[intermediate] @ Q4[intermediate, hidden]
void q4_0_vecmat_c(
    const float*   activation,
    const uint8_t* q4_data,
    float*         out,
    size_t         intermediate,
    size_t         hidden
) {
    size_t blocks_per_row = hidden / 32;
    size_t bytes_per_row  = blocks_per_row * 18;

    for (size_t j = 0; j < hidden; j++) out[j] = 0.0f;

    for (size_t row = 0; row < intermediate; row++) {
        float act = activation[row];
        if (act > -1e-10f && act < 1e-10f) continue;

        const uint8_t* row_data = q4_data + row * bytes_per_row;

        for (size_t b = 0; b < blocks_per_row; b++) {
            const uint8_t* block  = row_data + b * 18;
            uint16_t scale_bits   = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float scale           = decode_f16(scale_bits) * act;
            const uint8_t* quants = block + 2;
            float* o = out + b * 32;

            for (size_t j = 0; j < 16; j++) {
                uint8_t byte = quants[j];
                int lo_v = (int)(byte & 0x0Fu) - 8;
                int hi_v = (int)((byte >> 4) & 0x0Fu) - 8;
                o[j * 2]     += (float)lo_v * scale;
                o[j * 2 + 1] += (float)hi_v * scale;
            }
        }
    }
}

#endif
