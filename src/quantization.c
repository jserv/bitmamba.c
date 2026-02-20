#include <stdint.h>

#include "bitmamba.h"

/*
 * Weight unpacking LUT: maps 1 packed byte (4 x 2-bit weights) to 4 int8.
 * BitNet 1.58b encoding: 00->-1, 01->0, 10->+1, 11->+1 (clipped)
 *
 * Clipping 11->+1 matches the AVX2 _mm256_sign_epi8 semantics in the
 * original C++ code, where sign(x, w) treats any w>0 as +x.
 */
int8_t UNPACK_LUT[256][4];

/*
 * T-MAC Symmetric LUTs (g=4)
 *
 * For ternary weights {-1, 0, +1}, the dot product can be computed as:
 *   sum = lut[P] - lut[N]
 * where P = bitmask of positions with w=+1, N = bitmask of w=-1,
 * and lut[mask] = sum of x[i] for bits set in mask.
 *
 * This replaces 4 muls + 3 adds with 2 lookups + 1 sub per 4 weights.
 */
uint8_t TMAC_P_MASK[256]; /* P[i]=1 iff weight[i] = +1 */
uint8_t TMAC_N_MASK[256]; /* N[i]=1 iff weight[i] = -1 */

static inline int8_t decode_weight(int bits)
{
    int val = bits - 1;
    return (int8_t) (val > 1 ? 1 : val);
}

void init_lut(void)
{
    for (int i = 0; i < 256; i++) {
        UNPACK_LUT[i][0] = decode_weight(i & 0x03);
        UNPACK_LUT[i][1] = decode_weight((i >> 2) & 0x03);
        UNPACK_LUT[i][2] = decode_weight((i >> 4) & 0x03);
        UNPACK_LUT[i][3] = decode_weight((i >> 6) & 0x03);

        /* T-MAC masks: extract P (positive) and N (negative) for 4 weights */
        uint8_t p_mask = 0, n_mask = 0;
        for (int j = 0; j < 4; j++) {
            int8_t w = decode_weight((i >> (j * 2)) & 0x03);
            if (w == 1)
                p_mask |= (1 << j);
            if (w == -1)
                n_mask |= (1 << j);
        }
        TMAC_P_MASK[i] = p_mask;
        TMAC_N_MASK[i] = n_mask;
    }
}
