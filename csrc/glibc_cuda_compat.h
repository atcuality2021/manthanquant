/*
 * Workaround for glibc 2.42 (Ubuntu 25.10) vs CUDA 12.8/12.9 conflict.
 * glibc declares cospi/sinpi/rsqrt with __THROW but CUDA declares them without.
 * We prevent glibc from declaring them by pre-defining the guard macro.
 */
#ifdef __CUDACC__
#include <features.h>
/* Override the IEC_60559 feature test macro AFTER features.h sets it */
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT
#define __GLIBC_USE_IEC_60559_FUNCS_EXT 0
#endif
