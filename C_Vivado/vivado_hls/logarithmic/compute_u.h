#include <ap_int.h>
#define M_row 10000 //row
#define N_col 784 //col
#define K_comp 30 //rank
/*
 *Below are bit operations to get exponent part, mantissa part and build 32-bit data with these 2 parts.
 */
#define DATA32_GET_EXPONENT(x) ((0xFF & ((x) >> 23)) - 0x7F)
#define DATA32_GET_MANTISSA(x) (0x00800000 | ((0x7FFFFF) & (x)))
#define BUILD_FLOAT(s, exponent, mantissa) ((0x80000000 & ((s) << 31)) | (0x7f800000 & (((exponent) + 0x7f) << 23)) | ((mantissa) & 0x7FFFFF))
typedef union
  {
      float f32;
      unsigned int u32;
  }Data32;

typedef float my_type;
int compute_u(float* uread_1d,float* vtread_1d,\
			float*  b_u_1d,\
			int ite_maxnum_arr,int control_sig,int N_col_hw,int K_comp_hw );

