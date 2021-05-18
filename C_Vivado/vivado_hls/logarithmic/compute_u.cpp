//
// Created by yz on 10.05.21.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "compute_u.h"
using namespace std;
#define mrow_test 1//process 1 image at one time. If the HW resource is infinity, can be 10,000 or higher
//The PU can store weights in a scalable way, matrix smaller than 50*1600, for example 30*784, can be stored.
#define K_comp_store 50//due the the BRAM size, this is the maximum size K of weight matrix[K][N] to store in BRAM
#define N_col_store 1600//due the the BRAM size, this is the maximum size N of weight matrix[K][N] to store in BRAM
my_type tempuvt;//temp value, just for calculation
my_type tempa_u;//temp value, just for calculation
my_type u_inner[mrow_test][K_comp_store] = { 0 };	// transformed matrix after NMF, called "W"(in thesis) or "u"(in code), updated in each iteration
my_type vt_inner[K_comp_store][N_col_store] = { 0 };//weight matrix, called "H"(in thesis) or "vt"(in code)
my_type b_u_inner[mrow_test][K_comp_store] = { 0 };//same size as matrix 'u', matrix product: b_u = X*vt.T = X*H.T
my_type uvt_inner[mrow_test][N_col_store] = { 0 };//same size as matrix 'x', matrix product: uvt = u*vt, updated in each iteration
my_type a_u_inner[mrow_test][K_comp_store] = { 0 };//same size as matrix 'u',  matrix product: a_u = uvt*vt.T, updated in each iteration
my_type au_avoid0 = 0.000000001;// avoid dividing by zero
//below values are special for logarithmic approximation
#define pruningbits 5	//for MNIST's exponent part, 5bit content  + 1bit sign is enough
ap_int < (pruningbits + 1) > vt_inner_exp[K_comp_store][N_col_store] = { 0 };// weight matrix in 8-bit or less bits
ap_int<8> vt_inner_exp_8bit = 0;
ap_int < (pruningbits + 1) > b_exponent;
ap_int < (pruningbits + 1) > b_exponent_au;
ap_int<8> a_exponent;
ap_uint<64> a_mantissa;
ap_int<8> ab_exponent;
ap_uint<64> ab_mantissa;
ap_int<8> exponent;
Data32 ab;
ap_uint<64> sum_magnitude;
int ite_maxnum;
int N_col_hw_inner;
int K_comp_hw_inner;

int compute_u(float *uread_1d, float *vtread_1d, \
	float *b_u_1d, \
	int ite_maxnum_arr, int control_sig, int N_col_hw, int K_comp_hw)
{
	int control_sig_inner = control_sig;//control signal for configuration mode or computation mode
/*
* Below define HLS AXI port
*/
#pragma HLS INTERFACE s_axilite port =return
# pragma HLS INTERFACE s_axilite port = control_sig
# pragma HLS INTERFACE s_axilite port = ite_maxnum_arr
# pragma HLS INTERFACE s_axilite port = N_col_hw
# pragma HLS INTERFACE s_axilite port = K_comp_hw
# pragma HLS INTERFACE m_axi depth = 300000 port = uread_1d offset = slave bundle = uread_1d
# pragma HLS INTERFACE m_axi depth = 23520 port = vtread_1d offset = slave bundle = vtread
# pragma HLS INTERFACE m_axi depth = 300000 port = b_u_1d offset = slave bundle = b_u
	/*
	* Below cofig mode
	*/
	if (!control_sig_inner)	// !0=1, read weight，cofig mode
	{
		Data32 bdata32config;
		ite_maxnum = ite_maxnum_arr;// read iteration number
		N_col_hw_inner = N_col_hw;// read column of input matrix X, which is also the column number of weight matrix vt(also called 'H')
		K_comp_hw_inner = K_comp_hw;// read rank number of NMF, which is also the row number of weight matrix
		for (int k = 0; k < K_comp_hw_inner; k++)
			readvt: for (int j = 0; j < N_col_hw_inner; j++)
			{
#pragma HLS PIPELINE
				bdata32config.f32 = vtread_1d[k *N_col_hw_inner + j];	//read data in the form of floating point
				vt_inner_exp_8bit = DATA32_GET_EXPONENT(bdata32config.u32);// get 8-bit exponent part
				if (vt_inner_exp_8bit < -(pow(2, pruningbits) - 1))// if 8-bit exponent part exceed the minimum values can be expressed, truncate it. One evidence for that '5+1'bit is enough for MNIST is that 'pruningbits' set to be 5 or 7 doesn't have influences.
				{
					vt_inner_exp[k][j] = -(pow(2, pruningbits) - 1);
				}
				else// if 8-bit exponent part NOT exceed the minimum values can be expressed, keep it.
				{
					vt_inner_exp[k][j] = vt_inner_exp_8bit;
				}
			}
	}

	if (control_sig_inner)
	{
		for (int i = 0; i < mrow_test; i++)
			readu: for (int k = 0; k < K_comp_hw_inner; k++)
			{
#pragma HLS PIPELINE
				u_inner[i][k] = uread_1d[k];
				b_u_inner[i][k] = b_u_1d[k];
			}

		// *****************below uvt ****5*************
		ite_: for (int ite = 0; ite < ite_maxnum; ite++)
		{
			uvt_1_mrow: for (int i = 0; i < mrow_test; i++)
			{
				uvt_2_ncol: for (int j = 0; j < N_col_hw_inner; j++)
				{
#pragma HLS PIPELINE
					sum_magnitude = 0;
					Data32 adata32;
					uvt_3_k: for (int k = 0; k < K_comp_hw_inner; k++)
					{
#pragma HLS PIPELINE
						/*Below is to achieve this function:
						 * tempuvt = 0;
						 for (int k = 0; k < K_comp_hw_inner; k++)
						{
							tempuvt += u_inner[i][k] *vt_inner[k][j];
						}
						uvt_inner[i][j] = tempuvt;
						 */
						adata32.f32 = u_inner[i][k];
						if (adata32.u32 != 0 &vt_inner_exp[k][j] != (-(pow(2, pruningbits) - 1)))// if values in input matrix and weight matrix are not zeros.
						{
							a_exponent = DATA32_GET_EXPONENT(adata32.u32);// get exponent part of adata
							a_mantissa = DATA32_GET_MANTISSA(adata32.u32);// get mantissa part of adata
							ab_exponent = a_exponent + vt_inner_exp[k][j];// floating-point multiplication is exponent part addition
							ab_mantissa = a_mantissa << 32;//1.This operation is to avoid losing info by ">>"  2.no operation of weight's mantissa.
							if (ab_exponent < 0)
							{
								sum_magnitude += ab_mantissa >> -ab_exponent;
							}
							else
							{
								sum_magnitude += ab_mantissa << ab_exponent;
							}
						}
					}	//uvt_3 end
					Data32 sum;
					sum.f32 = 0;
					sum_magnitude = sum_magnitude >> 8;
					if (sum_magnitude)
					{
						NORMALIZE_SUM: for (exponent = -8; !(0xff80000000000000 &(sum_magnitude)); exponent++)
						{
#pragma HLS pipeline
							sum_magnitude <<= 1;
						}
						sum.u32 = BUILD_FLOAT(0, -exponent, sum_magnitude>> 32);//build data with exponent and sum_mag
					}
					uvt_inner[i][j] = sum.f32;
				}	//uvt 2 ends
			}	//uvt1 ends

			/*
			* Below compute matrix a_u
			*/
			au_1_Mrow: for (int i = 0; i < mrow_test; i++)
			{
				au_2_Kcomp: for (int j = 0; j < K_comp_hw_inner; j++)
				{
					ap_uint<64> ausum_magnitude = 0;
					Data32 adata32_au;
					au_3_Ncol: for (int k = 0; k < N_col_hw_inner; k++)
					{
#pragma HLS PIPELINE
						adata32_au.f32 = uvt_inner[i][k];
						if (adata32_au.u32 != 0 &vt_inner_exp[j][k] != (-(pow(2, pruningbits) - 1)))
						//   if(adata32_au.u32!=0 &vtread_1d[k*N_col_hw_inner+j]!=0)
						{
							a_exponent = DATA32_GET_EXPONENT(adata32_au.u32);
							a_mantissa = DATA32_GET_MANTISSA(adata32_au.u32);
							ab_exponent = a_exponent + vt_inner_exp[j][k];
							ab_mantissa = a_mantissa << 32;
							if (ab_exponent < 0)
							{
								ausum_magnitude += ab_mantissa >> -ab_exponent;
							}
							else
							{
								ausum_magnitude += ab_mantissa << ab_exponent;
							}
						}	//if zero end

					}	//au_3_loop end
					Data32 sum_au;
					sum_au.f32 = 0;
					ausum_magnitude = ausum_magnitude >> 8;
					if (ausum_magnitude)
					{
					 			//  #pragma HLS pipeline
						NORMALIZE_SUMau: for (exponent = -8; !(0xff80000000000000 & ausum_magnitude); exponent++)
						{
#pragma HLS pipeline
							ausum_magnitude <<= 1;
						}
						sum_au.u32 = BUILD_FLOAT(0, -exponent, ausum_magnitude>> 32);
					}
					a_u_inner[i][j] = sum_au.f32;
				}	//au2 ends
			}	//au1 ends
			/*
			* Below compute u
			*/
			u_inner_1_Mrow: for (int i = 0; i < mrow_test; i++)
			{
				u_inner_2_Kcomp: for (int j = 0; j < K_comp_hw_inner; j++)
				{
#pragma HLS PIPELINE
					u_inner[i][j] = u_inner[i][j] *b_u_inner[i][j] / (a_u_inner[i][j] + au_avoid0);
				}
			}
		}
		/*
		* Below update u in DRAM by AXI-Master bus
		*/
		uread_1_Mrow: for (int i = 0; i < mrow_test; i++)
		{
			uread_2_Kcomp: for (int j = 0; j < K_comp_hw_inner; j++)
			{
#pragma HLS PIPELINE
				uread_1d[j] = u_inner[i][j];
			}
		}
	}
	return 0;

}

