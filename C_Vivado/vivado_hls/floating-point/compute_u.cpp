// Created by yz on 10.05.21.
#include <stdio.h>
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
int ite_maxnum;//iteration number, configured by PS
int N_col_hw_inner;//actual column number of weight matrix. scalable for different settings. In this example, it is 784.
int K_comp_hw_inner;//actual row number of weight matrix. scalable for different settings. In this example, it is 30.
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
	if (!control_sig_inner)
	{
		ite_maxnum = ite_maxnum_arr;// read iteration number
		N_col_hw_inner = N_col_hw;// read column of input matrix X, which is also the column number of weight matrix vt(also called 'H')
		K_comp_hw_inner = K_comp_hw;// read rank number of NMF, which is also the row number of weight matrix
		for (int k = 0; k < K_comp_hw_inner; k++)
			readvt: for (int j = 0; j < N_col_hw_inner; j++)
			{
#pragma HLS PIPELINE
				vt_inner[k][j] = vtread_1d[k *N_col_hw_inner + j];// read values in weight matrix and store in BRAM
			}
	}//cofig mode ends here
	/*
	 * Below computation mode
	 */
	if (control_sig_inner)
	{	/*
	 	 * Below read data to be computed
	 	 */
		for (int i = 0; i < mrow_test; i++)
			readu: for (int k = 0; k < K_comp_hw_inner; k++)
			{
#pragma HLS PIPELINE
				u_inner[i][k] = uread_1d[k];
				b_u_inner[i][k] = b_u_1d[k];
			}

		ite: for (int ite = 0; ite < ite_maxnum; ite++)
		{
			/*
			 * Below compute matrix uvt
			 */
			uvt1_morw: for (int i = 0; i < mrow_test; i++)
			{
				uvt2_ncol: for (int j = 0; j < N_col_hw_inner; j++)
				{
#pragma HLS PIPELINE
					tempuvt = 0;
					uvt3_k: for (int k = 0; k < K_comp_hw_inner; k++)
					{
#pragma HLS PIPELINE
						tempuvt += u_inner[i][k] *vt_inner[k][j];
					}
					uvt_inner[i][j] = tempuvt;
				}
			}
			/*
			 * Below compute matrix a_u
			 */
			a_u1_mrow: for (int i = 0; i < mrow_test; i++)
			{
				a_u2_kcomp: for (int j = 0; j < K_comp_hw_inner; j++)
				{
				 		//#pragma HLS PIPELINE// if not commented, requires too much HW resource and synthesis time dramatically increased(usually  hours)
					tempa_u = 0;
					a_u3_ncol: for (int k = 0; k < N_col_hw_inner; k++)
					{
#pragma HLS PIPELINE
						tempa_u += uvt_inner[i][k] *vt_inner[j][k];
					}
					a_u_inner[i][j] = tempa_u;
				}
			}
			/*
			 * Below compute u
			 */
			for (int i = 0; i < mrow_test; i++)
			{
				for (int j = 0; j < K_comp_hw_inner; j++)
				{
#pragma HLS PIPELINE
					u_inner[i][j] = u_inner[i][j] *b_u_inner[i][j] / (a_u_inner[i][j] + au_avoid0);
				}
			}
		}
		/*
		 * Below update u in DRAM by AXI-Master bus
		 */
		for (int i = 0; i < mrow_test; i++)
		{
			u_writeout: for (int j = 0; j < K_comp_hw_inner; j++)
			{
#pragma HLS PIPELINE
				uread_1d[j] = u_inner[i][j];
			}
		}

	}
	return 0;
}

