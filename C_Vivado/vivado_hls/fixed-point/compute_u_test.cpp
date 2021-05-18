#include <stdio.h>
#include <iostream>
#include <fstream>
#include "compute_u.h"
#include <time.h>
using namespace std;
//******************below values and matrices
#define ite_maxnum 50	// max iteration nmumber
#define normalize_constant 16
#define  ImageNumber 10000 // 1 to process just one figure, 10000 to process all image
int control_sig[1] = {};
int control_sig_slite;
float xread[M_row][N_col] = { 0 };	//define matrix xtest:100，784 ： 100figure,one figure wiht 784pixels； exported from mnist
float vtread[K_comp][N_col] = { 0 };	// matrix vt:  30，784； calculated by training steps in python
//below are values calculated  in cpp
float vread[N_col][K_comp] = { 0 };	//matrix v=vt.T; transpose of vt
float b_u[M_row][K_comp] = { 0 };	//matrix b_u =xtest *v ; fixed after calculated
float uread[M_row][K_comp] = { 0 };	//*************this is what we want**************; import a random matrix as initial values;updated  by   uread[i][j] = uread[i][j] *b_u[i][j] / a_u[i][j];
float uvt[M_row][N_col] = { 0 };	//matrix uvt=u *vt ; updated while u updated;
float a_u[M_row][K_comp] = { 0 };	// matrix a_u=uvt* v; updated while u updated
float tempb_u;
//******************below functions
int write_u()
{
	std::ofstream outu(("/home/yz/myprojects/Seafile_sourcecode_Yizhi/C_Vivado/vivado_hls/floating-point/0510.bindata"), std::ios::binary);
	outu.write((char*) &uread, sizeof uread);
	outu.close();
	return 0;
}
//import 10000figures with 784 pixels: stored in xread, size:10000 * 784
int addx_test()
{
	ifstream inputx("/home/yz/CLionProjects/datatype/xtestfloat3210k.bin", ios:: in | ios::binary);
	inputx.read((char*) &xread, sizeof xread);
	cout << inputx.gcount() << " bytes read x data\n";
	inputx.close();
	return 0;
}
//import trained components: stored in vtread, size:30 * 784
int add_vt()
{
	ifstream inputv("/home/yz/CLionProjects/datatype/H_componentsfloat3210k.bin", ios:: in | ios::binary);
	inputv.read((char*) &vtread, sizeof vtread);
	cout << inputv.gcount() << " bytes read v components\n";
	inputv.close();
	return 0;
}
//import initial values of matrx u; a random matrix :size 100 * 30
int add_uread()
{
	ifstream inputu("/home/yz/CLionProjects/datatype/W_randomfloat3210k.bin", ios:: in | ios::binary);
	inputu.read((char*) &uread, sizeof uread);
	inputu.close();
	return 0;
}
// while vt imported from file, get v=vt.T; vt size:30 * 784, v size:784 * 30
int getvread()
{
	for (int i = 0; i < N_col; i++)
	{
		for (int j = 0; j < K_comp; j++)	//
			vread[i][j] = vtread[j][i];
	}
	return 0;
}
//calculate matrix b_u =xtest *v ; fixed after calculated once;
int b_ucompute()
{
	for (int i = 0; i < M_row; i++)
	{
		for (int j = 0; j < K_comp; j++)	//
		{
			tempb_u = 0;
			for (int k = 0; k < N_col; k++)
			{
				tempb_u += xread[i][k] *vread[k][j];
			}
			b_u[i][j] = tempb_u;
		}
	}
	return 0;
}
//divide input matrix by constant value, for example, 16
int normalize_x()
{
	for (int i = 0; i < M_row; i++)
	{
		for (int j = 0; j < N_col; j++)
		{
			xread[i][j] = xread[i][j] / normalize_constant;
		}
	}
	return 0;
}
//divide weight matrix by constant value, for example, 16
int normalize_vt()
{
	for (int i = 0; i < N_col; i++)
	{
		for (int j = 0; j < K_comp; j++)	//
			vtread[j][i] = vtread[j][i] / normalize_constant;
	}
	return 0;
}

int main()
{
	add_uread();	// import initial values of matrx u; a random matrix :size 10000 * 30
	addx_test();	//import matrix xtest:10,000，784 ： 10,000figure,one figure wiht 784pixels； exported from mnist
	add_vt();	//import weight matrix: stored in vtread, size:30 * 784
	normalize_x();//divide input matrix by constant value, for example, 16
	normalize_vt();//divide weight matrix by constant value, for example, 16
	getvread();	//while vt imported from file, get v=vt.T; vt size:30 * 784, v size:784 * 30
	b_ucompute();	//calculate matrix b_u =xtest *v ; fixed after calculated once;
	//print sth to observe
	cout << "before done  \r\n";
	cout << uread[0][0] << "u00 \r\n";
	cout << uread[9999][0] << "u9999  \r\n";
	//configure mode below
	control_sig_slite = 0;//set the mode to be configuration mode
	compute_u((uread[0]), vtread[0], \
		(b_u[0]), \
		ite_maxnum, control_sig_slite, N_col, K_comp);
	//computation mode below
	control_sig_slite = 1;//set the mode to be cpmputation mode
	time_t currentTm = time(NULL);
	puts(asctime(localtime(&currentTm)));
	for (int i = 0; i < ImageNumber; i = i + 1)
	{
		compute_u((uread[i]), vtread[0], \
			(b_u[i]), \
			ite_maxnum, control_sig_slite, N_col, K_comp);
	}
	time_t currentTm2 = time(NULL);
	puts(asctime(localtime(&currentTm2)));
	write_u();//save the matrix after NMF
	//print sth to observe
	cout << uread[0][0] << "u00 compute\r\n";
	cout << uread[9999][0] << "u9999  \r\n";
	return 0;
}

