#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;
#define M_ROW 10000	//row number
#define N_COL 784	//col number
#define K_COMP 30	//rank of nmf
#define ite_maxnum 50	// max iteration nmumber
//X=W*H: W is replaced with the name u, H is replaced with the name vt -> X=u*vt
//below are 2 imported values
float NMF_xread[M_ROW][N_COL] = { 0 };	//define matrix xtest:10000，784 ： 10000figure,one figure wiht 784pixels；
float NMF_vtread[K_COMP][N_COL] = { 0 };	// matrix vt:  30，784； calculated by training steps in python
//below are values calculated  in cpp
float NMF_vread[N_COL][K_COMP] = { 0 };	//matrix v=vt.T; transpose of vt
float NMF_b_u[M_ROW][K_COMP] = { 0 };	//matrix NMF_b_u =xtest *v ; fixed after calculated once
float NMF_uread[M_ROW][K_COMP] = { 0 };	//*************this is what we want**************; import a random matrix as initial values;updated  by   NMF_uread[i][j] = NMF_uread[i][j] *NMF_b_u[i][j] / NMF_a_u[i][j];
float NMF_uvt[M_ROW][N_COL] = { 0 };	//matrix NMF_uvt=u *vt ; updated while u updated;
float NMF_a_u[M_ROW][K_COMP] = { 0 };	// matrix NMF_a_u=NMF_uvt* v; updated while u updated
float eps = 0.000000001;	//eps is a small value to avoid dividing by zero
/******
 import 100figures with 784 pixels: stored in NMF_xread, size:10000 * 784
******/
int addx_test()
{
    ifstream inputx("/home/yz/CLionProjects/datatype/xtestfloat3210k.bin", ios:: in | ios::binary);
    inputx.read((char*) &NMF_xread, sizeof NMF_xread);
    cout << inputx.gcount() << " bytes read x data\n";
    inputx.close();
    return 0;
}
/******
 import trained components: stored in NMF_vtread, size:30 * 784
******/
int add_vt()
{
    ifstream inputv("/home/yz/CLionProjects/datatype/H_componentsfloat3210k.bin", ios:: in | ios::binary);
    inputv.read((char*) &NMF_vtread, sizeof NMF_vtread);
    cout << inputv.gcount() << " bytes read v components\n";
    inputv.close();
    return 0;
}
/******
 import initial values of matrx u; a random matrix :size 10000 * 30
******/
int add_uread()
{
    ifstream inputu("/home/yz/CLionProjects/datatype/W_randomfloat3210k.bin", ios:: in | ios::binary);
    inputu.read((char*) &NMF_uread, sizeof NMF_uread);
    cout << inputu.gcount() << " bytes read u initial random\n";
    inputu.close();
    return 0;
}
/******
 while vt imported from file, get v=vt.T; vt size:30 * 784, v size:784 * 30
******/
int getvread()
{
    for (int i = 0; i < N_COL; i++)
    {
        for (int j = 0; j < K_COMP; j++)
            NMF_vread[i][j] = NMF_vtread[j][i];
    }
    return 0;
}
/******
calculate matrix NMF_b_u =xtest *v ; fixed after calculated;
******/
int b_ucompute()
{
    for (int i = 0; i < M_ROW; i++)
    {
        for (int j = 0; j < K_COMP; j++)
        {
            float tempb_u = 0;
            for (int k = 0; k < N_COL; k++)
            {
                tempb_u += NMF_xread[i][k] *NMF_vread[k][j];
            }
            NMF_b_u[i][j] = tempb_u;
        }
    }
    return 0;
}
int main()
{
    add_uread();	// import initial values of matrx u; a random matrix :size 100 * 30
    addx_test();	//import matrix xtest:100，784 ： 100figure,one figure wiht 784pixels； exported from mnist
    add_vt();	//import trained components: stored in NMF_vtread, size:30 * 784
    getvread();	//while vt imported from file, get v=vt.T; vt size:30 * 784, v size:784 * 30
    b_ucompute();	//calculate matrix NMF_b_u =xtest *v ; fixed after calculated;
    cout << NMF_b_u[1][1] << "b_U11" << '\n';	//not necessary;just print sth to see;any value is fine;
    for (int ite = 0; ite < ite_maxnum; ite++)
    {
        //First step, get NMF_uvt=u*v.t: size of 10000 * 784
        for (int i = 0; i < M_ROW; i++)
            for (int j = 0; j < N_COL; j++)
            {
                float tempuvt = 0;
                for (int k = 0; k < K_COMP; k++)
                {
                    tempuvt += NMF_uread[i][k] *NMF_vtread[k][j];
                }
                NMF_uvt[i][j] = tempuvt;
            }
        //2  a_U=u*vt *v =NMF_uvt*v: size of 10000 * 30
        for (int i = 0; i < M_ROW; i++)
            for (int j = 0; j < K_COMP; j++)
            {
                float tempa_u = 0;
                for (int k = 0; k < N_COL; k++)
                {
                    tempa_u += NMF_uvt[i][k] *NMF_vread[k][j];
                }
                NMF_a_u[i][j] = tempa_u;
            }
        for (int i = 0; i < M_ROW; i++)
            for (int j = 0; j < K_COMP; j++)
            { 	{ 		NMF_uread[i][j] = NMF_uread[i][j] *NMF_b_u[i][j] / (NMF_a_u[i][j] + eps);
                }
            }
    }
    /******
    save matrix u ; can be used for knn in python
    ******/
    std::ofstream outu(("/home/yz/CLionProjects/0510_seafile/stavoidoverwrite.bindata"), std::ios::binary);//change the directory to your directory and name
    outu.write((char*) &NMF_uread, sizeof NMF_uread);
    outu.close();
}
