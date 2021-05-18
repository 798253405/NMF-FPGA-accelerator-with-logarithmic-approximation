#define M_row 10000	//row of input matrix
#define N_col 784	//col of input matrix
#define K_comp 30	//rank  of NMF

typedef float my_type;//define data type
int compute_u(float *uread_1d, float *vtread_1d, \
	float *b_u_1d, \
	int ite_maxnum_arr, int control_sig, int N_col_hw, int K_comp_hw);

