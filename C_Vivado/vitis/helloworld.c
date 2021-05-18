#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "ff.h"
#include "xdevcfg.h"
#include "xcompute_u.h"
#include "xparameters.h"
#include "timer.h"
#include "sleep.h"

static FATFS fatfs;	//// <==== library
#define M_row 10000	//row
#define N_col 784	//col
#define K_comp 30	//rank
#define ite_maxnum 10	// max iteration nmumber
#define normalize_constant 16

XCompute_u XCopmputu;
XCompute_u_Config * ExamplePtr;

//static char xreadFileName[32] = "X.bin";
static char xreadFileName[32] = "X_5.bin";
static char ureadFileName[32] = "U_random.bin";
static char vtreadFileName[32] = "VT.bin";
static char uwriteFileName[32] = "m0n0_";

int control_sig[1] = { 0 };
float xread[M_row][N_col] = { 0 };
size_t xread_size;
size_t xdata_size = 4 *M_row * N_col;	//
float uread[M_row][K_comp] = { 0 };
size_t uread_size;
size_t uwrite_size;
size_t udata_size = 4 *M_row * K_comp;	//
float vtread[K_comp][N_col] = { 0 };
size_t vtread_size;
size_t vtdata_size = 4 *N_col * K_comp;	//
float vread[N_col][K_comp] = { 0 };
float b_u[M_row][K_comp] = { 0 };
float uvt[M_row][N_col] = { 0 };	//matrix uvt=u *vt ; updated while u updated;
float a_u[M_row][K_comp] = { 0 };	// matrix a_u=uvt* v; updated while u updated
float uoutread[M_row][K_comp] = { 0 };
float eps_avoid0=0.000000001;
//initialize SD card
static int SnnApp_initializeSD(void)
{
	FRESULT rc;
	TCHAR *path = "0:/"; /*Logical drive number is 0 */	///<==== _______ Here you select the partition, I have 0 by default _______
	/*Register volume work area, initialize device */
	rc = f_mount(&fatfs, path, 0);
	if (rc != FR_OK)
	{
		return XST_FAILURE;
	}
	return XST_SUCCESS;
}
/*
 * save result matrix u in SD card
 */
int write_u(void)
{
	FIL fil;
	FRESULT rc;
	rc = f_open(&fil, (char*) uwriteFileName, FA_CREATE_ALWAYS | FA_WRITE);
	if (rc)
	{
		xil_printf("ERROR:write u f_open returned %d\r\n", rc);
		//return XST_FAILURE;
	}
	rc = f_write(&fil, &uread, udata_size, &uwrite_size);
	if (rc)
	{
		return XST_FAILURE;
	}
	print("is writing u 0413 file\n\r");
	f_close(&fil);	//////
	return 0;
}
/*
 * read result matrix u saved in SD card
 */
int outuread(void)
{

	FIL fil;
	FRESULT rc;
	rc = f_open(&fil, (char*) uwriteFileName, FA_READ);
	if (rc)
	{
		xil_printf("ERROR:outureadf_open returned %d\r\n", rc);
	}

	uint16_t sizeoffile = f_size(&fil);
	if (rc == FR_OK)
	{
		rc = f_read(&fil, &uoutread, udata_size, &uread_size);	//////////////<========---
		if (rc)
		{
			return XST_FAILURE;
		}
		print("is reading u file\n\r");
	}
	f_close(&fil);	////////
	return 0;
}
/*
 * read random matrix u as the initialization
 */
int read_u(void)
{
	FIL fil;
	FRESULT rc;
	rc = f_open(&fil, (char*) ureadFileName, FA_READ);
	if (rc)
	{
		xil_printf("ERROR:f_open returned %d\r\n", rc);
	}
	uint16_t sizeoffile = f_size(&fil);
	if (rc == FR_OK)
	{
		rc = f_read(&fil, &uread, udata_size, &uread_size);	//////////////<========---_______ READ FILE ___________
		if (rc)
		{
			return XST_FAILURE;
		}
		print("is reading u file\n\r");
		f_close(&fil);	//////////////<========---_______ CLOSE FILE ___________
	}
	//*/
	return XST_SUCCESS;
}
/*
 * read input matrix x (10,000 figures)
 */
int read_x(void)
{
	FIL fil;
	FRESULT rc;
	rc = f_open(&fil, (char*) xreadFileName, FA_READ);
	if (rc)
	{
		xil_printf("ERROR:f_open returned %d\r\n", rc);
		//return XST_FAILURE;
	}
	//    ASSERT(rc == FR_OK);
	uint16_t sizeoffile = f_size(&fil);
	if (rc == FR_OK)
	{
		rc = f_read(&fil, &xread, xdata_size, &xread_size);	//////////////<========---_______ READ FILE ___________
		//   ASSERT((rc == FR_OK) && (read_size == data_size));
		if (rc)
		{
			return XST_FAILURE;
		}
		print("is reading  x file\n\r");
		f_close(&fil);	//////////////<========---_______ CLOSE FILE ___________
	}
	//*/
	return XST_SUCCESS;
}
/*
 * read weight matrix vt (also called H)
 */
int read_vt(void)
{
	FIL fil;
	FRESULT rc;
	rc = f_open(&fil, (char*) vtreadFileName, FA_READ);
	if (rc)
	{
		xil_printf("ERROR:f_open returned %d\r\n", rc);
	}
	uint16_t sizeoffile = f_size(&fil);
	if (rc == FR_OK)
	{
		rc = f_read(&fil, &vtread, vtdata_size, &vtread_size);	//////////////<========---_______ READ FILE ___________
		if (rc)
		{
			return XST_FAILURE;
		}
		print("is reading  vt file\n\r");
		f_close(&fil);	//////////////<========---_______ CLOSE FILE ___________
	}
	return XST_SUCCESS;
}
/*
 * transpose matrix
 */
int getvread()
{
	for (int i = 0; i < N_col; i++)
	{
		for (int j = 0; j < K_comp; j++)	//
			vread[i][j] = vtread[j][i];
	}
	return 0;
}
/*
 *calculate matrix b_u =xtest *v ; fixed after calculated once;
 */
int b_u_compute()
{
	print("start computing b_u\n\r");
	for (int i = 0; i < M_row; i++)
	{
		for (int j = 0; j < K_comp; j++)	//
		{
			float tempb_u = 0;
			for (int k = 0; k < N_col; k++)
			{
				tempb_u += xread[i][k] *vread[k][j];
			}
			b_u[i][j] = tempb_u;
			//   cout<<b_u[i][j]<<"bu";
		}
	}
	print("done computing b_u\n\r");
	return 0;
}
/*
 *calculate u in ARM cpu
 */
int uoutput_compute()
{
	printf("start computing uoutput\n\r");
	for (int ite = 0; ite < ite_maxnum; ite++)
	{
		printf("ite%d\n\r", ite);
		for (int i = 0; i < M_row; i++)
		{
			for (int j = 0; j < N_col; j++)
			{
				float tempuvt = 0;
				for (int k = 0; k < K_comp; k++)
				{
					tempuvt += uread[i][k] *vtread[k][j];
				}
				uvt[i][j] = tempuvt;
			}
		}
		for (int i = 0; i < M_row; i++)
		{
			for (int j = 0; j < K_comp; j++)
			{
				float tempa_u = 0;
				for (int k = 0; k < N_col; k++)
				{
					tempa_u += uvt[i][k] *vread[k][j];
				}
				a_u[i][j] = tempa_u;
			}
		}
		for (int i = 0; i < M_row; i++)
		{
			for (int j = 0; j < K_comp; j++)
			{
				uread[i][j] = uread[i][j] *b_u[i][j] / (a_u[i][j] + eps_avoid0);
			}
		}
		printf("uread %f \n\r", uread[0][0]);
	}
	print("done computing uoutput\n\r");
	return 0;
}
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
int normalize_vt()
{
	for (int i = 0; i < N_col; i++)
	{
		for (int j = 0; j < K_comp; j++)	//
			vtread[j][i] = vtread[j][i] / normalize_constant;
	}
	return 0;
}

/*
 *calculate u in ARM cpu for 1st figure
 */
int uoutput_compute_onefigure()
{
	for (int ite = 0; ite < ite_maxnum; ite++)
	{
		for (int i = 0; i < 1; i++)
		{
			for (int j = 0; j < N_col; j++)
			{
				float tempuvt = 0;
				for (int k = 0; k < K_comp; k++)
				{
					tempuvt += uread[i][k] *vtread[k][j];
				}
				uvt[i][j] = tempuvt;
			}
		}
		for (int i = 0; i < 1; i++)
		{
			for (int j = 0; j < K_comp; j++)
			{
				float tempa_u = 0;
				for (int k = 0; k < N_col; k++)
				{
					tempa_u += uvt[i][k] *vread[k][j];
				}
				a_u[i][j] = tempa_u;
			}
		}
		for (int i = 0; i < 1; i++)
		{
			for (int j = 0; j < K_comp; j++)
			{
				uread[i][j] = uread[i][j] *b_u[i][j] / (a_u[i][j] + 0.000000001);
			}
		}
		printf("uread %f \n\r", uread[0][0]);
	}
	print("done computing uoutput ONEFIGURE\n\r");
	return 0;
}

int main()
{
	/*
	*Below is to initialize platform
	*/
	init_platform();
	Timer * timer1;
	timer1 = Timer_new(1);
	Timer_start(timer1);
	print("Hello World\n\r");
	/*
	*Below is to to initialize SD card
	*/
	int rc_main;
	rc_main = SnnApp_initializeSD();
	if (XST_SUCCESS != rc_main)
	{
		xil_printf("fail to open SD Card~\n\r");
	}
	else
	{
		xil_printf("success to open SD Card~\n\r");
	}
	for (int noise_amp = 0; noise_amp <= 90; noise_amp = noise_amp + 5)
	{
		Xil_DCacheEnable();
		sprintf(xreadFileName, "X_%d", noise_amp);
		strcat(xreadFileName, ".bin");
		read_x();//import matrix x:10,000，784 ： 10,000figure,one figure wiht 784pixels； exported from mnist
		printf("\nuread00 %f\n", uread[0][0]);
		read_u();// import initial values of matrx u; a random matrix :size 10000 * 30
		printf("\nuread00 %f\n", uread[0][0]);
		read_vt();//import weight matrix: stored in vtread, size:30 * 784
		normalize_x();//divide input matrix by constant value, for example, 16
		normalize_vt();//divide weight matrix by constant value, for example, 16
		getvread();//while vt imported from file, get v=vt.T; vt size:30 * 784, v size:784 * 30
		b_u_compute();//calculate matrix b_u = x *vt.T ; fixed after calculated once;

		/*
		 *Below is to record computing time in ARM CPU
		 */
		/*
  	  	Timer * timer3;
  		timer3 = Timer_new (1);
  		Timer_start(timer3);
  	  	uoutput_compute_onefigure();
		printf ("\nTime3: %f secs\n", Timer_getCurrentTime(timer3));
		Timer_delete(&timer3);
		 */
		Xil_DCacheDisable();
		int _status = XCompute_u_Initialize(&XCopmputu, XPAR_XCOMPUTE_U_0_DEVICE_ID);
		if (_status != XST_SUCCESS)
		{
			xil_printf("XExample_initialize failed\n");
			return XST_FAILURE;
		}
		printf("\n after config uread00 %f  ", uread[0][0]);
		for (int ite_write = 1; ite_write <= ite_maxnum; ite_write = ite_write + 1)
		{
			/*
			*Set the ports in PL for configuration mode
			*/
			XCompute_u_Set_vtread_1d(&XCopmputu, vtread[0]);
			XCompute_u_Set_N_col_hw(&XCopmputu, 784);
			XCompute_u_Set_K_comp_hw(&XCopmputu, 30);
			XCompute_u_Set_ite_maxnum_arr(&XCopmputu, ite_write);
			int control_sig_slite0 = 0;
			XCompute_u_Set_control_sig(&XCopmputu, control_sig_slite0);
			XCompute_u_Start(&XCopmputu);//start
			while (!XCompute_u_IsDone(&XCopmputu)) {}
			read_u();	//initial u to be random
			for (int i = 0; i < M_row; i = i + 1)
			{	/*
				*Set the ports in PL for computation mode
				*/
				XCompute_u_Set_uread_1d_offset(&XCopmputu, uread + i);
				XCompute_u_Set_b_u_1d(&XCopmputu, b_u[i]);
				int control_sig_slite = 1;
				XCompute_u_Set_control_sig(&XCopmputu, control_sig_slite);
				Timer * timer2;
				timer2 = Timer_new(1);
				Timer_start(timer2);
				XCompute_u_Start(&XCopmputu);
				while (!XCompute_u_IsDone(&XCopmputu))
				{
					//HLS IP is computing
				}
				//	printf ("\nTime2_ite%d: %f secs\n",ite_write, Timer_getCurrentTime(timer2));// for 1 iteration
				Timer_delete(&timer2);
			}
			printf("\n after pl uread00 %f  ", uread[0][0]);//print sth to observe, not necessary but useful
			sprintf(uwriteFileName, "m0i%dn%d", ite_write, noise_amp);//get a file name contains iterations and noise amplitude automatically
			printf("uwriteFileName%s", uwriteFileName);//print the name of file to be save
			write_u();// save file in SD card
		}
	}


	printf("\nTime1: %f secs\n", Timer_getCurrentTime(timer1));
	Timer_delete(&timer1);
	printf("\n after pl uread9999 0 %f\n", uread[9999][0]);	//print sth to observe, not necessary but useful
	printf("\nuread00 %f\n", uread[0][0]);//print sth to observe, not necessary but useful
	printf("\nEND\n");	//27.197s	//20//553。36- 537.06	//50-1359.85
	cleanup_platform();
	return 0;

}
