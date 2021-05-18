# NMF-FPGA-accelerator-with-logarithmic-approximation

# 0.Before start: 
	The softwares are Python3.7, tensorflow2.1.0, sklearn0.23.2, Vivado design suite 2019.2, and putty.
	There are four folders of code. Python_train_test and Python_addnoise working in Python 3.7, Cpp_check works in C++ and C_Vivado contains code to run in Vivado design suite.

# 1.Brief instruction if you are familiar with the project
	1.1Route to reproduce the results: 1.Python_train_test(train)->2.Cpp_check->3.C_vivado->4.Python_train_test(test). With the noise added in Python_addnoise, repeat 2.Cpp_check->3.C_vivado->4.Python_train_test(test).
	1.2 Python code to train and test model with sklearn should directly work. Other codes require files generated by users so the directory needs to be manually changed to reproduce results in your computers.

# 2.Steps
## 2.1'Python_train_test' trains NMF and KNN. 
		2.1.1NMF and KNN model for MNIST and Fashion-MNIST can be trained and saved in the same code, with just input dataset changed.
		2.1.2Matrix W generated by sklearn or 2.2 and 2.3 can be classified with saved KNN model. 
## 2.2'Cpp_check' implement NMF algorithm toinference the transformed matrix W by C++ program in x86 system.
		2.2.1 It should produce results than be the same or almost the same with result by sklearn.
		2.2.2 This program is for fastly checking the algorithm and code in C++, while getting results from 2.3 consumes much time.
		2.2.3 However, this part is just for checking results in floating-point. Fixed-point and logarithmic approximation are done only in 2.3.
## 2.3'C_Vivado' designs HLS up to accelerate the computation of NMF to get matrix W and generate results in Zybo-7.
		### -2.3.1 vivado_hls contails 3 folders for 3 kinds of data type. Compute_u.cpp and Compute_u.h are source files and Compute_u_test.cpp is the test bench file.
			#### 2.3.1.1'Floating-point' is for desing HLS IP to accelerate NMF algorithm with floating-point arthimetic.
			#### 2.3.1.2'Fixed-point' is for desing HLS IP to accelerate NMF algorithm with fixed-point arthimetic. There are only 2 part different from the code for floating-point: one part is the datatype in compute_u.h, the other part is eps to avoid dividing 0 is no longer 10e-9 but 10e-2. The reason is that fixed point strategy used will transform 10e-9 to 0 due to limited precision.
			#### 2.3.1.3'Logarithmic' is for desing HLS IP to accelerate NMF algorithm with logarithmic approximation.
		### 2.3.2 Vitis is for c code runs in embedded system, more exactly, in Zybo-7. It contains code such as reading data from SD card and driving the HLS IP designed in 2.3.1 to do NMF computaion.	
## 2.4'Python_train_test' process the results generated in 2.2 and 2.3 to get accuracy.
## 2.5Add noise to input dataset in 'Python_addnoise', and repeat 2.2-2.4 with same model in 2.1. 
