from sklearn.decomposition import NMF
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from joblib import dump, load
#*******load MNIST dataset from tensorflow
MNIST_TRAIN_IMAGENUMBER=60000#60,000 figures in training dataset
MNIST_TEST_IMAGENUMBER=10000#10,000 figures in testing dataset
MNIST_TRAIN_IMAGESIZE=784#image size 28*28=784
MNIST_TEST_IMAGESIZE=784
MNIST_NORMALIZING_CONSTANT=255
Switch_Verify=1 #switch to verify the result in testing dataset
#*******load MNIST dataset from tensorflow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / MNIST_NORMALIZING_CONSTANT, x_test / MNIST_NORMALIZING_CONSTANT#normalizing by 255
x_train=np.resize(x_train,(MNIST_TRAIN_IMAGENUMBER,MNIST_TRAIN_IMAGESIZE))
x_test=np.resize(x_test,(MNIST_TEST_IMAGENUMBER,MNIST_TEST_IMAGESIZE))
#*******save x_test to be read in SD card
x_test_float32=(x_test).astype('float32')
x_test_float32.astype('float32').tofile("X_test_saved.bin")
#*******Save NMF model after trained
nmf = NMF(n_components=30,random_state=0,tol=1e-5,max_iter=1000,l1_ratio=0.0)
nmf.fit(x_train)
dump(nmf, 'MNIST_TrainModel/nmf1e-5_1k.joblib')
NMF_Train_W = nmf.transform(x_train)
NMF_Train_H =nmf.components_
NMF_Train_H.astype('float32').tofile("H_weights_saved.bin")
#*******Save KNN model after trained
knn_nmf = KNeighborsClassifier(n_neighbors=5,n_jobs=12)# n_neighbors=5 is default, n_jobs represents how many cpus are using
knn_nmf.fit(NMF_Train_W,y_train)
dump(knn_nmf, 'MNIST_TrainModel/knn5nmf1e-5_1k.joblib')
#*******check results
if (Switch_Verify):
    nmf = load('MNIST_TrainModel/nmf1e-5_1k.joblib')
    knn_nmf=load('MNIST_TrainModel/knn5nmf1e-5_1k.joblib')
    #Inference data with trained NMF model
    NMF_Test_W = nmf.transform(x_test)
    #Inference data after NMF with trained KNN model
    print('test score:',knn_nmf.score(NMF_Test_W ,y_test))
    #0.9523 is expected
    #sklearn is convenient but has some limitations. For example, cannot set the iteration number for inferencing
#*******Code ends

#Ref1:Phon-Amnuaisuk S. Applying non-negative matrix factorization to classify superimposed handwritten digits[J]. Procedia Computer Science, 2013, 24: 261-267.
