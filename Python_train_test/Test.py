from sklearn.decomposition import NMF
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from joblib import dump, load
#*******Constant values
MNIST_TRAIN_IMAGENUMBER=60000
MNIST_TEST_IMAGENUMBER=10000
MNIST_TRAIN_IMAGESIZE=784
MNIST_TEST_IMAGESIZE=784
MNIST_NORMALIZING_CONSTANT=255
NMF_RANK=30
#*******load MNIST dataset from tensorflow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / MNIST_NORMALIZING_CONSTANT, x_test / MNIST_NORMALIZING_CONSTANT#normalizing by 255
x_train=np.resize(x_train,(MNIST_TRAIN_IMAGENUMBER,MNIST_TRAIN_IMAGESIZE))
x_test=np.resize(x_test,(MNIST_TEST_IMAGENUMBER,MNIST_TEST_IMAGESIZE))
#**********Load NMF&KNN model after training
nmf = load('MNIST_TrainModel/yz_model/nmf1e-5_1k.joblib')#change to your directory
knn_nmf=load('MNIST_TrainModel/yz_model/knn5nmf1e-5_1k.joblib')#change to your directory
#Read data after NMF
working_directory="stavoidoverwrite.bindata"#change to your directory
readfile = np.fromfile((working_directory),dtype='float32')
readfile_reshape=np.reshape(readfile,(MNIST_TEST_IMAGENUMBER,NMF_RANK))
#Test data after NMF with trained KNN model
print('test score:',knn_nmf.score(readfile_reshape ,y_test))
#0.9523 is expected.

