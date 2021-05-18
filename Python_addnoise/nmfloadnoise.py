import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF
''''''
#load data
test_samples=10000

images_noise=[]
def Loader(PatternID, Noise_Amplitude, Bit_Compression, working_directory):
    if Bit_Compression > 0:
        Bit_Compression = np.power(2.0, Bit_Compression) - 1.0
    Temp = np.load(working_directory + "/Test/Data" + str(PatternID) + ".npz")
    Label = Temp['Label']
    A = Temp['Data'] / 255.0
    Temp = np.load(working_directory + "/Test_Noise/Data" + str(PatternID) + ".npz")
    B = Temp['Data']
    C = A * (1.0 - Noise_Amplitude) + B * Noise_Amplitude
    if Bit_Compression > 0:
        C = np.round(C.astype(np.float64) * Bit_Compression) / Bit_Compression
    C = np.where(C > 1, 1, C)
    Data = C.astype(np.float32)
    return Data, Label


Bit_Compression = 0  # 0 (don't compress), 1, ..., N bits
Number_samples = 10000  # From the 10K test set, 1, ..., 10000
Noise_percentage_start = 0
Noise_percentage_step = 5 # 5% to 5% increase
Noise_percentage_stop = 100
Show_image = False  # True or False
# detect the current working directory and print it
working_directory = os.getcwd()
print("The current working directory is %s" % working_directory)
for Noise_Amplitude in np.arange(start=Noise_percentage_start, stop=Noise_percentage_stop + 1,
                                 step=Noise_percentage_step):  # Noise_Amplitude = 0.0 .... 1.0
    path = working_directory + "/NOISY_MNIST/N_{}".format(Noise_Amplitude)
    if os.path.exists(path) == False:
        os.makedirs(path)
    images_noise = []
    labels10000=[]
    for PatternID in np.arange(0, Number_samples):  # for test set 0...9999
        Data_Raw, Label = Loader(PatternID, Noise_Amplitude / 100, Bit_Compression, working_directory)
        images_noise.append(Data_Raw)
        labels10000.append(Label)
        '''
        if Show_image:
            plt.imshow(Data_Raw)
            plt.show()
        file_name = path + "/in" + str(PatternID) + ".bin"
        '''
        '''
        with open(file_name, mode='wb') as f:
            Data_Raw.tofile(f)
            f.write(bytes([Label]))
            print(Label,'f.write')
        '''
    images_noise=np.resize(images_noise,(test_samples,784))
        #### do your export thing...
    images_noise = images_noise.astype(np.float32)
    starttime = datetime.datetime.now()
#output
    images_noise.astype('float32').tofile("NOISE_MNIST_DATA/X_" + str(Noise_Amplitude) + ".bin")

    oneruntime = datetime.datetime.now()
    print('Noise_Amplitude', Noise_Amplitude, )
    print((oneruntime - starttime).seconds, 'time')

