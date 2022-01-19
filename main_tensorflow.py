# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 18:24:05 2021

@author: czhu
"""

#### requirements ===> kears<=2.4.3, numpy<=1.19.2, gzip



from keras.datasets import mnist
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
 
local_file = "D:/test/DeepTraffic-master/1.malware_traffic_classification/3.PreprocessedResults/10class/Malware/FlowAllLayers/"
 
#(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'  #训练集图像的文件名
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'  #训练集label的文件名
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'    #测试集图像的文件名
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'    #测试集label的文件名

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    """Extract the images into a 4D uint8 np array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream) #读取前四个字节所表示的magic number
        if magic != 2051: #图片文件的magic number = 2051
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream) #读取接下来4个字节作为图片数目
        #每一张图片包含28X28个像素点,即rows=[28],cols=[28]
        rows = _read32(bytestream) 
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)  #将buf转化为1维数组
        data = data.reshape(num_images, rows, cols, 1)  #改变数组形状
        return data

def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 np array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream) #读取前四个字节所表示的magic number
        if magic != 2049: #label文件的magic number = 2049
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)  #读取接下来4个字节作为label数目
        buf = bytestream.read(num_items) 
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels) #转化为one_hot形式,即每个label由一个10维向量表示,label对应的维度为1,其余为0
        return labels

def dense_to_one_hot(labels_dense, num_classes=10):  ###类别
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

X_train = extract_images(os.path.join(local_file,TRAIN_IMAGES))
y_train = extract_labels(os.path.join(local_file,TRAIN_LABELS))
X_test = extract_images(os.path.join(local_file,TEST_IMAGES))
y_test = extract_labels(os.path.join(local_file,TEST_LABELS))

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)



X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

num_classes = 10

from keras import utils as np_utils
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)


# model = Sequential()
# model.add(Dense(512, input_shape=(784,)))
# model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
#                               # of the layer above. Here, with a "rectified linear unit",
#                               # we clamp all values below 0 to 0.
                           
# model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes))
# model.add(Activation('softmax')) # This special "softmax" activation among other things,
#                                  # ensures the output is a valid probaility distribution, that is
#                                  # that its values are all non-negative and sum to 1.




# model.compile(loss='categorical_crossentropy',
#              optimizer='adam',metrics=['accuracy'])

# train_history = model.fit(x=X_train,
#                         y=Y_train,
#                         epochs=100,batch_size=64,verbose=2)


