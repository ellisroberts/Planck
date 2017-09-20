from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import gzip
from time import time
from sklearn.metrics import log_loss
import pickle
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

batch_size = 1
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 32,32

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_labels, train_data = [], []

for batch_num in range(1, 6):
	batch_dict = unpickle('/home/balerion/Downloads/cifar-10-batches-py/data_batch_'+str(batch_num))

	batch_labels = np.array(batch_dict[b'labels'])
	train_labels.append(batch_labels)

	batch_data = np.array(batch_dict[b'data'])
	batch_data = batch_data.reshape(-1, img_rows, img_cols, 3, order='F')
	batch_data = np.swapaxes(batch_data, 1, 2)
	train_data.append(batch_data)

x_train, y_train = np.array(train_data).reshape(50000, 32, 32, 3), np.array(train_labels).reshape(50000)

def plot_embedding(X, Y, color_dict):
  x_min, x_max = np.min(X, 0), np.max(X, 0)
  X = (X - x_min) / (x_max - x_min)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for i in range(X.shape[0]):
    ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=color_dict[Y[i]], s=50.)

N_samples = 2000
print(y_train[:N_samples])

tsne = manifold.TSNE(n_components=3, init='pca', random_state=0, learning_rate=10000.0)
X_tsne = tsne.fit_transform(x_train.reshape(x_train.shape[0], img_rows*img_cols*3)[:N_samples])

color_dict = {0:'yellow', 1:'blue', 2:'red', 3:'green', 4:'brown',
        5:'tan', 6:'black', 7:'orange', 8:'orchid', 9:'darkturquoise'}

plot_embedding(X_tsne, y_train[:N_samples], color_dict)
plt.show()
plt.close()


'''batch_dict = unpickle('/home/balerion/Downloads/cifar-10-batches-py/test_batch')

y_test = np.array(batch_dict[b'labels'])

batch_data = np.array(batch_dict[b'data'])
batch_data = batch_data.reshape(-1, img_rows, img_cols, 3, order='F')
x_test = np.swapaxes(batch_data, 1, 2)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

for epoch_num in range(epochs):
  checkpoint = ModelCheckpoint('cifar10_model.hdf5', monitor='val_loss', verbose=0, save_best_only=True)
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=1,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks = [checkpoint])
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])'''