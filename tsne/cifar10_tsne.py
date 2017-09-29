from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
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

#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

batch_size = 128
num_classes = 10
epochs = 1000

# input image dimensions
img_rows, img_cols = 32,32

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_labels, train_data = [], []

for batch_num in range(1, 6):
	batch_dict = unpickle('/Users/fangdali/Downloads/cifar-10-batches-py/data_batch_'+str(batch_num))

	batch_labels = np.array(batch_dict[b'labels'])
	train_labels.append(batch_labels)

	batch_data = np.array(batch_dict[b'data'])
	batch_data = batch_data.reshape(-1, img_rows, img_cols, 3, order='F')
	batch_data = np.swapaxes(batch_data, 1, 2)
	train_data.append(batch_data)

x_train, y_train = np.array(train_data).reshape(50000, 32, 32, 3), np.array(train_labels).reshape(50000)

batch_dict = unpickle('/Users/fangdali/Downloads/cifar-10-batches-py/test_batch')

y_test_original = np.array(batch_dict[b'labels'])

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
y_test = keras.utils.to_categorical(y_test_original, num_classes)

input_shape = (img_rows, img_cols, 3)

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
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
  print('Test accuracy:', score[1])

'''model.load_weights('cifar10_model.hdf5')
print(model.summary())
N_samples = 10000

per_sample_logloss_list = []
encoded_input_list = []

intermediate_layer_model = keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(index=15).output)

for i in range(N_samples):
  per_sample_logloss_list.append(log_loss(y_test[i:i+1], model.predict(x_test[i:i+1]), normalize=False))
  encoded_input_list.append(intermediate_layer_model.predict(x_test[i:i+1])[0])

encoded_input_list = np.array(encoded_input_list)

per_sample_logloss_list /= np.max(per_sample_logloss_list)
per_sample_logloss_list = per_sample_logloss_list**0.5
print(np.mean(per_sample_logloss_list), np.std(per_sample_logloss_list))
print(np.min(per_sample_logloss_list), np.max(per_sample_logloss_list))

def plot_embedding(X, Y, color_dict):
  x_min, x_max = np.min(X, 0), np.max(X, 0)
  X = (X - x_min) / (x_max - x_min)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  for i in range(X.shape[0]):
    ax.scatter(X[i, 0], X[i, 1], c=color_dict[Y[i]], s=max(10, 100*per_sample_logloss_list[i]))

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(encoded_input_list)

color_dict = {0:'yellow', 1:'blue', 2:'red', 3:'green', 4:'brown',
        5:'tan', 6:'black', 7:'orange', 8:'orchid', 9:'darkturquoise'}

plot_embedding(X_tsne, y_test_original[:N_samples], color_dict)

plt.show()
plt.close()'''


