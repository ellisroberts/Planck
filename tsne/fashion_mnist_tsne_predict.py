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

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

test_images_file = open('/Users/fangdali/Downloads/fashion-mnist-master/data/fashion/t10k-images-idx3-ubyte.gz', "rb")
test_labels_file = open('/Users/fangdali/Downloads/fashion-mnist-master/data/fashion/t10k-labels-idx1-ubyte.gz', "rb")

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def extract_labels(f, one_hot=False, num_classes=10):
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

test_images_array = extract_images(test_images_file)
test_labels_array = extract_labels(test_labels_file)

(x_test, y_test) = (test_images_array, test_labels_array)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.load_weights('fashion_mnist_model.hdf5')

print(model.summary())

N_samples = 10000
per_sample_logloss_list = []
encoded_input_list = []

intermediate_layer_model = keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer(index=6).output)

for i in range(N_samples):
  per_sample_logloss_list.append(log_loss(y_test[i:i+1], model.predict(x_test[i:i+1]), normalize=False))
  encoded_input_list.append(intermediate_layer_model.predict(x_test[i:i+1])[0])

encoded_input_list = np.array(encoded_input_list)
original_per_sample_logloss_list = np.array(per_sample_logloss_list)
print(encoded_input_list.shape)

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
    ax.scatter(X[i, 0], X[i, 1], c=color_dict[Y[i]], s=max(5, 100*per_sample_logloss_list[i]))

tsne = manifold.TSNE(n_components=2, init='pca', random_state=1337)
#X_tsne = tsne.fit_transform(test_images_array.reshape(test_images_array.shape[0], img_rows*img_cols)[:N_samples])
X_tsne = tsne.fit_transform(encoded_input_list[:N_samples])

filename = 'fashion_mnist_tsne_model.pkl'
pickle.dump(tsne, open(filename, 'wb'))

color_dict = {0:'yellow', 1:'blue', 2:'red', 3:'green', 4:'brown',
        5:'tan', 6:'black', 7:'orange', 8:'orchid', 9:'darkturquoise'}

np.savetxt('tsne_embedding_coords.txt', 
            np.hstack((X_tsne,
                      test_labels_array[:N_samples].reshape(N_samples,1), 
                      original_per_sample_logloss_list[:N_samples].reshape(N_samples,1))),
            fmt='5.5%f')

plot_embedding(X_tsne, test_labels_array[:N_samples], color_dict)

plt.show()
plt.close()
