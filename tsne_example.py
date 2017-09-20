import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import gzip
from time import time

images_file = open('/Users/fangdali/Downloads/fashion-mnist-master/data/fashion/train-images-idx3-ubyte.gz', "rb")
labels_file = open('/Users/fangdali/Downloads/fashion-mnist-master/data/fashion/train-labels-idx1-ubyte.gz', "rb")

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  """Extract the images into a 4D uint8 np array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 np array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  #print('Extracting', f.name)
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
    data = data.reshape(num_images, rows*cols)
    return data

def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 np array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 np array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  #print('Extracting', f.name)
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

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, Y, color_dict):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=color_dict[Y[i]], s=0.2)


images_array = extract_images(images_file)[:2000,:]
labels_array = extract_labels(labels_file)[:2000]
#print(images_array)
#print(labels_array)

start = time()
tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(images_array)

print(X_tsne)
print('took', time()-start, 's')

color_dict = {0:'yellow', 1:'blue', 2:'red', 3:'green', 4:'brown',
				5:'tan', 6:'black', 7:'orange', 8:'orchid', 9:'darkturquoise'}

plot_embedding(X_tsne, labels_array, color_dict)

plt.show()