import h5py as hp
import numpy as np

def import_dataset(path, res=(28, 28)):
  data = np.loadtxt(path, delimiter=',', skiprows=1, converters=np.int8)
  labels = data[:, 0]
  images = data[:, 1:].reshape(-1, *res)
  return labels, images

def store_dataset(file, labels, images, name):
  file.create_dataset(f'{name} labels', data=labels)
  file.create_dataset(f'{name} images', data=images)

if __name__== '__main__':
  print('Generating HDF5 dataset...')
  h5_path = '../dataset/dataset.h5'
  paths = ['../dataset/mnist_test.csv', '../dataset/mnist_train.csv']
  names = ['test', 'train']
  with hp.File(h5_path, 'w') as file:
    for path, name in zip(paths, names):
      print(f'Adding dataset [{name}] from [{path}]...')
      labels, images = import_dataset(path)
      store_dataset(file, labels, images, name)
  print('Done.')
