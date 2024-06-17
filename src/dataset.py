import h5py as hp
import torch as pt
from torch.utils.data import Dataset

class DigitDataset(Dataset):

  def __init__(self, path, name):
    self.path = path
    self.file = hp.File(path, 'r')
    self.images = self.file[f'{name} images']
    self.labels = self.file[f'{name} labels']

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, ind):
    images = self.images[ind]
    label = self.labels[ind]
    return images, label

  def __del__(self):
    self.file.close()
