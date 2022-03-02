import gzip
import matplotlib.pyplot as plt 
import torch
import numpy as np

def load_dataset(images_path, labels_path, n):
  image_size = 28
  
  f = gzip.open(labels_path,'r')
  f.read(16)
  buf = f.read(n)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  labels = labels.reshape(n, 1)

  f = gzip.open(images_path,'r')
  f.read(16)
  buf = f.read(image_size * image_size * n)
  images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  images = images.reshape(n, image_size, image_size, 1)
    
  return torch.tensor(images), torch.tensor(labels)


def show_sample(loader):
  batch_idx, (example_data, example_targets) = next(loader)
  fig = plt.figure()

  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
  
  return fig
  
def get_size(loader):
  batch_idx, (example_data, example_targets) = next(loader)
  n, h, w, c = example_data.size()
  return n, h, w, c