from config import *

# All the imports
import warnings 
warnings.filterwarnings('ignore')

# deep learning libraries
import torch
import torch.nn as nn 

# utility libraries

# system libraries
import os

# project libraries
from mnist import *
from model import *


train_images_path = os.path.join(mnist_path, 'raw/train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(mnist_path, 'raw/train-labels-idx1-ubyte.gz')

test_images_path = os.path.join(mnist_path, 'raw/t10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(mnist_path, 'raw/t10k-labels-idx1-ubyte.gz')

x_train, y_train = load_dataset(train_images_path, train_labels_path, 5000)
x_test, y_test = load_dataset(test_images_path, test_labels_path, 5000)

# Sizes
_, h, w, c = x_test.shape

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters  
batch_size_train = 256
batch_size_test = 256
model = Siamese().to(device)
learning_rate = 1e-3
num_epochs = 1
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scaler = torch.cuda.amp.GradScaler()
num_hard = int(batch_size_train * 0.5)

# training loop
n_total_steps = len(x_train)