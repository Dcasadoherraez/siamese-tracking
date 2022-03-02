import torch.nn as nn 
import torch

# https://github.com/AceEviliano/Siamese-network-on-MNIST-PyTorch/blob/master/Siamese/Siamese%20Net.ipynb

class Siamese(nn.Module):
    
    def __init__(self):
        super(Siamese,self).__init__()
        # A simple two layer convolution followed by three fully connected layers should do
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        
        self.pool1 = nn.MaxPool2d( kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d( kernel_size=3)
        
        self.lin1 = nn.Linear(144, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 10)
        
    def forward(self,x):
        # forwarding the input through the layers
          
        out = self.pool1(nn.functional.relu(self.conv1(x)))
        out = self.pool2(nn.functional.relu(self.conv2(out)))
        
        out = out.view(-1,144)
        
        out = nn.functional.relu(self.lin1(out))
        out = nn.functional.relu(self.lin2(out))
        out = self.lin3(out)
        
        return out

    def evaluate(self, x, y):
        # this can be used later for evalutation
        
        m = torch.tensor(1.0, dtype=torch.float32)
        
        if type(m) != type(x):
            x = torch.tensor(x, dtype = torch.float32, requires_grad = False)
            
        if type(m) != type(y):
            y = torch.tensor(y, dtype = torch.float32, requires_grad = False)
        
        x = x.view(-1,1,28,28)
        y = y.view(-1,1,28,28)
        
        with torch.no_grad():
            out1, out2 = self.forward(x, y)
            return nn.functional.pairwise_distance(out1, out2)
        
class ContrastiveLoss(nn.Module):
 
  def __init__(self, margin=2.0):
      super(ContrastiveLoss, self).__init__()
      self.margin = margin

  def forward(self, anchor, positive, negative):
      return self.forward_once(anchor, positive, 1) + self.forward_once(anchor, negative, 0)

  def forward_once(self, anchor, sample, label):
        dist = nn.functional.pairwise_distance(sample, anchor)
        
        loss_contrastive = torch.mean((1-label) * torch.pow(dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

        return loss_contrastive