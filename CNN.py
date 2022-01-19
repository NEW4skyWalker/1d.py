import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
classes = 2
batch_size = 400
learning_rate = 0.001


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            # TransNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(32),
            # TransNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
net=ConvNet()
print(net)


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])
dataset = ImageFolder("train-images-idx3-ubyte",transform = data_transform)
train_loader = Data.DataLoader(dataset=dataset, batch_size=400, shuffle=True)

model = ConvNet(classes).to(device)

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(Y_train)//batch_size
len_train = len(X_train)
for epoch in range(num_epochs):
  for i in range(len_train//batch_size):
    batch_x, batch_y = next_batch(X_train, Y_train)
    batch_x = torch.tensor(batch_x).to(device).float()
    batch_y = torch.tensor(batch_y,).to(device).float()

    # Forward pass
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y.long())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.12f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    for i in range(len(Y_test)):
      image = torch.tensor([X_test[i]]).to(device).float()
      label = torch.tensor([Y_test[i]]).to(device).float()
      outputs = model(image)
      _, predicted = torch.max(outputs.data,1)
      if predicted == label:
        correct += 1
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / len(Y_test)))
