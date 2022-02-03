from torchvision import datasets , transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
testset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 600, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)
class autonet(nn.Module):
    def __init__(self):
        super(autonet,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
           
            
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        
    def forward(self,x):
        output = self.encoder(x)
        output = self.decoder(output)
        
        return output

losslist=[]
result = torch.Tensor([])
ground_truth = torch.Tensor([])

def train_auto_encoder(epoch):
    auto_encoder.train()
   
    for i in range(epoch):
        running_loss = 0
        avg_loss = 0
        for j , batch in enumerate(trainloader_encoder):
            image,_ = batch
            
            image = Variable(image).cuda()
            
            
            output = auto_encoder(image)
            loss = criterion_encoder(output,image)
            
            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()
            running_loss += loss.detach()
            avg_loss = running_loss/100
            if i == epoch-1:
                result = output
                ground_truth = image
                f = plt.figure()
                f.add_subplot(1,2, 1)
                plt.imshow(ground_truth.data[0].cpu().numpy().squeeze())
                plt.xlabel('ground truth')
                f.add_subplot(1,2, 2)
                plt.imshow(result.data[0].cpu().numpy().squeeze())
                plt.xlabel('result')
                plt.show(block=True)
        
        print('average loss '+str(avg_loss.item())+' on '+str(i)+' epoch') 
        losslist.append(avg_loss.item())
    print(result.shape)
    print(ground_truth.shape)

if __name__ == '__main__':
    
    auto_encoder = autonet()
    auto_encoder = auto_encoder.cuda()

    criterion_encoder = nn.MSELoss()
    optimizer_encoder = optim.SGD(auto_encoder.parameters(),lr = 0.9)
    trainloader_encoder = torch.utils.data.DataLoader(trainset, batch_size = 600, shuffle = True)
    testloader_encoder = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)

    train_auto_encoder(200)