import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from model import Net

def train(net):

    N_EPOCH = 1
    loss_list=[]

    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])

    trainset = tv.datasets.MNIST(
        root='data/',
        train=True,
        download=True,
        transform=transform
    )
    
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=6,
        shuffle=True
    )

 
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(N_EPOCH):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
  
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            

 
            optimizer.zero_grad()

    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

 
            optimizer.step()

            running_loss += loss.item()
            
 
            if i % 100 == 99:
                print('[{}/{}][{}/{}] loss: {:.3f}'.format(epoch + 1, N_EPOCH, (i + 1)*6, len(trainset), running_loss / 100))  
                loss_list.append(running_loss/100)
                running_loss = 0.0


    # torch.save(net, './model/model.pth')
        
    # # print('model_org.pth saved')
    # real_time = time.asctime()
    # plt.plot(loss_list,'b',label='loss_org')
    # plt.legend()
    # plt.savefig('./loss/loss__{}.png'.format(real_time))
    # plt.show()

    print('Finished Training')

if __name__ == "__main__":
    model = Net()
    train(model)