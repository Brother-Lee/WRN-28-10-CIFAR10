import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import wideResnet28_10
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
def main():
    epoch=1
    preprocess=transforms.Compose([transforms.Pad(4, padding_mode='reflect'),transforms.RandomHorizontalFlip(),transforms.RandomCrop(32),transforms.ToTensor()])
    trainset=torchvision.datasets.CIFAR10('./data',train=True,download=False,transform=preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=3)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=1)
    net=wideResnet28_10()
    net.to(0)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4,nesterov=True)

    epoch=restore(net,optimizer,'ckpt/model-192-0.9575.pth')
    max_accuracy=0
    while 1:
        train(net,trainloader,optimizer,criterion)
        accuracy=evaluate(net,testloader)
        
        save(net,optimizer,epoch,accuracy)
        if accuracy>max_accuracy:
            max_accuracy=accuracy
            save(net,optimizer,epoch,max_accuracy,'ckpt_best')
        print(epoch,': ',accuracy,'max: ',max_accuracy)
        epoch+=1

        


def restore(net,optimizer,path):
    if not os.path.exists(path):
        return 1
    else:
        checkpoint=torch.load(path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr']=0.0008
        print('restore successfully')
        return checkpoint['epoch']+1




def save(net,optimizer,epoch,accuracy,path='ckpt'):
    torch.save({
            'state_dict':net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch
    }, path+'/model-%03d-%.4f.pth'%(epoch,accuracy))
def train(net,trainloader,optimizer,criterion):
    net.train()
    for images,labels in tqdm(trainloader,ncols=70):
        images,labels=images.to(0),labels.to(0)
        optimizer.zero_grad()
        outputs=net(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
def evaluate(net,testloader):
    net.eval()
    correct=0
    with torch.no_grad():        
        for images,labels in testloader:
            images,labels=images.to(0),labels.to(0)
            outputs=net(images)
            _,predicted=torch.max(outputs.data,1)
            correct+=(predicted==labels).sum().item()
    return correct/10000
class Preprocess:
    def __init__(self,pad=4,crop=32):
        self.pad=pad
        self.crop=crop
    def __call__(self,image):
        image=F.pad(image,[4,4,4,4])
        x,y=np.random.randint(0,8,[2])
        image=image[:,x:x+32,y:y+32]
        return image     

if __name__=="__main__":
    main()