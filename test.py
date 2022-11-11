import torchvision as tv
import torchvision.transforms as transforms
import time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from thop import profile
from pytorch_model_summary import summary
from ResourceProfile import RP, stop_RP
import numpy as np
from model import Net


def get_dataset():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    testset = tv.datasets.MNIST(
        root='data/',
        train=False,
        download=False,
        transform=transform,
    )
    testloadter = DataLoader(
        dataset = testset,
        batch_size = 1,
        shuffle = False,
    )
    
    return len(testset), testloadter


def test(model):


    # model = Net()
    # model.load_state_dict(torch.load('./checkpoints/model.pth'))
    data_len, dataloader = get_dataset()

    correct_num = 0 
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        print(type(inputs))
        print(inputs)
        break
        outputs = model(inputs)
       
    
        _, predicted = torch.max(outputs, 1)

        for j in range(len(predicted)):
            predicted_num = predicted[j].item()
            label_num = labels[j].item()
          
            if predicted_num == label_num:
                correct_num += 1
                
    exec_time = []
    
	# compute Execution time 
    for i, data in enumerate(dataloader):
        if i == 9:
            break
        else:     
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            print(labels)
            sum = 0
            
            for _ in range(10000):
                start = time.time()
                outputs = model(inputs)
                end = time.time()
                sum = sum + end - start
                
                
        
            Execution_time = sum/10000
            exec_time.append(Execution_time)
        
    
    # compute parameters
    dummy_input = torch.randn(1, 1, 28, 28)
    # print(summary(model, dummy_input, show_input=False, show_hierarchical=False))
    macs, params = profile(model, inputs=(dummy_input,))
    print("macs:" , macs)
    print("params:" , params)
    
    correct_rate = correct_num / data_len
    print('correct rate is {:.3f}%'.format(correct_rate * 100))
    
    for i in range(len(exec_time)):
        print('Execution time of group{:d} is {:f}s'.format(i+1,exec_time[i]))      

 
def test_model_file(model, resource=False ,log_file ="resource.pkl" ):


    # model = Net() 
    # # model.load_state_dict(torch.load(model_path))
    # model = torch.load(model_path)
    
    data_len, dataloader = get_dataset()

    # model.eval()
    
    correct_num = 0        
    exec_time = []
    
	# compute Execution time 
    iter_dataloader = iter(dataloader)
    print("iter" ,len(iter_dataloader))
    num_iterators = 10000

    if resource:
        RP(useGPU=False, filename = log_file )
    for i in range(num_iterators):
        inputs, labels = iter_dataloader.next()
        inputs, labels = Variable(inputs), Variable(labels)
        
        
        start = time.time()
        outputs = model(inputs)
        end = time.time()
        sum =  end - start
        exec_time.append(sum)
        
        _, predicted = torch.max(outputs, 1)
        for j in range(len(predicted)):
            predicted_num = predicted[j].item()
            label_num = labels[j].item()
          
            if predicted_num == label_num:
                correct_num += 1
       

    if resource:
        stop_RP()
    # compute parameters
    dummy_input = torch.randn(1, 1, 28, 28)
    # print(summary(model, dummy_input, show_input=False, show_hierarchical=False))
    macs, params = profile(model, inputs=(dummy_input,))
    print("macs:" , macs)
    print("params:" , params)
    
    correct_rate = correct_num / num_iterators
    print('correct rate is {:.3f}%'.format(correct_rate*100))
    
    print('Execution time of %d pieces of  data is %fs, one piece of data is %fs'%(num_iterators, np.sum(exec_time), np.mean(exec_time)))
       
   
if __name__ == "__main__":
    
    model = Net() 
    model = torch.load('./model/model.pth')
    # model_path2 = './checkpoints/model_pruning.pth'
    test_model_file(model, resource=True ,log_file= "./logFiles/model.pkl")
    # time.sleep(3)
    # test_model_file(model_path2, resource=False , log_file= "logFiles/model_pruning.pkl")