import torch
import torchvision as tv
from model import Net
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.autograd import Variable
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L1NormPruner, L2NormPruner, FPGMPruner, LevelPruner
from nni.algorithms.compression.pytorch.pruning import AGPPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from test import test,test_model_file
from train import train
from pytorch_model_summary import summary
    
    
def NNI():

    model = Net() 
    model = torch.load('./model/model.pth')

    config_list = [{
        'sparsity_per_layer': 0.5,
        'op_types': ['Conv2d']
    }]
    

    # get mask
    pruner = L2NormPruner(model, config_list)
    _, masks = pruner.compress()
    # for name, mask in masks.items():
    #     print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
        
    # speedup
    pruner._unwrap_model()
    ModelSpeedup(model, torch.rand(1, 1, 28, 28), masks).speedup_model()
    # print(model)
    # print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=False, show_hierarchical=True))
    
    test_model_file(model, resource=True ,log_file= "./logFiles/nni_prun.pkl")
    torch.save(model, './model/NNI_Prun_model.pth')
    
    
def PY():
    model = Net() 
    model = torch.load('./model/model.pth')
    parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'))
    

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5,
    )
    print("Sparsity in conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.conv1.weight == 0))/ float(model.conv1.weight.nelement())))
    print("Sparsity in conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.conv2.weight == 0))/ float(model.conv2.weight.nelement())))

    print("Global sparsity: {:.2f}%".format(100. * float(torch.sum(model.conv1.weight == 0)+ torch.sum(model.conv2.weight == 0))
        / float(model.conv1.weight.nelement()+ model.conv2.weight.nelement())))
    print(summary(model, torch.zeros((1, 1, 28, 28)), show_input=False, show_hierarchical=True))
    print(model)
    test_model_file(model, resource=True ,log_file= "./logFiles/PY_prun.pkl")
    torch.save(model, './model/PY_Prun_model.pth')
   
  

if __name__ == "__main__":
   NNI()