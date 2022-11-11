import torch
import torch.nn as nn
from model import Net,Net_S
import torch 
from torch import optim
from nni.algorithms.compression.pytorch.quantization import LsqQuantizer, NaiveQuantizer, QAT_Quantizer
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import FPGMPruner
from test import test
from train import train
import numpy as np
from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
from pytorch_model_summary import summary
from test import test,test_model_file

def QAT():
    model = Net_S() 
    model.load_state_dict(torch.load('./checkpoints/model.pth'))
    model.train() 
    model.qconfig = torch.quantization.get_default_qconfig("'qnnpack'")
    model_prepared =torch.quantization.prepare_qat(model, inplace=True)
    train(model_prepared)
    model_prepared_int8 = torch.quantization.convert(model_prepared)
    print(summary(model_prepared_int8 , torch.zeros((1, 1, 28, 28)), show_input=False, show_hierarchical=True))
    test_model_file(model_prepared_int8, resource=True ,log_file= "./logFiles/PY_QAT.pkl")
    torch.save(model_prepared_int8, './model/PY_QAT.pth')
    


def PTSQ():
    model = Net_S() 
    model.load_state_dict(torch.load('./checkpoints/model.pth')) # 
    model.eval()
    
    model.qconfig = torch.quantization.get_default_qconfig("'qnnpack'")
    
    model_prepared = torch.quantization.prepare(model)
    for i in range(100):
        input_data = torch.randn(1,1,28,28)
        model_prepared(input_data)
    model_prepared_int8 = torch.quantization.convert(model_prepared)
    
    print("OK")
    print(summary(model_prepared_int8 , torch.zeros((1, 1, 28, 28)), show_input=False, show_hierarchical=True))
    test_model_file(model_prepared_int8, resource=True ,log_file= "./logFiles/PY_Q_D.pkl")
    torch.save(model_prepared_int8, './model/PY_Q_S.pth')
    




def PTDQ():
    model = Net_S() 
    model = torch.load('./model/model.pth')
    quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8  )
    print(summary(quantized_model, torch.zeros((1, 1, 28, 28)), show_input=False, show_hierarchical=True))
    test_model_file(quantized_model, resource=True ,log_file= "./logFiles/PY_Q_D.pkl")
    torch.save(quantized_model, './model/PY_Q_D_model.pth')
 
 
  

  
def NNI():

    model = Net() 

    model.load_state_dict(torch.load('./checkpoints/model.pth'))
    config_list2 = [{
    'op_types': ['Conv2d'],
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'quant_dtype': 'int',
    'quant_scheme': 'per_channel_symmetric'
}]
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # quantizer = QAT_Quantizer(model, config_list2, optimizer, torch.rand(6,1,28,28))
    quantizer = LsqQuantizer(model, config_list2, optimizer)
    quantizer.compress()

    model_path = "mnist_model.pth"
    calibration_path = "mnist_calibration.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)
    # print("calibration_config: ", calibration_config)

    batch_size = 1
    input_shape = (batch_size, 1, 28, 28)
    engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=batch_size)
    engine.compress()
    
    test(model)



if __name__ == "__main__":
    PTDQ() 
    
