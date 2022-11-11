import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from model import Net
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf
import torchvision.transforms as transforms
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torch.autograd import Variable

def py2onnx(model,PATH):
    test_arr = np.random.randn(6, 1, 28, 28).astype(np.float32)
    dummy_input = torch.tensor(test_arr)
    torch_output = model(torch.from_numpy(test_arr))
 
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, 
                  dummy_input, 
                  PATH, 
                  verbose=False, 
                  input_names=input_names, 
                  output_names=output_names)

    model = onnx.load(PATH)
    ort_session = ort.InferenceSession(PATH)
    onnx_outputs = ort_session.run(None, {'input': test_arr})
    print('Export ONNX!')
       

def onnx2TF(ONNX_PATH,TF_PATH):
    onnx_model = onnx.load(ONNX_PATH)  # load onnx model
    tf_rep = prepare(onnx_model)  # creating TensorflowRep object
    tf_rep.export_graph(TF_PATH)
    print("Export Tensorflow!")   

def tf2TFLite(TF_PATH, TFLITE_PATH):
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tf_lite_model)
    print("Export TFLite!")
    
def tf2TFLite_Quantization(TF_PATH, TFLITE_PATH):
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.inference_type = tf.int8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0] : (0., 1.)} 
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tf_lite_model)
    print("Export TFLite!")
    
    

def test(PATH):
    interpreter = tf.lite.Interpreter(model_path=PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
        batch_size = 6,
        shuffle = False,
    )
    correct_num = 0 
    for i, data in enumerate(testloadter):
        if i== 1666:
            continue
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.numpy()
        labels = labels.numpy()
        index = input_details[0]['index']
        interpreter.set_tensor(index, inputs)   
        interpreter.invoke()
        outputs= interpreter.get_tensor(output_details[0]['index'])
        outputs = np.asarray(outputs)
        outputs=torch.from_numpy(outputs)    
         
        _, predicted = torch.max(outputs, 1)     
        for j in range(len(predicted)):
            predicted_num = predicted[j].item()
            label_num = labels[j].item()
          
            if predicted_num == label_num:
                correct_num += 1
    correct_rate = correct_num / len(testset)
    print('correct rate is {:.3f}%'.format(correct_rate * 100))     
  
                
    
if __name__ == "__main__":
    # model = Net()
    # model.load_state_dict(torch.load('./checkpoints/model.pth'))
    # py2onnx(model,"lenet.onnx")
    # onnx2TF("lenet.onnx", "tf_model")
    # tf2TFLite("tf_model" , "lenet.tflite")
    tf2TFLite_Quantization("tf_model" , "lenet_quantized.tflite")

    test("lenet.tflite")
    
