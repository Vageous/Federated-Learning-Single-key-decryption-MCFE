import model
import torch
from numpy.polynomial import Polynomial
import numpy as np

class server(object):
    def __init__(self,test_set,num):

        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model=model.cnn().to(device)
        self.test_loader=torch.utils.data.DataLoader(test_set,batch_size=16,shuffle=True)
        self.num=num

    def model_aggragate(self,weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            
            update_per_layer = weight_accumulator[name] / self.num

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.float64))
            else:
                data.add_(update_per_layer)

    def setup(self):

        weight_accumulator={}
        weight_accumulator["conv1.weight"]=[]
        weight_accumulator["conv1.bias"]=[]
        weight_accumulator["conv2.weight"]=[]
        weight_accumulator["conv2.bias"]=[]
        
        weight_accumulator["fc1.bias"]=[]
        
        weight_accumulator["fc2.bias"]=[]
        weight_accumulator["fc3.weight"]=[]
        weight_accumulator["fc3.bias"]=[]
        
        return weight_accumulator

    def init(self,num):

        u={}
        u["conv1.weight"]=Polynomial(torch.zeros(16))
        u["conv1.bias"]=Polynomial(torch.zeros(16))
        u["conv2.weight"]=Polynomial(torch.zeros(16))
        u["conv2.bias"]=Polynomial(torch.zeros(16))
        u["fc1.bias"]=Polynomial(torch.zeros(16))
        u["fc2.bias"]=Polynomial(torch.zeros(16))
        u["fc3.weight"]=Polynomial(torch.zeros(16))
        u["fc3.bias"]=Polynomial(torch.zeros(16))
        return u

    def true(self,name):
        if name=="conv1.weight":
            return 150
        elif name=="conv1.bias":
            return 6
        elif name=="conv2.weight":
            return 2400
        elif name=="conv2.bias":
            return 16
        elif name=="fc1.bias":
            return 120
        elif name=="fc2.bias":
            return 84
        elif name=="fc3.weight":
            return 840
        elif name=="fc3.bias":
            return 10

    def model_test(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0

        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            # print(data.shape) 
            dataset_size += data.size()[0]
            # print(dataset_size)
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                
            
            output = self.global_model(data)
            # print(output.shape)
            total_loss += torch.nn.functional.cross_entropy(output, target,reduction='sum').item() # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        
        
        return acc, total_l