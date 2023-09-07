import torch
import model
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from CKKSEncoder import CKKSEncoder
from CKKSKeyGenerator import CKKSKeyGenerator
from CKKSEncrypter import CKKSEncrypter
import time

class client(object):

    def __init__(self,train_set,num,id=-1):
        # self.cnn=model.cnn()
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model=model.cnn().to(device)
        # print(self.local_model)
        self.client_id=id
        self.train_set=train_set
        self.clinet_num=num

        # 返回train_set的索引列表
        all_range=list(range(len(self.train_set)))
        data_len=int(len(self.train_set) / self.clinet_num)#划分数据集
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        # 训练迭代器
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=16, 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
        # print(self.train_loader)

    def local_train(self,model,public_key,a):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
                # print(self.local_model.state_dict()[name].copy_(param.clone()))
                
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.1,momentum=0.0001)
        self.local_model.train()
            #本地迭代轮数
        for e in range(3):
            for batch_id, batch in enumerate(self.train_loader):
                data,target=batch
                # print(batch[0].shape)
                # print(batch[1].shape)

                if torch.cuda.is_available():
                    data=data.cuda()
                    target=target.cuda()

                optimizer.zero_grad()
                output=self.local_model(data)
                # print(output.shape)
                # print(output)
                loss=F.cross_entropy(output,target)
                loss.backward()

                optimizer.step()
                
        diff = dict()
        ciphertext= dict()
        # size=dict()
        plaintext= dict()
        plain=dict()
        item=[]
        item1=[]
        sum=0
        # q=1023928107044561
        q=1088733865348074401
        encrypt=CKKSEncrypter(4000,q,2,public_key,a)
        # for name,data in self.local_model.state_dict().items():
        #     self.local_model1.state_dict()[name].copy_(param.clone())
        time1=time.time()
        for name, params in self.local_model.state_dict().items():
            diff[name] = (params - model.state_dict()[name]) * pow(2,30)
            # 编码
            # print(diff[name].size())
            if name=="fc1.weight":
                # print(diff[name][0] / pow(2,30))
                # encode=CKKSEncoder(4 * 576, pow(2,10))
                for i in range(120):
                    item.append(encrypt.encrypt(diff[name].tolist()[i]))
                ciphertext[name]=item 
            elif name=="fc2.weight":
                # encode=CKKSEncoder(4 * 120, pow(2,10))
                for i in range(84):
                    item1.append(encrypt.encrypt(diff[name].tolist()[i]))
                ciphertext[name]=item1
            else:
                # print(diff[name] / pow(2,30))
                plaintext[name]=diff[name].data.reshape(-1)#降维，转为一维
                # encode=CKKSEncoder(4 * len(plaintext[name]), pow(2,10))
                ciphertext[name]=encrypt.encrypt(plaintext[name].tolist())
        time2=time.time()
        sum+=time2-time1
        print("加密结束！！!")
                # print(diff[name])
        # 因为选了三个参与者，所以下面每层都是三个
        # conv1.weight: 6 torch.size: [6,1,5,5]
        # conv1.bias: 6 torch.size: [6]
        # conv2.weight: 16 torch.size: [16,6,5,5]
        # conv2.bias: 16 torch.size: [16]
        # fc1.weight: 120 torch.size: [120,576]
        # fc1.bias:  120 torch.size: [120]
        # fc2.weight: 84 torch.size: [84,120]
        # fc2.bias:  84 torch.size: [84]
        # fc3.weight: 10 torch.size: [10,84]
        # fc3.bias: 10 torch.size: [10]
        return ciphertext,sum


