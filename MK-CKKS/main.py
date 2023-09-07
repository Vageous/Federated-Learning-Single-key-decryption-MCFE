import torch
import random
from client import *
from server import *
import data
from CKKSAggregate import CKKSAggregate
from numpy.polynomial import Polynomial
from CKKSDecrypter import CKKSDecrypter
import numpy as np
# 生成密钥

num=10
num1=3
q=1023928107044561 #50bits
# q=894770345362127507956660870327 #100bits
# q=819201471576550858718372633314846011702797860254006055340363 #200bits
aggregate=CKKSAggregate(2048,q)

ckkskey=CKKSKeyGenerator(2048,2)
decrypt=CKKSDecrypter(2048,ckkskey.secret_key,2,q)

train_set,test_set=data.get_data()

Server=server(test_set,num1)
Client=[]
for c in range(num):
    Client.append(client(train_set,num,c))

for e in range(50):#全局模型迭代轮数
    candidates=random.sample(Client,num1)#3表示每轮选择的用户数量
    plaintext={}
    weight_accumulator = {}
    u=dict()
    # for name, params in Server.global_model.state_dict().items():
    #     weight_accumulator[name] = torch.zeros_like(params)
    C_sum=Server.setup()
    C_sum1=[[] for _ in range(120)]
    C_sum2=[[] for _ in range(84)]
    u1=[[Polynomial(torch.zeros(2048))] for _ in range(120)]
    u2=[[Polynomial(torch.zeros(2048))] for _ in range(84)]
    # weight_accumulator=Server.setup(num)
    u=Server.init(num)
    item= np.array([])
    item1= np.array([])
    plain={}


    for c in candidates:
        diff = c.local_train(Server.global_model, ckkskey.public_key)#返回每个网络的梯度信息//已加密
        for name, params in Server.global_model.state_dict().items():
            if name=="fc1.weight":
                # print(len(diff[name]))
                weight_accumulator[name]=aggregate.aggrergate1(diff[name],C_sum1)
                u[name] = decrypt.PartDec1(diff[name],u1)
            elif name=="fc2.weight":
                weight_accumulator[name]=aggregate.aggrergate1(diff[name],C_sum2)
                u[name] = decrypt.PartDec1(diff[name],u2)
            else:
                weight_accumulator[name]=aggregate.aggregate(diff[name],C_sum[name])
                u[name] += decrypt.PartDec(diff[name])

    print("所有用户聚合完成!!!")
    
    for name,params in weight_accumulator.items():
        if name=="fc1.weight":
            plaintext[name]=decrypt.Merge1(weight_accumulator[name],u[name])
            # print(plaintext[name])
            encode=CKKSEncoder(4*576, pow(2,20))
            for i in range(len(plaintext[name])):
                item = np.append(item,encode.decode(plaintext[name][i]).real,axis=0)
            plaintext[name]=torch.tensor(item)
            # print(plaintext[name])
        elif name=="fc2.weight":
            plaintext[name]=decrypt.Merge1(weight_accumulator[name],u[name])
            encode=CKKSEncoder(4*120, pow(2,20))
            for i in range(len(plaintext[name])):
                item1 = np.append(item1,encode.decode(plaintext[name][i]).real,axis=0)
            plaintext[name]=torch.tensor(item1)
        else:
            plain[name]=decrypt.Merge(weight_accumulator[name],u[name])
            encode=CKKSEncoder(4 * Server.true(name), pow(2,20))
            plaintext[name]=torch.tensor(encode.decode(plain[name]).real)
            # print(plaintext[name])
            # while(Server.true(plaintext[name],name)):
            #     plaintext[name]=torch.tensor(encode.decode(plain[name]).real)

            # print(plaintext[name].shape)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        plaintext[name]=plaintext[name].to(device)


    plaintext["conv1.weight"]=plaintext["conv1.weight"].data.reshape(6,1,5,5)
    plaintext["conv2.weight"]=plaintext["conv2.weight"].data.reshape(16,6,5,5)
    plaintext["fc1.weight"]=plaintext["fc1.weight"].data.reshape(120,576)
    plaintext["fc2.weight"]=plaintext["fc2.weight"].data.reshape(84,120)
    plaintext["fc3.weight"]=plaintext["fc3.weight"].data.reshape(10,84)
    # print(plaintext["conv1.weight"].shape)
    print("解密完成!!!!")
    
    Server.model_aggragate(plaintext)# 1/num

    acc, loss = Server.model_test()

    print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))