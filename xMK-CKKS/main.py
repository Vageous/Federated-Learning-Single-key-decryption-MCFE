
import torch
import random
from client import *
from server import *
import model
import data
from CKKSAggregate import CKKSAggregate
from numpy.polynomial import Polynomial
from CKKSDecrypter import CKKSDecrypter
import numpy as np
from CKKSKeyGenerator import CKKSKeyGenerator
import time
import compute_time
import psutil
import os
import compute_cpu
# 生成密钥
# p=31  q=41    participant number=10    
# p=31
# q=41
# n=p*q
# N=n**2
num=20
num1=1
q=1023928107044561
aggregate=CKKSAggregate(4000,q)
ckkskey=CKKSKeyGenerator(4000,2,num)
decrypt=CKKSDecrypter(4000,2,q)

sk=ckkskey.generate_secret_key()#私钥
pk=ckkskey.generate_public_key(sk)#公钥
aggregate_key=ckkskey.aggregate_key(pk[0])

train_set,test_set=data.get_data()

Server=server(test_set,num1)
Client=[]
for c in range(num):
    Client.append(client(train_set,num,c))



for e in range(20):#全局模型迭代轮数
    t1=time.time()
    candidates=random.sample(Client,num1)#3表示每轮选择的用户数量
    plaintext={}
    weight_accumulator = {}
    # u=dict()
    # for name, params in Server.global_model.state_dict().items():
    #     weight_accumulator[name] = torch.zeros_like(params)
    C_sum=[]
    C_sum1=[[] for _ in range(120)]
    C_sum2=[[] for _ in range(84)]
    # u1=[[Polynomial(torch.zeros(16))] for _ in range(120)]
    # u2=[[Polynomial(torch.zeros(16))] for _ in range(84)]
    # weight_accumulator=Server.setup(num)
    u=Server.init()
    item=np.array([])
    item1=np.array([])
    plain=[]
    sum=0
    sum1=0
    sum2=0
    sum3=0
    # u1=[[Server.init(num)] for _ in range(120)]
    # print(weight_accumulator)
    # weight_accumulator=Server.setup(num)

    for c in candidates:
        diff,a = c.local_train(Server.global_model, aggregate_key, pk[1])#返回每个网络的梯度信息//已加密
        sum +=a

        time1=time.time()
        for name, params in Server.global_model.state_dict().items():
            if name=="fc1.weight":
                weight_accumulator[name]=aggregate.aggrergate1(diff[name],C_sum1)
            elif name=="fc2.weight":
                weight_accumulator[name]=aggregate.aggrergate1(diff[name],C_sum2)
            else:
                weight_accumulator[name]=aggregate.aggregate(diff[name],C_sum)
        time2=time.time()-time1
            
        time3=time.time()
        for name, params in Server.global_model.state_dict().items():
            if name=="fc1.weight":
                # s1=compute_cpu.show_info()
                u[name] = decrypt.PartShare1(weight_accumulator[name],sk[0],u[name])
                # s2=compute_cpu.show_info()-s1
            elif name=="fc2.weight":
                s3=compute_cpu.show_info()
                u[name] = decrypt.PartShare1(weight_accumulator[name],sk[0],u[name])
                s4=compute_cpu.show_info()-s3
            else:
                # print(weight_accumulator[name])
                # s5=compute_cpu.show_info()
                u[name] += decrypt.PartShare(weight_accumulator[name],sk[0])
                # s6=compute_cpu.show_info()-s5
        time4=time.time()-time3

    # compute_time.time("encrypt.txt","{}".format(sum)+",")
    # compute_time.time("aggregate.txt","{}".format(time2)+",")
    compute_time.time("share.txt","{}".format(s4)+",")
    print("所有用户聚合完成!!!")
    
    time5=time.time()
    for name,params in weight_accumulator.items():
        if name=="fc1.weight":
            plaintext[name]=decrypt.Dec1(weight_accumulator[name],u[name])
            # encode=CKKSEncoder(4*576, pow(2,10))
            for i in range(len(plaintext[name])):
                item = np.append(item,plaintext[name][i].coef[0:576] / pow(2,30),axis=0)
            plaintext[name]=torch.tensor(item)
            # print(plaintext[name][0:576])
        elif name=="fc2.weight":
            plaintext[name]=decrypt.Dec1(weight_accumulator[name],u[name])
            # encode=CKKSEncoder(4*120, pow(2,10))
            for i in range(len(plaintext[name])):
                item1=np.append(item1,plaintext[name][i].coef[0:120] / pow(2,30),axis=0)
            plaintext[name]=torch.tensor(item1)
        else:
            plaintext[name]=decrypt.Dec(weight_accumulator[name],u[name])
            # encode=CKKSEncoder(4 * Server.true(name)), pow(2,10)
            plaintext[name]=torch.tensor(plaintext[name].coef[0:Server.true(name)] / pow(2,30))
            # print(plaintext[name])
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        plaintext[name]=plaintext[name].to(device)
    time6=time.time()-time5
    compute_time.time("decrypt.txt","{}".format(time4+time6)+",")

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
    # compute_time.time("total.txt","{}".format(time.time()-t1)+",")
    print(time.time()-t1)