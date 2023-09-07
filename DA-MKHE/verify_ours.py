
from pypbc import *
import torch
import math
import test

params=Parameters(qbits=20, rbits=10)
pairing=Pairing(params)
# print(params)

class verify(object):

    def __init__(self,q):
        self.q=q
        #padding
        # self.plaintext.reshape(-1)
        # self.result.reshape(-1)

    def padding(self,plaintext):
        length=len(plaintext)
        if (length % self.q !=0 ):
            for i in range(self.q-length % self.q):
                plaintext.append(0)
        return plaintext

    def setup(self):
        g=Element.random(pairing,G1)#生成元
        h=[]
        H=[]
        for i in range(self.q):
            z_i=Element.random(pairing,Zr)
            # z_j=Element.random(pairing,Zr)
            h.append(Element(pairing,G1,g**z_i))#公开参数h_i for i in [1,...,q]
            H.append(Element(pairing,G1,h[0]**z_i))#公开参数h_(1,j) for i in [1,...,q]
        return h,H,g

    def commit(self,h,plaintext):#单个用户对于q长消息向量m生成的承诺
        # length=len(self.plaintext)
        Commit=[]
        commitment=Element.one(pairing,G1)
        for i in range(len(plaintext) // self.q):
            for i in range(self.q):
            # commitment = 
                commitment *= h[i]**(Element(pairing,Zr,plaintext[i]))#计算承诺使其位于群G1中使双线性配对正确
            Commit.append(Element(pairing,G1,commitment))
        return Commit

    def init(self,commit):
        weight_accumulator={}
        for name,params in commit.items():
            weight_accumulator[name]=[]
            for i in range(math.ceil(len(commit[name]) / self.q)):
                weight_accumulator[name].append(Element.one(pairing,G1))
        return weight_accumulator

    def aggregate(self,weight,commit):
        for i in range(len(commit)):
            weight[i] *= commit[i]
            Element(pairing,G1,weight[i])
        return weight

    def opencom(self,H,result):#生成打开承诺的证明lambda
        Lambda=Element.one(pairing,G1)
        proof=[]
        for i in range(len(result) // self.q):
            for i in range(self.q):
                Lambda *= H[i]**(Element(pairing,Zr,result[i]))#在FL中要使用从服务器返回的聚合明文
            proof.append(Element(pairing,G1,Lambda))
        return proof
    
    def verify(self,h,g,Commit,lamda):
        count1=0
        count2=0
        for i in range(len(Commit)):
            e1=pairing.apply(Commit[i],h[0])
            e2=pairing.apply(lamda[i],g)
            if e1==e2:
                count1 +=1
            else:
                count2 +=1
        count1 = count1 // len(Commit)
        count2 = count2 // len(Commit)
        return count1,count2

# #test two user:
# plaintext=[10,20,30]
# plain=[20,40,60]
# verify1=verify(3)
# pp=verify1.setup()
# com1=verify1.commit(pp[0],plaintext)
# for i in range(len(com1)):
#     com1[i] *= com1[i]
# # print(com)
# lam=verify1.opencom(pp[1],plain)
# count=verify1.verify(pp[0],pp[2],com1,lam)
# print(count)
