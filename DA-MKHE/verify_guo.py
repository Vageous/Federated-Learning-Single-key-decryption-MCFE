from itertools import count
from pypbc import *
import hashlib

params=Parameters(qbits=20,rbits=10)
pairing=Pairing(params)

class verify_guo(object):

    def __init__(self,q) -> None:
        self.q=q
        self.g=[]
        self.g1=Element(pairing,G1)
        for i in range(self.q):
            self.g.append(Element.random(pairing,G1))
    
    def padding(self,plaintext):
        length=len(plaintext)
        if (length % self.q !=0 ):
            for i in range(self.q-length % self.q):
                plaintext.append(0)
        return plaintext

    def linear_hash(self,plaintext):
        h=Element.one(pairing,G1)
        for i in range(self.q):
            h *= self.g[i]**Element(pairing,Zr,plaintext[i])
        h=Element.from_hash(pairing,Zr,str(h))
        return h
    
    def hash(self,plaintext):
        H=[]
        h=Element.one(pairing,G1)
        for i in range(len(plaintext) // self.q):
            for i in range(self.q):
                h *= self.g[i]**Element(pairing,Zr,plaintext[i])
                h = Element(pairing,G1,h)
            H.append(h)
        return H

    def linear(self,h,coeff):
        h1=Element.one(pairing,G1)
        for i in range(len(coeff)):
            h1 *= h[i]**Element(pairing,Zr,coeff[i])
        h1=Element(pairing,G1,h1)
        return h1
    
    def verify(self,result,hash):
        count1=0
        count2=0
        for i in range(len(hash)):
            if result[i]==hash[i]:
                count1 +=1
            else:
                count2 +=1
        count1 = count1 // len(hash)
        count2 = count2 // len(hash)
        return count1,count2


# test
# 用户端执行一次上述哈希函数，接收到聚合结果后，实行一次hash函数，对比两次的hash结果
verify=verify_guo(4)
print(verify.g)
m=[1,2,3,4,5,6]
h=verify.hash(verify.padding(m))
coun=verify.verify(verify.hash(verify.padding(m)),h)
print(h)
print(coun)
