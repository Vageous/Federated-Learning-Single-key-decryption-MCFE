from pypbc import *
from verify_guo import *
import math

params=Parameters(qbits=20,rbits=10)
pairing=Pairing(params)

class verify_xu(object):

    def __init__(self,q) -> None:
        self.q=q
        self.g=Element.random(pairing,G1)
        self.h=Element.random(pairing,G1)
        self.d=Element.random(pairing,Zr)
        self.PF_1=[]
        self.PF_2=[]
        for i in range(2):
            self.PF_1.append(Element.random(pairing,Zr))
            self.PF_2.append(Element.random(pairing,Zr))
        self.psi=Element(pairing,Zr,self.PF_1[0]*self.PF_2[0]+self.PF_1[1]*self.PF_2[1])

    def padding(self,plaintext):
        length=len(plaintext)
        if (length % self.q !=0 ):
            for i in range(self.q-length % self.q):
                plaintext.append(0)
        return plaintext

    def param(self,plaintext):
        A,B,L,Q,PF=[],[],[],[],[]
        verify=verify_guo(self.q)
        PF.append(Element(pairing,G1,self.g**(self.psi)))
        PF.append(Element(pairing,G1,self.h**(self.psi)))
        for i in range(0,len(plaintext),self.q):
            A.append(Element(pairing,G1,self.g**(verify.linear_hash(plaintext[i:i+self.q]))))
            B.append(Element(pairing,G1,self.h**(verify.linear_hash(plaintext[i:i+self.q]))))
        for i in range(len(A)):
            L.append(Element(pairing,G1,(PF[0]*Element.__invert__(A[i]))**Element.__invert__(self.d)))
            Q.append(Element(pairing,G1,(PF[1]*Element.__invert__(B[i]))**Element.__invert__(self.d)))
        return A,B,L,Q

    def init(self,commit):
        weight_accumulator={}
        for name,params in commit.items():
            weight_accumulator[name]=[[],[],[],[]]
            for i in range(len(weight_accumulator[name])):
                for j in range(math.ceil(len(commit[name]) / self.q)):
                    weight_accumulator[name][i].append(Element.one(pairing,G1))
        return weight_accumulator

    def aggregate(self,weight,commit):
        for i in range(len(weight)):
            for j in range(len(weight[i])):
                weight[i][j] *= commit[i][j]
                Element(pairing,G1,weight[i][j])
        return weight

    def verify(self,result,A,B,L,Q):
        count1=0
        count2=0
        verify=verify_guo(self.q)
        psi=(pairing.apply(self.g,self.h))**self.psi
        A1,B1=[],[]
        e1,e2,e3,e4,e5,e6,e7=[],[],[],[],[],[],[]
        for i in range(0,len(result),self.q):
            A1.append(Element(pairing,G1,self.g**(verify.linear_hash(result[i:i+self.q]))))
            B1.append(Element(pairing,G1,self.h**(verify.linear_hash(result[i:i+self.q]))))
        for i in range(len(A1)):
            e1.append(pairing.apply(A[i],B[i]))
            e2.append(pairing.apply(A1[i],B1[i]))
            e3.append(pairing.apply(A[i],self.h))
            e4.append(pairing.apply(self.g,B[i]))
            e5.append(pairing.apply(L[i],self.h))
            e6.append(pairing.apply(self.g,Q[i]))
            e7.append(pairing.apply(A[i],self.h)*(pairing.apply(L[i],self.h)**self.d))
        for i in range(len(result) // self.q):
            if e1[i]==e2[i] and e3[i]==e4[i] and e5[i]==e6[i] and psi==e7[i]:
                count1 +=1
            else:
                count2 +=1
        count1 = count1 // (len(result) // self.q)
        count2 = count2 // (len(result) // self.q)
        return count1,count2

# test one user:
# m=[1,2,3,4,5,6,7,8,9,10]
# verify=verify_xu(4)
# pp=verify.param(verify.padding(m))
# # print(pp)
# count=verify.verify(m,pp[0],pp[1],pp[2],pp[3])
# print(count)

