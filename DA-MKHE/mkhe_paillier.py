from gmpy2 import mpz,invert,t_mod,gcd,powmod,is_prime,mpz_random,div,random_state
from Crypto.Util.number import getPrime
import random,sys
import numpy as np

rand=random_state(random.randrange(sys.maxsize))
class KGC(object):
    def __init__(self,bits,num) -> None:
        self.bits=bits
        self.num=num
        self.p_q_gen()
        self.n_gen()
        self.psi_gen()
        self.g_gen()
        self.sk_u()
        self.polynomial()
        self.pk_u()
        self.sk_s()
        self.pk_s()

    def p_q_gen(self):
        self.p=getPrime(self.bits)
        self.q=getPrime(self.bits)

    def n_gen(self):
        self.n=self.p*self.q
        self.N=self.n*self.n
    
    def psi_gen(self):
        self.psi=self.n*(self.p-mpz(1))*(self.q-mpz(1))

    def g_gen(self):
        self.g=2
        while(powmod(self.g,self.psi,self.N)!=1):
            self.g=self.g+1

    def sk_u(self):
        self.sk=[]
        for i in range(self.num):
            self.sk.append(mpz_random(rand,1000))
        
    def polynomial(self):
        self.poly=np.poly(self.sk)
        for i in range(len(self.poly)):
            self.poly[i]= self.poly[i] % self.psi
        

    def pk_u(self):
        self.pk=[]
        for i in range(len(self.poly)):
            self.pk.append(powmod(self.g,self.poly[len(self.poly)-i-1],self.N))
    
    def sk_s(self):
        self.sk1=mpz_random(rand,1000)

    def pk_s(self):
        self.pk1=powmod(self.g,self.sk1,self.N)

class params(object):
    def __init__(self,g,n,N,psi) -> None:
        self.g=g
        self.n=n
        self.N=N
        self.psi=psi


class participant(object):
    def __init__(self,params) -> None:
        self.n=params.n
        self.N=params.N
        self.g=params.g
        self.psi=params.psi

    def ran_pick(self,num):
        r=[]
        j=0
        t=[mpz_random(rand,1000)for i in range(num)]
        while j<num:
            r1=mpz_random(rand,1000)
            temp=powmod(self.g,r1,self.N)
            if(gcd(temp,self.N)==1):
                r.append(r1)
                j=j+1
        return t,r
        

    def encrypt(self,m,pk_u,pk_s):
        cipher,pre_compute=[],[]
        ran_num=self.ran_pick(len(m))
        
        for i in range(len(ran_num[0])):
            temp1=[]
            for j in range(2,len(pk_u)):
                temp1.append(powmod(pk_u[j],ran_num[0][i],self.N))
            pre_compute.append(temp1)

        for i in range(len(m)):
            temp=[]
            temp.append(t_mod(powmod((mpz(1)+self.n),t_mod(m[i],self.N),self.N)*powmod(pk_u[0],ran_num[0][i],self.N),self.N))
            temp.append(t_mod(powmod(pk_u[1],ran_num[0][i],self.N)*powmod(pk_s,ran_num[1][i],self.N),self.N))
            for j in range(len(pre_compute[i])):
                temp.append(pre_compute[i][j])
            temp.append(powmod(self.g,ran_num[1][i],self.N))
            cipher.append(temp)
        return cipher

    def decrypt(self,agg_cipher,sk_u,sk_s):
        plain=[]
        for i in range(len(agg_cipher)):
            agg_cipher[i][1]=t_mod(agg_cipher[i][1]*invert(powmod(agg_cipher[i][len(agg_cipher[i])-1],sk_s,self.N),self.N),self.N)

        for i in range(len(agg_cipher)):
            temp=mpz(1)
            for j in range(len(agg_cipher[i])-1):
                temp=t_mod(temp*powmod(agg_cipher[i][j],powmod(sk_u,j,self.psi),self.N),self.N)
            plain.append(int(temp-mpz(1))/self.n)
        return plain

class server(object):
    def __init__(self,params) -> None:
        self.N=params.N

    def aggregate(self,agg_cipher,cipher):
        for i in range(len(cipher)):
            for j in range(len(cipher[i])):
                agg_cipher[i][j]=t_mod(agg_cipher[i][j]*cipher[i][j],self.N)
        return agg_cipher


def main(bits,num):
    kgc=KGC(bits,num)
    param=params(kgc.g,kgc.n,kgc.N,kgc.psi)
    temp=mpz(1)
    client=participant(param)
    m=[1000 for i in range(num)]
    for i in range(5):
        agg_cipher=[[mpz(1)for i in range(num+2)]for i in range(len(m))]
        for i in range(num):
            cipher=client.encrypt(m,kgc.pk,kgc.pk1)
            agg_cipher=client.aggregate(agg_cipher,cipher)
        plain=client.decrypt(agg_cipher,kgc.sk[0],kgc.sk1)
        print(plain)

# if __name__=='__main__':
    # main(20,10)
