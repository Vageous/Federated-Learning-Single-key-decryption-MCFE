import numpy as np
from numpy.polynomial import Polynomial
from Ciphertext import Ciphertext
import utils



class CKKSAggregate:
    def __init__(self, poly_degree, cipher_modulus) -> None:
        self.poly_degree = poly_degree
        self.cipher_modulus = cipher_modulus

    def aggregate(self,ciphertext,C_sum):
        c0=ciphertext.c0
        c1=ciphertext.c1
        if len(C_sum) == 0:
            C_sum.append(c0)
            C_sum.append(c1)
        else:
            C_sum[0] = Polynomial(utils.crange(((C_sum[0] + c0).coef % self.cipher_modulus), self.cipher_modulus))
            C_sum.append(c1)
        return C_sum

    def aggrergate1(self,ciphertext,C_sum):
        for i in range(len(ciphertext)):
            c0=ciphertext[i].c0
            c1=ciphertext[i].c1
            if len(C_sum[i]) ==0:
                C_sum[i].append(c0)
                C_sum[i].append(c1)
            else:
                C_sum[i][0]=Polynomial(utils.crange(((C_sum[i][0] + c0).coef % self.cipher_modulus), self.cipher_modulus))
                C_sum[i].append(c1)

        return C_sum
