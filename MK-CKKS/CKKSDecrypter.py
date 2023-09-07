import numpy as np
from numpy.polynomial import Polynomial
from Ciphertext import Ciphertext
from CKKSKeyGenerator import CKKSKeyGenerator
from CKKSEncrypter import CKKSEncrypter
import utils

class CKKSDecrypter:
  def __init__(self, poly_degree, secret_key, big_modulus, cipher_modulus):
    self.poly_degree = poly_degree
    self.big_modulus = big_modulus
    self.cipher_modulus = cipher_modulus
    self.secret_key = secret_key

  def decrypt(self, ciphertext):
    c0 = ciphertext.c0
    c1 = ciphertext.c1

    poly_modulus = Polynomial([1] + [0] * (self.poly_degree-2) + [1])

    plain=c0 + (c1 * self.secret_key % poly_modulus)
    
    plain = Polynomial(utils.crange((plain.coef % self.cipher_modulus), self.cipher_modulus))

    return plain

# 多密钥解密过程
  def PartDec(self, ciphertext):
    c0 = ciphertext.c0
    c1 = ciphertext.c1

    err = Polynomial(utils.discrete_gaussian1(self.poly_degree)) #噪声
    poly_modulus = Polynomial([1] + [0] * (self.poly_degree-2) + [1])
    u = c1 * self.secret_key % poly_modulus
    u += err
    u = Polynomial(utils.crange((u.coef % self.cipher_modulus), self.cipher_modulus))
    return u

  def PartDec1(self,ciphertext,u):
    for i in range(len(ciphertext)):
      c0=ciphertext[i].c0
      c1=ciphertext[i].c1
      err = Polynomial(utils.discrete_gaussian1(self.poly_degree))
      poly_modulus = Polynomial([1] + [0] * (self.poly_degree-2) + [1])
      u[i] = c1 * self.secret_key % poly_modulus
      u[i] += err
      u[i] = Polynomial(utils.crange((u[i].coef % self.cipher_modulus), self.cipher_modulus))
    return u

  def Merge(self, C_sum, u):
    plain = C_sum[0] + u
    plain = Polynomial(utils.crange((plain.coef % self.cipher_modulus), self.cipher_modulus))
    return plain

  def Merge1(self, C_sum, u):
    plain1=[]
    for i in range(len(u)):
      # plain[i].append(C_sum[i][0] + u[i])
      plain1.append(Polynomial(utils.crange(((C_sum[i][0] + u[i]).coef % self.cipher_modulus), self.cipher_modulus)))
    return plain1

