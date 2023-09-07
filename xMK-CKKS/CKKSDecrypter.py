import numpy as np
from numpy.polynomial import Polynomial
from Ciphertext import Ciphertext
from CKKSKeyGenerator import CKKSKeyGenerator
from CKKSEncrypter import CKKSEncrypter
import utils
import torch
from memory_profiler import *

class CKKSDecrypter:
  def __init__(self, poly_degree, big_modulus, cipher_modulus):
    self.poly_degree = poly_degree
    self.big_modulus = big_modulus
    self.cipher_modulus = cipher_modulus
    # self.secret_key = secret_key

  # xMK-CKKS的解密共享生成
  def PartShare(self, ciphertext, s):
    # ciphertext为[c_sum0,c_sum1]中的c_sum1
    err = Polynomial(utils.discrete_gaussian1(self.poly_degree))
    poly_modulus = Polynomial([1] + [0] * (self.poly_degree-2) + [1])
    D = s * ciphertext[1] % poly_modulus
    D += err
    D = Polynomial(utils.crange((D.coef % self.cipher_modulus), self.cipher_modulus))
    return D

  def PartShare1(self, ciphertext,s,d):
    for i in range(len(ciphertext)):
      err = Polynomial(utils.discrete_gaussian1(self.poly_degree))
      poly_modulus = Polynomial([1] + [0] * (self.poly_degree-2) + [1])
      D = s * ciphertext[i][1] % poly_modulus
      D += err
      D = Polynomial(utils.crange((D.coef % self.cipher_modulus), self.cipher_modulus))
      d[i][0] += D
    return d
  
  # xMK-CKKS的解密共享生成
  def Dec(self, ciphertext, D):
    # ciphertext为c_sum0
    plain = Polynomial(utils.crange(((ciphertext[0] + D).coef % self.cipher_modulus), self.cipher_modulus))
    return plain

  def Dec1(self,ciphertext,D):
    plain1=[]
    for i in range(len(ciphertext)):
      plain1.append(Polynomial(utils.crange(((ciphertext[i][0] + D[i][0]).coef % self.cipher_modulus), self.cipher_modulus)))
    return plain1
    



