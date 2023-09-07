import random
import numpy as np
from numpy.polynomial import Polynomial
import utils
from PublicKey import PublicKey

class CKKSKeyGenerator:
  def __init__(self, poly_degree, big_modulus, num):
    self.poly_degree = poly_degree# 多项式阶数，即N, 是2的幂次
    self.big_modulus = big_modulus# 均匀分布的上界
    # self.cipher_modulus = cipher_modulus# 多项式模数，即q
    self.num=num
    # self.generate_secret_key()
    # self.generate_public_key()

  def generate_secret_key(self):
    key=[]
    # key = utils.sample_hamming_weight_vector(self.poly_degree)
    for i in range(self.num):
      key.append(Polynomial(utils.discrete_gaussian(self.poly_degree)))# 从标准正态分布中选取私钥
    # self.secret_key = Polynomial(key)
    # return self.secret_key
    return key

  def generate_public_key(self,secret_key):# big_modulus 是均匀分布的上界
    public_key=[]
    for i in range(self.num):
      coeff = utils.sample_from_uniform_distribution(0, self.big_modulus, self.poly_degree)# 公共参数a
      pk_coeff = Polynomial(coeff)# 公共参数a对应的多项式
      err = utils.discrete_gaussian(self.poly_degree)# 噪声e
      pk_err = Polynomial(err)# 噪声e对应的多项式
      poly_modulus = Polynomial([1] + [0] * (self.poly_degree-2) + [1]) # x^n+1
      p0 = pk_coeff * secret_key[i] % poly_modulus# a*s mod x^n+1
    # p0 %= self.big_modulus# a*s对应的多项式系数 mod big_modulus
      p0 *= -1# -(a*s)
      p0 += pk_err# -(a*s)+e
    # p0 %= self.big_modulus# -(a*s)+e mod big_modulus
      p1 = pk_coeff# 公开参数a的多项式
      public_key.append(p0)
    return public_key, p1
      # self.public_key = PublicKey(p0, p1)
    # return p0,p1# (-a*s+e, a)
  def aggregate_key(self,public_key):
    b=Polynomial(np.zeros(self.poly_degree))
    for i in range(len(public_key)):
      b += public_key[i]
    return b
        


# ckks=CKKSKeyGenerator(16,2,5)
# secret_key=ckks.generate_secret_key()
# print(ckks.generate_secret_key())
# print(ckks.generate_public_key(secret_key)[1])
# print(ckks.aggregate_key(ckks.generate_public_key(secret_key)[0]))