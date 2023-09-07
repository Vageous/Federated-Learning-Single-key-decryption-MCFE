import random
import numpy as np
from numpy.polynomial import Polynomial
import utils
from PublicKey import PublicKey

class CKKSKeyGenerator:
  def __init__(self, poly_degree, big_modulus):
    self.poly_degree = poly_degree# 多项式阶数，即N, 是2的幂次
    self.big_modulus = big_modulus# 均匀分布的上界
    # self.cipher_modulus = cipher_modulus# 多项式模数，即q
    self.generate_secret_key()
    self.generate_public_key()

  def generate_secret_key(self):
    # key = utils.sample_hamming_weight_vector(self.poly_degree)
    key = utils.discrete_gaussian(self.poly_degree)# 从标准正态分布中选取私钥
    self.secret_key = Polynomial(key)
    # return self.secret_key

  def generate_public_key(self):# big_modulus 是均匀分布的上界
    coeff = utils.sample_from_uniform_distribution(0, self.big_modulus, self.poly_degree)# 公共参数a
    pk_coeff = Polynomial(coeff)# 公共参数a对应的多项式
    err = utils.discrete_gaussian(self.poly_degree)# 噪声e
    pk_err = Polynomial(err)# 噪声e对应的多项式
    poly_modulus = Polynomial([1] + [0] * (self.poly_degree-2) + [1]) # x^n+1
    p0 = pk_coeff * self.secret_key % poly_modulus# a*s mod x^n+1
    # p0 %= self.big_modulus# a*s对应的多项式系数 mod big_modulus
    p0 *= -1# -(a*s)
    p0 += pk_err# -(a*s)+e
    # p0 %= self.big_modulus# -(a*s)+e mod big_modulus
    p1 = pk_coeff# 公开参数a的多项式
    self.public_key = PublicKey(p0, p1)
    # return p0,p1# (-a*s+e, a)



# ckks=CKKSKeyGenerator(8,2,41)
# # print(ckks.generate_secret_key())
# print(ckks.generate_public_key())