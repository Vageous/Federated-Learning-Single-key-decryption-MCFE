import numpy as np
from numpy.polynomial import Polynomial
import utils
from Ciphertext import Ciphertext
from PublicKey import PublicKey
from CKKSKeyGenerator import CKKSKeyGenerator

class CKKSEncrypter:
  def __init__(self, poly_degree, cipher_modulus, big_modulus, public_key: PublicKey):
    self.poly_degree = poly_degree
    self.cipher_modulus = cipher_modulus
    self.big_modulus = big_modulus
    # self.crt_context = crt_context
    self.public_key = public_key
    # self.secret_key = secret_key

  def encrypt(self, plaintext):
    p0 = self.public_key.p0# 用户公钥
    p1 = self.public_key.p1# 公开参数a


    random_poly = Polynomial(utils.discrete_gaussian(self.poly_degree))
    err0 = Polynomial(utils.discrete_gaussian(self.poly_degree))
    err1 = Polynomial(utils.discrete_gaussian(self.poly_degree))

    poly_modulus = Polynomial([1] + [0] * (self.poly_degree-2) + [1])

    cipher1 = random_poly * p0.coef[0] + plaintext + err0
    # print(cipher1)
    cipher1 = Polynomial(utils.crange((cipher1.coef % self.cipher_modulus), self.cipher_modulus))
    # print(cipher1)
    # cipher1 = cipher1._div(cipher1,self.cipher_modulus)
    cipher2 = random_poly * p1.coef[0] + err1
    cipher2 = Polynomial(utils.crange((cipher2.coef % self.cipher_modulus), self.cipher_modulus))
    # c0 = p0 * random_poly % poly_modulus
    # c0 += self.cipher_modulus * err0
    # c0 += plaintext
    # c0 %= self.big_modulus

    # c1 = p1 * random_poly % poly_modulus
    # c1 += self.cipher_modulus * err1
    # c1 %= self.big_modulus

    return Ciphertext(cipher1, cipher2)

# ckks=CKKSKeyGenerator(8,2,41)
# encrypt=CKKSEncrypter(8,41,2,ckks.public_key)
# cipher=encrypt.encrypt([1,2,3,4,5,6,7,8])
# print(cipher.c0)
# print(cipher.c1)




