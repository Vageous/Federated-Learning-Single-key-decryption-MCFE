import numpy as np
from numpy.polynomial import Polynomial
from Ciphertext import Ciphertext
from CKKSKeyGenerator import CKKSKeyGenerator
from CKKSEncrypter import CKKSEncrypter
from CKKSDecrypter import CKKSDecrypter
from CKKSAggregate import CKKSAggregate
from CKKSEncoder import CKKSEncoder
import torch



# 参数设置：CKKS方案中主要有三个参数: 
# poly_modulus_degree(polynomial modulus):必须为2的幂，编码后的明文是 poly_modulus / 2
# coefficient modulus: 多项式系数模
# scale:编码时的缩放因子
# 8192 200bits 2^40

# 参数选取十分重要，否则会无法正确近似解密

q=1023928107044561 #50bits
ckkskey=CKKSKeyGenerator(2048,2)
encrypt=CKKSEncrypter(2048,q,2,ckkskey.public_key)
# 编码
m1=[-0.00001,-0.02,-0.003,0.004,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]
m2=[-0.00001,-0.02,-0.003,0.004,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]
m3=[]
for i in range(576):
    m3.append(0.001)
encoder=CKKSEncoder(4 * 576, pow(2,30))
plaintext1=encoder.encode(m3)
# print(len(plaintext1))
# print(plaintext1)
plaintext2=encoder.encode(m2)
# 加密s
C_sum=[]
cipher=[]
for i in range(3):
    cipher.append(encrypt.encrypt(plaintext1))
    cipher.append(encrypt.encrypt(plaintext1))
    cipher.append(encrypt.encrypt(plaintext1))
aggregate=CKKSAggregate(2048,q)
for i in range(3):
    C=aggregate.aggregate(cipher[i],C_sum)

# print(cipher.c0)
# 解密
decrypt=CKKSDecrypter(2048,ckkskey.secret_key,2,q)

u = Polynomial(torch.zeros(2048))
for i in range(3):
    u += decrypt.PartDec(cipher[i])

plain1=decrypt.Merge(C,u)
print(plain1)
# 解码
encoder1=CKKSEncoder(4*576, pow(2,30))
encoder1.decode(plain1)
print(encoder1.decode(plain1).real)

