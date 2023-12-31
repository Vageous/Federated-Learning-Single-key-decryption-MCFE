import numpy as np
import random

def sample_hamming_weight_vector(num_samples):
  sample = np.zeros(num_samples)
  total_weight = 0

  while total_weight < num_samples:
    index = random.randrange(0, num_samples)
    if sample[index] == 0:
      r = random.randrange(0, 2)
      if r == 0: sample[index] = -1
      else: sample[index] = 1
      total_weight += 1

  return sample

def discrete_gaussian(n):# 标准正态分布
    coeffs = np.round(np.random.normal(0,1,n))
    return coeffs

def discrete_gaussian1(n):
    coeffs=np.round(np.random.normal(0,3,n))# 方差较大的正态分布
    return coeffs

def sample_from_uniform_distribution(min_val, max_val, num_samples):#均匀分布
  assert(num_samples > 0 & isinstance(num_samples, int))

  return np.array([random.randrange(min_val, max_val) for _ in range(num_samples)])

def sample_from_triangle(num_samples):
  sample = np.zeros(num_samples)

  for i in range(num_samples):
    r = random.randrange(0, 4)
    if r == 0: sample[i] = -1
    elif r == 1: sample[i] = 1

  return sample

# 将模数变为(-p/2, p/2]
def crange(coeffs, q):
    coeffs = np.where((coeffs >= 0) & (coeffs <= q//2),
                      coeffs,
                      coeffs - q)

    return coeffs