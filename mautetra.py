#!/usr/bin/env python3
import numpy as np
import sys
from math import gcd, sqrt
import math

# alias run="python3 -i ./AC-18.Tetra/mautetra.py"
def find_Dmatrix(e12,e13,e14,e23,e24,e34):
  """Finds the D matrix described in Wirth-Dreiding from the given edge lengths"""
  D = np.array([[0,e12**2,e13**2,e14**2,1],
  [e12**2,0,e23**2,e24**2,1],
  [e13**2,e23**2,0,e34**2,1],
  [e14**2,e24**2,e34**2,0,1],
  [1,1,1,1,0]])
  return D

def find_Dunsquare(e12,e13,e14,e23,e24,e34):
  """Finds D matrix but the values aren't squared. Pretty helpful for some computations."""
  D = np.array([[0,e12,e13,e14,1],
  [e12,0,e23,e24,1],
  [e13,e23,0,e34,1],
  [e14,e24,e34,0,1],
  [1,1,1,1,0]])
  return D

def minor(arr,i,j):
    """Removes the ith row and jth colum of arr. indexing here starts at
    1 istead of zero"""
    return arr[np.array(list(range(i-1))+list(range(i,arr.shape[0])))[:,np.newaxis],
               np.array(list(range(j-1))+list(range(j,arr.shape[1])))]

def D_ijk(D,i,j,k):
  """Finds the D_ijk matrix from D matrix (as described in Wirth-Dreiding)"""
  i_in_range = 1 <= i and i <=4
  j_in_range = 1 <= j and j <=4
  k_in_range = 1 <= k and k <=4
  if not (i_in_range and j_in_range and k_in_range):
    raise Exception('i,j,k must be between 1 and 4 inclusive')
  if i==j or j==k or i==k:
    raise Exception('i,j,k must be all distinct')
  l = [h for h in range(1,5) if h not in [i,j,k]][0]
  bigmat = minor(D,l,l)
  result = round(np.linalg.det(bigmat))
  return result

def D_ij(D,i,j):
  """Finds the D_ij matrix from D matrix (as described in Wirth-Dreiding)"""
  i_in_range = 1 <= i and i <=4
  j_in_range = 1 <= j and j <=4
  if not (i_in_range and j_in_range):
    raise Exception('i,j,k must be between 1 and 4 inclusive')
  if i==j:
    raise Exception('i,j must be all distinct')
  notin = [h for h in range(1,5) if h not in [i,j]]
  l = notin[0]
  k = notin[1]
  if k+l%2 == 0:
    sign = 1
  else:
    sign = -1
  bigmat = minor(D,k,l)
  result = sign * round(np.linalg.det(bigmat))
  return result

def check_tetra(e12,e13,e14,e23,e24,e34):
  """Checks whether this sextuple describes a tetrahedron. If and only if"""
  D = find_Dmatrix(e12,e13,e14,e23,e24,e34)
  if round(np.linalg.det(D)) > 0:
    det_condition = True
  else:
    det_condition = False
  unsquareD = find_Dunsquare(e12,e13,e14,e23,e24,e34)
  possible_triangles = [(1,2,3),(1,2,4),(1,3,4),(2,3,4)]
  triple_face_condition = False
  for tup in possible_triangles:
    inequality1 = unsquareD[tup[0],tup[1]] < unsquareD[tup[1],tup[2]]+unsquareD[tup[0],tup[2]]
    inequality2 = unsquareD[tup[1],tup[2]] < unsquareD[tup[0],tup[1]]+unsquareD[tup[0],tup[2]]
    inequality3 = unsquareD[tup[0],tup[2]] < unsquareD[tup[1],tup[2]]+unsquareD[tup[0],tup[1]]
    total_truth = inequality1 and inequality2 and inequality3
    if total_truth:
      triple_face_condition = True
      break
  if det_condition and triple_face_condition:
    return True
  return False

def w_val_p(a,p):
  """Gets the p-adic valuation of w. Needs some improvement."""
  k1 = 1
  while a % p**k1 == 0:
    k1 += 1
  k1 -= 1
  return k1

def one_combination_matrix(n):
  result = -np.ones((0,n))
  for i in range(2**n):
    newarr = -np.ones((1,n))
    for idx, char in enumerate(bin(i)[::-1][:-2]):
      if char == '1':
        newarr[0,idx] = 1
    result = np.vstack((result,newarr))
  return result


def get_coeff(e12,e13,e14,e23,e24,e34):
  """Gets the squared cosine according to Wirth Dreiding"""
  D = find_Dunsquare(e12,e13,e14,e23,e24,e34)
  cos = [[0 for _ in range(0,5)] for _ in range(0,5)]
  for i in range(1,4):
    for j in range(i+1,5):
      notin = [h for h in range(1,5) if h not in [i,j]]
      k = notin[0]
      l = notin[1]
      cos[i][j] = (D_ij(D,i,j),D_ijk(D,i,j,k)*D_ijk(D,i,j,l))
  return [
    '{}/{}'.format(cos[1][2][0]**2//gcd(cos[1][2][0]**2,cos[1][2][1]),cos[1][2][1]//gcd(cos[1][2][0]**2,cos[1][2][1])),
    '{}/{}'.format(cos[1][3][0]**2//gcd(cos[1][3][0]**2,cos[1][3][1]),cos[1][3][1]//gcd(cos[1][3][0]**2,cos[1][3][1])),
    '{}/{}'.format(cos[1][4][0]**2//gcd(cos[1][4][0]**2,cos[1][4][1]),cos[1][4][1]//gcd(cos[1][4][0]**2,cos[1][4][1])),
    '{}/{}'.format(cos[2][3][0]**2//gcd(cos[2][3][0]**2,cos[2][3][1]),cos[2][3][1]//gcd(cos[2][3][0]**2,cos[2][3][1])),
    '{}/{}'.format(cos[2][4][0]**2//gcd(cos[2][4][0]**2,cos[2][4][1]),cos[2][4][1]//gcd(cos[2][4][0]**2,cos[2][4][1])),
    '{}/{}'.format(cos[3][4][0]**2//gcd(cos[3][4][0]**2,cos[3][4][1]),cos[3][4][1]//gcd(cos[3][4][0]**2,cos[3][4][1]))
    ]

def get_prime_factors(number):
  prime_factors = []
  while number % 2 == 0:
    prime_factors.append(2)
    number = number / 2
  for i in range(3, int(math.sqrt(number)) + 1, 2):
    while number % i == 0:
      prime_factors.append(int(i))
      number = number / i
  if number > 2:
    prime_factors.append(int(number)) 

  return prime_factors

def get_poly_coeffs_denom(e12,e13,e14,e23,e24,e34):
  """Gets the squared cosine according to Wirth Dreiding. Not quite the square but the coefficient of the polynomial"""
  D = find_Dmatrix(e12,e13,e14,e23,e24,e34)
  cos = [[0 for _ in range(0,5)] for _ in range(0,5)]
  for i in range(1,4):
    for j in range(i+1,5):
      notin = [h for h in range(1,5) if h not in [i,j]]
      k = notin[0]
      l = notin[1]
      cos[i][j] = (D_ij(D,i,j),D_ijk(D,i,j,k)*D_ijk(D,i,j,l))
  return [
    cos[1][2][1]//gcd(4*cos[1][2][0]**2,cos[1][2][1]),
    cos[1][3][1]//gcd(4*cos[1][3][0]**2,cos[1][3][1]),
    cos[1][4][1]//gcd(4*cos[1][4][0]**2,cos[1][4][1]),
    cos[2][3][1]//gcd(4*cos[2][3][0]**2,cos[2][3][1]),
    cos[2][4][1]//gcd(4*cos[2][4][0]**2,cos[2][4][1]),
    cos[3][4][1]//gcd(4*cos[3][4][0]**2,cos[3][4][1])
    ]

def check_result_p_adic(e12,e13,e14,e23,e24,e34):
  edges = [e12,e13,e14,e23,e24,e34]
  print(edges)
  set_o_primes = set({})
  if not check_tetra(*edges):
    return None
  D = find_Dmatrix(*edges)
  unsquareD = find_Dunsquare(*edges)
  poly_coeffs = get_poly_coeffs_denom(*edges)
  for coeff in poly_coeffs:
    for p in get_prime_factors(coeff):
      set_o_primes.add(p)
  print(set_o_primes)
  for p in set_o_primes:
    val_arr = np.zeros((0,1))
    filtered_edges = np.zeros((0,1))
    num_edges = 0
    if w_val_p(poly_coeffs[0],p)>0:
      val_arr = np.vstack((val_arr,np.array([[w_val_p(poly_coeffs[0],p)]])))
      filtered_edges = np.vstack((filtered_edges,np.array([[edges[0]]])))
      num_edges += 1
    if w_val_p(poly_coeffs[1],p)>0:
      val_arr = np.vstack((val_arr,np.array([[w_val_p(poly_coeffs[1],p)]])))
      filtered_edges = np.vstack((filtered_edges,np.array([[edges[1]]])))
      num_edges += 1
    if w_val_p(poly_coeffs[2],p)>0:
      val_arr = np.vstack((val_arr,np.array([[w_val_p(poly_coeffs[2],p)]])))
      filtered_edges = np.vstack((filtered_edges,np.array([[edges[2]]])))
      num_edges += 1
    if w_val_p(poly_coeffs[3],p)>0:
      val_arr = np.vstack((val_arr,np.array([[w_val_p(poly_coeffs[3],p)]])))
      filtered_edges = np.vstack((filtered_edges,np.array([[edges[3]]])))
      num_edges += 1
    if w_val_p(poly_coeffs[4],p)>0:
      val_arr = np.vstack((val_arr,np.array([[w_val_p(poly_coeffs[4],p)]])))
      filtered_edges = np.vstack((filtered_edges,np.array([[edges[4]]])))
      num_edges += 1
    if w_val_p(poly_coeffs[5],p)>0:
      val_arr = np.vstack((val_arr,np.array([[w_val_p(poly_coeffs[5],p)]])))
      filtered_edges = np.vstack((filtered_edges,np.array([[edges[5]]])))
      num_edges += 1
    if not np.any((one_combination_matrix(num_edges)*(val_arr.T))@filtered_edges==0):
      return None
  return edges

def check_result_numerical(e12,e13,e14,e23,e24,e34):
  """Checks if the edges we furnished actually make a tetrahedron"""
  EPS = 1e-10
  edges = [e12,e13,e14,e23,e24,e34]
  if not check_tetra(*edges):
    return None
  prod = 1
  D = find_Dmatrix(*edges)
  unsquareD = find_Dunsquare(*edges)
  for i in range(1,4):
    for j in range(i+1,5):
      notin = [h for h in range(1,5) if h not in [i,j]]
      k = notin[0]
      l = notin[1]
      d = (D_ij(D,i,j))**2/(D_ijk(D,i,j,k)*D_ijk(D,i,j,l))
      b = 4*d - 2
      w = (b+np.sqrt(complex(b**2-4)))/2
      prod *= w**(24*unsquareD[i-1,j-1])
  print(edges, prod)
  imag_pos = prod.imag if prod.imag > 0 else -prod.imag
  if imag_pos < EPS and prod.real > 0:
    return edges
  return None

def iterate_edges(form):
  """
  Checks if edge lengths satisfy the property using numerical
  or p-adic analysis. Numerical: form='numerical', p-adic: form='p-adic'.
  """
  EPS = 1e-10
  MAX_EDGE_GUESS = 100
  try:
    while True:
      edges = np.random.randint(1,MAX_EDGE_GUESS,(1,6)).tolist()[0]
      if form == 'numerical':
        result = check_result_numerical(*edges)
        original_stdout = sys.stdout
        with open('tetranumerical.txt', 'a+') as f:
          sys.stdout = f
          if not result is None:
            print(result)
          sys.stdout = original_stdout
      elif form == 'p-adic':
        result = check_result_p_adic(*edges)
        original_stdout = sys.stdout
        with open('tetrapadic.txt', 'a+') as f:
          sys.stdout = f
          if not result is None:
            print(result)
          sys.stdout = original_stdout
      else:
        raise Exception('Not a valid form')
  except KeyboardInterrupt:
    print('\nEnded Successfully')

if __name__ == '__main__':
  iterate_edges('p-adic')