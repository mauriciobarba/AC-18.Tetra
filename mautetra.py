import numpy as np
# alias run="python3 -i ./AC-18.Tetra/mautetra.py"
def find_Dmatrix(e12,e13,e14,e23,e24,e34):
  """Finds the D matrix described in Wirth-Dreiding from the given edge lengths"""
  D = np.array([[0,e12**2,e13**2,e14**2,1],
  [e12**2,0,e23**2,e24**2,1],
  [e13**2,e23**2,0,e34**2,1],
  [e14**2,e24**2,e34**2,0,1],
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
  result = np.linalg.det(bigmat)
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
  bigmat = sign * minor(D,k,l)
  result = np.linalg.det(bigmat)
  return result

def check_tetra(e12,e13,e14,e23,e24,e34):
  """Checks this sextuple can determine a tetrahedron by
  lemma 4 of Wirth-Dreiding. This lemma is not a
  sufficiency"""
  D = find_Dmatrix(e12,e13,e14,e23,e24,e34)
  if np.linalg.det(D) > 0:
    det_condition = True
  else:
    det_condition = False
  unsquareD = np.sqrt(D)
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
def find_relation():
  """Given the D matrix, find the appropriate relations"""
  MAX_GUESSES = 0
  EPS = 0.000001
  MAX_INT_GUESS = 1000
  prod = 0
  iterate = 0
  def max_iter(iterate):
    if MAX_GUESSES == 0:
      return True
    else:
      return iterate < MAX_GUESSES
  while max_iter(iterate):
    iterate += 1
    edges = np.random.randint(1,MAX_INT_GUESS,(1,6)).tolist()[0]
    if not check_tetra(*edges):
      continue
    prod = 1
    D = find_Dmatrix(*edges)
    unsquareD = np.sqrt(D)
    for i in range(1,4):
      for j in range(i+1,5):
        notin = [h for h in range(1,5) if h not in [i,j]]
        k = notin[0]
        l = notin[1]
        d = (D_ij(D,i,j))**2/(D_ijk(D,i,j,k)*D_ijk(D,i,j,l))
        b = 4*d - 2
        w = (b+np.sqrt(complex(b**2-4)))/2
        prod *= w**(unsquareD[i-1,j-1])
    checksum = np.absolute(prod)
    print(prod,checksum,np.absolute(prod-1),iterate)
    if np.absolute(prod-1)<EPS:
      return (edges,np.absolute(prod-1),prod,iterate)
  return None


testmat = find_Dmatrix(1,1,1,1,1,1)
othertest = find_Dmatrix(1,2,1,2,1,2)
value = find_relation()
