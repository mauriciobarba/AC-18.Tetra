#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Updated on on Jan 24 2021

@author: anaschentouf
"""
from sympy import *
from itertools import permutations, combinations
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
from sympy.simplify.fu import fu, TR10i
from sympy import init_printing
from sympy import print_latex
from sympy import latex
from sympy import Symbol
from sympy.printing.latex import LatexPrinter, print_latex
from sympy.solvers.inequalities import solve_rational_inequalities
from sympy import Poly
from sympy import simplify, cos, sin, pi
from sympy import trigsimp,cancel
from sympy import sqrt, simplify, count_ops, oo
init_printing(use_latex='mathjax')

def cosine_function(x,y,z): 
    """
    function that gives the planar angle defined by the three dihedral angles x,y,z;

    """
    numerator= cos(z)+cos(x)*cos(y)
    denominator= sin(x)*sin(y)
    return acos(numerator/denominator)

def list_to_dict(l): 
    """
    converts a list l of 6 dihedral to a dictionary that can be used to easily map
    the list is assumed to be of length 6, 
    the dihedral angles must be in the order 12,13,14,23,24,34 
    NOTE THAT THIS IS DIFFERENT FROM THE NOTATION USED IN THE PAPER
    """
    dict={}
    copy=l.copy()
    for i in range(1,4):
        for j in range(i+1,5):
            dict[(i,j)]=copy[0]
            copy.remove(copy[0])
    for j in range(1,4):
        for i in range(j+1,5):
            dict[(i,j)]=dict[(j,i)]
    return dict
            

def complement(i,j,k): 
    """
    Input:
    i,j elements of {1,2,3,4}
    returns the only element of (1,2,3,4) not in the SET {i,j}
    """
    list=[1,2,3,4]
    list.remove(i)
    list.remove(j)
    list.remove(k)
    return list[0]

def doublecomplement(i,j):
    """
    returns the two element of (1,2,3,4) not in the input
    """
    list=[1,2,3,4]
    list.remove(i)
    list.remove(j)
    return list
    

def dihed_to_planar(i,j,k,dict):
    """
    computs the dihedral angle i,j,k
    """
    l=complement(i,j,k)
    return cosine_function( dict[(j,k)], dict[(j,i)], dict[(j,l)])

perms=[] #to avoid redundancy, we only consider angles (i,j,k) so that i<k
for i in permutations([1,2,3,4], 3):
    if i[0]<i[2]:
        perms.append(i)
        


def overall(dihedrals=[acos(1/3)]*6):
    """
    Returns the planar angles in order. 
    """
    planar={}
    for tupl in perms:
        planar[tupl]=dihed_to_planar(tupl[0], tupl[1], tupl[2], list_to_dict(dihedrals))
    return planar


edges=[] #to avoid redundancy, we only consider sides (i,j) so that i<j
for i in permutations([1,2,3,4], 2):
        edges.append(i)
        
def calculate_lengths(dihedrals=[acos(1/3)]*6):
    planar=overall(dihedrals)
    def sine_law(i,j,k, planar, length):
        """
    function that applies the sine law to the triangle i,j,k; where the length of the edge ij 
    is given and 
    Input:
        i,j,k: distinct numbers in the set {1,2,3,4} corresponding to vertives, and 
        planar: a dictionary telling us what each of the tetrahedron's planar angles are.  
        length: a (possible incomplete) dictionary of edge lengths in the terahedron
    Output:
        None, but the function adds the lengths of sides ik and jk to the dictionary""" 
        d=length[(i,j)]/sin(planar[(min(i,j),k,max(i,j))])
        length[(i,k)]=d* sin(planar[(min(i,k),j,max(i,k))])
        length[(k,i)]=length[(i,k)]
        length[(j,k)]=d* sin(planar[(min(j,k),i,max(j,k))])
        length[(k,j)]=length[(j,k)]
    length={}
    planar=overall(dihedrals)
    length[(1,4)]=1
    length[(4,1)]=1
    sine_law(1,4,2, planar, length) #Sine Law on Triangle (124)
    for side in edges: 
        if side in length: 
            continue
        else: 
            for k in doublecomplement(side[0], side[1]):
                if (side[0],k) in length:
                    sine_law(side[0],k,side[1],planar, length)
                    length[(side[1],side[0])]=length[(side[0],side[1])]
                    length[(side[1],k)]=length[(k,side[1])]
                elif (k,side[0]) in length:
                    sine_law(k,side[0],side[1],planar, length)
                    length[(side[1],side[0])]=length[(side[0],side[1])]
                    length[(side[1],k)]=length[(k,side[1])]
    return length
        
def rotation(A,x,s):
    """
    provides the result when point x is rotated counterclockwise about A by an angle of s
    A,x are numpy coordinates
    s is in radians
    """
    rot_s=np.array([[cos(s), -sin(s)], [sin(s), cos(s)]]) #Remember Chapter 6 ;)
    return (rot_s.dot(x-A))+A



#THE FOLLOWING get_intersect() FUNCTION WAS OBTAINED FROM THE  STACK OVERFLOW THREAD
#https://stackoverflow.com/questions/3252194/numpy-and-line-intersections, answered by Norbu Tsering 

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return np.array([x/z, y/z])

def thirdpoint(A,B, s,t):
    """
    function that given the endpoints of segment AB, finds the UNIQUE point P on the plane so that 
    dir(BAP)=s and dir(PBA)=t, where dir refers to the directed angle mod pi, and x,y are ... in radians!
    A,B are each a tuple of coordinate
    x,y are angles in radian
    """
    X=rotation(A,B,s)
    Y=rotation(B,A,-t)
    return get_intersect(A, X, B, Y)


def connect(A,B,C,linetype="dotted",clr="blue"):
    """
        input: 
        A,B,C: each is a coordinate (numpy array) of a point in R^2
        clr: a string for the color to be used, e.g. "blue"
    output:
        None, but the function draws the segments AB and BC of the desired linetype and color
    """
    plt.plot([A[0], B[0]], [A[1], B[1]], ls=linetype, lw="0.5",color=clr)
    plt.plot([C[0], B[0]], [C[1], B[1]], ls=linetype, lw="0.5",color=clr)


def tri_connect(A,B,C,linetype="solid",clr="black"):
    """
    input: 
        A,B,C: each is a coordinate (numpy array) of a point in R^2
        clr: a string for the color to be used, e.g. "blue"
    output:
        None, but the function draws the sides of triangle ABC
        using a dotted segments in a plot
    """
    plt.plot([A[0], B[0]], [A[1], B[1]], ls=linetype, lw="0.2",color=clr)
    plt.plot([C[0], B[0]], [C[1], B[1]], ls=linetype, lw="0.2",color=clr)
    plt.plot([A[0], C[0]], [A[1], C[1]], ls=linetype, lw="0.2",color=clr)
    
    

def get_positions(planar, returner=False):
    position={} #initializes new dictionary
    position[1]=np.array([0, 0])
    position[4]=np.array([1,0])
    position[2]=thirdpoint(position[1],position[4], planar[(2,1,4)], planar[(1,4,2)])
    three1=thirdpoint(position[2],position[4], planar[(3,2,4)], planar[(2,4,3)])
    three2=thirdpoint(position[1],position[2], planar[(2,1,3)], planar[(1,2,3)])
    three3=thirdpoint(position[1],position[4], -planar[(3,1,4)], -planar[(1,4,3)])
    X=[position[1], position[2], position[4], three1, three2, three3]
    plt.figure(figsize=(3,4))
    plt.scatter([t[0] for t in X], [t[1] for t in X],color='blue',s=5)
    tri_connect(position[1], position[2], position[4])
    connect(position[2], three1, position[4])
    connect(position[1], three3, position[4])
    connect(position[1], three2, position[2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('filename.png', dpi=300)
    plt.show()
    if returner:
        return (position[1], position[2], position[4])
        


def family1(x):
    """
    Input:
        x, a rational multiple of pi such that pi/6<x<pi/2
    Output:
        A list of the dihedral angles in the tetrahedron
    """
    return [pi/2,pi-2*x,x,x,pi/3,pi/2]

def family2(x):
    """
    Input:
        x, a rational number such that pi/6<x=<pi/3
    Output:
        A list of the dihedral angles in the tetrahedron
    """
    return [5*pi/6-x,2*(pi/3)-x,x,x,2*pi/3-x,pi/6+x]


def calculate_sixth(dihedrals):
    """
    Input:
        dihedrals: a list of 5 angles corresponding to 12, 13, 14, 23, 24 (notice the order)
    Output: 
        list of all 6 dihedrals in the order 12, 13, 14, 23, 24
    """
    angle132=cosine_function(dihedrals[0], dihedrals[3], dihedrals[4])
    angle213=cosine_function(dihedrals[0], dihedrals[1], dihedrals[2])
    supplement=pi-angle213-angle132
    u=sin(dihedrals[1])*sin(dihedrals[3])*cos(supplement)-cos(dihedrals[1])*cos(dihedrals[3])
    dihedrals.append(acos(u))
    return dihedrals

def get_3d(planar):
    pos=get_positions(planar, True)
    v12=pos[1]
    x=lengths[(1,3)]*cos(planar[(3,1,4)])
    y=(lengths[(1,3)]*lengths[(1,2)]*cos(planar[(2,1,3)])-v12[0]*x)/v12[1]
    z=sqrt(lengths[(1,3)]**2-x**2-y**2)
    position3=(x,y,z)
    return ( tuple(pos[0]),tuple(pos[1]),position3, tuple(pos[2]))

def check_congruence(dihedrals1,dihedrals2):
    """
    Parameters
    Dihedral angles of both tetrahedra, both in the order [12,13,14,23,24,34]
    
    Returns:
    All pairs of confruent faces F1, F2 such that F1 is on Tetrahedron 1, and F2 is on Tetrahedron 2
    """
    length1=calculate_lengths(dihedrals1)
    length2=calculate_lengths(dihedrals2)
    list_congruences=[]
    for face1 in permutations([1,2,3,4], 3):
        for face2 in permutations([1,2,3,4], 3):
            A1,B1,C1=face1
            A2,B2,C2=face2
            if abs(length1[(A1,B1)]-length2[(A2,B2)])<1e-10:
                if abs(length1[(A1,C1)]-length2[(A2,C2)])<1e-10:
                    if abs(length1[(B1, C1)]-length2[(B2,C2)])<1e-10:
                        list_congruences.append((face1, face2))
    return list_congruences  

#The Follow section is a symbolically computational implementation of the first section's results of 
#Wirth-Dreiding paper in J. Math Chem., which will be useful in transitioning form lengths to dihedral angles 
#(although one can also directly reverse the process used in the first part of the code...)



def WD_matrix(lengths):
        """
    Parameters
    List of Edgelengths [12,13,14,23,24,34]
    
    Returns:
    The D matrix!
    """
    D=Matrix([[0, lengths[0]**2, lengths[1]**2, lengths[2]**2, 1], 
         [lengths[0]**2, 0, lengths[3]**2, lengths[4]**2, 1],
         [lengths[1]**2, lengths[3]**2, 0, lengths[5]**2, 1],
         [lengths[2]**2, lengths[4]**2,lengths[5]**2,0,1],
         [1,1,1,1,0]])
    return D


def check_welldefined(lengths):
    """
    Parameters
    List of Edgelengths [12,13,14,23,24,34]
    
    Returns:
    True if the edge lengths define a non-degenerate tetrahedron
    False Otherwise
    """
    D=WD_matrix(lengths)
    if D.det()>0:
        return True

def find_welldefinedrange(lengths):
    """
    Parameters
    lengths: List of Edgelengths [12,13,14,23,24,34]; one or more of which is a symbol (variable).
    This can be initialized using x=symbols("x").
    Returns:
    Conditions for which the tetrahedron is non-degenerate. 
    The reader must only take the positive edge lengths... 
    (generally in the form of an inequality or a solution to it)
    """
    D=WD_matrix(lengths)
    return solve(D.det()>0, domain='RR')         

def D_3(i,j,k, lengths):
    """
    Parameters
    lengths:List of Edgelengths [12,13,14,23,24,34]; one or more of which is a symbol (variable).
    Distinct i,j,k in {1,2,3,4}
    Returns:
    D_ijk, as defined in WD's paper
    """
    l=complement(i, j, k)
    M=WD_matrix(lengths)
    M_new=M.copy()
    M_new.col_del(l-1)
    M_new.row_del(l-1)
    return M_new.det()

def D_2(i,j, lengths):
    """
    Parameters
    lengths: List of Edgelengths [12,13,14,23,24,34]; one or more of which is a symbol (variable).
    Distinct i,j in {1,2,3,4}
    Returns:
    D_ij, as defined in WD's paper
    """
    k,l=doublecomplement(i, j)
    M=WD_matrix(lengths)
    M_new=M.copy()
    M_new.col_del(k-1)
    M_new.row_del(l-1)
    return (-1)**(k+l)*M_new.det()

def lengths_to_dihedrals(lengths):
    angles={}
    for i in range(1,5):
        for j in range(1,5):
            if i!=j:
                k,l=doublecomplement(i, j)
                angles[(i,j)]=acos(D_2(i,j,lengths)/sqrt( D_3(i,j,k, lengths)* D_3(i,j,l, lengths)))
                
    return angles


#TO RUN, USE THE FOLLOWING SECTION (AND CHANGE THE DIHEDRALS ANGLES, IF NECESSARY)
#RECALL THAT THE ORDER IS 12,13,14,23,24,34
#You can generate families of tetrahedra with rational angles, as in Theorem 1.8 of 
#Professor Poonen's paper (http://math.mit.edu/~poonen/papers/space_vectors.pdf), using the two
#family functions
# x=symbols("x")
# dihedrals=[S("acos(1/3)")]*6 #parantheses instruct Sympy to use the exact value
# dihedrals=family1(x)
dihedrals=[pi/7,pi/3,4*pi/7,4*pi/7, pi/3, 3*pi/7]
# dihedrals=[pi/5, pi/3,pi/2, pi/2, pi-2*pi/5, pi/5]
print("Congruences in Faces:", check_congruence(dihedrals, dihedrals))
# planar=overall(dihedrals)
# for i in planar: 
#     print(i, simplify(planar[i]))
# lengths=calculate_lengths(dihedrals)
# for i in lengths:
#     print(i, lengths[i])
    



                    
