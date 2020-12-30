#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:42:47 2020
@author: anaschentouf

18.TETRA, Project 3

The following code aims to allow users to compute the lengths and planar angles in a tetrahedron 
given the dihedral angles. 
"""

import math
from itertools import permutations 
import numpy as np
import matplotlib.pyplot as plt

def cosine_function(x,y,z): 
    """
    function that gives the planar angle defined by the three dihedral angles x,y,z;

    """
    numerator= math.cos(z)+math.cos(x)*math.cos(y)
    denominator= math.sin(x)*math.sin(y)
    return math.acos(numerator/denominator)

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


def overall(dihedrals=[math.acos(1/3)]*6):
    """
    Returns the planar angles in order. If an angle has radian measure q*pi, only q is returned. 
    """
    planar={}
    for tupl in perms:
        planar[tupl]=dihed_to_planar(tupl[0], tupl[1], tupl[2], list_to_dict(dihedrals))/math.pi
    return planar


edges=[] #to avoid redundancy, we only consider sides (i,j) so that i<j
for i in permutations([1,2,3,4], 2):
        edges.append(i)
        
def calculate_lengths(dihedrals=[math.acos(1/3)]*6):
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
        d=length[(i,j)]/math.sin(math.pi*planar[(min(i,j),k,max(i,j))])
        length[(i,k)]=d* math.sin(math.pi*planar[(min(i,k),j,max(i,k))])
        length[(k,i)]=length[(i,k)]
        length[(j,k)]=d* math.sin(math.pi*planar[(min(j,k),i,max(j,k))])
        length[(k,j)]=length[(j,k)]
    length={}
    planar=overall(dihedrals)
    length[(1,4)]=1.0
    length[(4,1)]=1.0
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
    rot_s=np.array([[math.cos(s), -math.sin(s)], [math.sin(s), math.cos(s)]])
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
    plt.plot([A[0], B[0]], [A[1], B[1]], ls=linetype, lw="0.5",color=clr)
    plt.plot([C[0], B[0]], [C[1], B[1]], ls=linetype, lw="0.5",color=clr)
    plt.plot([A[0], C[0]], [A[1], C[1]], ls=linetype, lw="0.5",color=clr)
    
    

def get_positions(planar):
    position={} #initializes new dictionary
    position[1]=np.array([0, 0])
    position[4]=np.array([1,0])
    position[2]=thirdpoint(position[1],position[4], planar[(2,1,4)]*math.pi, planar[(1,4,2)]*math.pi)
    three1=thirdpoint(position[2],position[4], planar[(3,2,4)]*math.pi, planar[(2,4,3)]*math.pi)
    three2=thirdpoint(position[1],position[2], planar[(2,1,3)]*math.pi, planar[(1,2,3)]*math.pi)
    three3=thirdpoint(position[1],position[4], -planar[(3,1,4)]*math.pi, -planar[(1,4,3)]*math.pi)
    X=[position[1], position[2], position[4], three1, three2, three3]
    plt.figure()
    plt.scatter([t[0] for t in X], [t[1] for t in X],color='blue',s=5)
    tri_connect(position[1], position[2], position[4])
    connect(position[2], three1, position[4])
    connect(position[1], three3, position[4])
    connect(position[1], three2, position[2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()


def family1(x):
    """
    Input:
        x, a rational multiple of pi such that pi/6<x<pi/2
    Output:
        A list of the dihedral angles in the tetrahedron
    """
    return [math.pi/2,math.pi-2*x,x,x,math.pi/3,math.pi/2]

def family2(x):
    """
    Input:
        x, a rational number such that pi/6<x=<pi/3
    Output:
        A list of the dihedral angles in the tetrahedron
    """
    return [5*math.pi/6-x,2*(math.pi/3)-x,x,x,2*math.pi/3-x,math.pi/6+x]

#TO RUN, USE THE FOLLOWING SECTION (AND CHANGE THE DIHEDRALS ANGLES, IF NECESSARY)
#RECALL THAT THE ORDER IS 12,13,14,23,24,34
#You can generate families of tetrahedra with rational angles, as in Theorem 1.8 of 
#Professor Poonen's paper (http://math.mit.edu/~poonen/papers/space_vectors.pdf), using the two
#family functions above. The following is an example!


dihedrals=family2(0.2*math.pi)
planar=overall(dihedrals)
for i in planar: #GIVES ANGLES IN DEGREE
    print(i, planar[i]*180)
lengths=calculate_lengths(dihedrals)
for i in lengths:
    print(i, lengths[i])
get_positions(planar)

