import numpy as np
import math
import copy

#Problem 1
def permutation_matrix(permutation):
	a = len(permutation)
	retMatrix = np.zeros((a,a))
	for x in range(a):
		retMatrix[permutation[x],x] = 1
	return retMatrix

#Problem 2
def orthogonalize(U):
	V = []
	for i in range(len(U)):
		temp = U[i]
		for j in range(i):
			temp -= proj(V[j],U[i])
		temp = unitize(temp)
		V.append(temp)
	return V

#computes projection given two arrays
def proj(v1,v2):
	return v1*(np.dot(v2,v1)/np.dot(v1,v1))

#scales a vector so that the norm=1
def unitize(v):
	sum=0
	for x in v:
		sum+=x**2
	sum = math.sqrt(sum)
	return v/sum


#problem 3, echelon form
def echelon_form(M, eps=10**-6):

	for i in range (len(M[0])):  #processing columns
		for j in range(i,len(M)):
			if leadingentryloc(M,j,eps) == i:   #finding the first leading entry of current col, leading entry location is [j,i]
				if not np.isclose(i,j,eps):	#step 2, if i != j, then swap rows i and j
					swaprow(i,j,M)
			break

		for r in range(i+1,len(M)):		#step 3, making zeros so that row r has a zero in column c
			if not np.isclose(M[i][i],0,eps):
				addrow(i,-1*M[r][i]/M[i][i],r,M)

#problem 3, reduced echelon form
def reduced_echelon_form(M, eps=10**-6):
	echelon_form(M,eps)
	for r in range(len(M)):
		c = leadingentryloc(M,r,eps)	#precomputed and stored in a variable so we don't have to keep computing this
		if c != None:
			c=int(c)
			if not np.isclose(M[r][c],0,eps):
				scalerow(r,(1/M[r][c]),M)	#scaling so that leading entry is 1
				for x in range(r):
					addrow(r,-1*M[x][c]/M[r][c],x,M)	#adding row to every row above it so that there are zeros


# finds the index of the leading entry of a row of a given matrix
def leadingentryloc(matrix, row,eps=10**-6):
	for x in range(len(matrix[row])):
		if not np.isclose(matrix[row][x], 0, eps):
			return x
	return None

#add a multiple of a row to another row, modifies the matrix in place
def addrow(row1, multiple, row2, matrix):
	matrix[row2]+=multiple*matrix[row1]

#swaps two rows, modifies the matrix in place
def swaprow(row1, row2, matrix):
	x=copy.deepcopy(matrix[row1])
	matrix[row1]=matrix[row2]
	matrix[row2]=x

#scales a row by a given scalar, modifies the matrix in place
def scalerow(row1, scalar, matrix):
	matrix[row1]*=scalar




