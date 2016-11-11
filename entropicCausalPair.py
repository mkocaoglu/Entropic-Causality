# Entropic Causality
# Input: A text file with two columns, no headers. First column is called X, second column is called Y.
# If no argument is given, program prompts the user to enter the name of a text file.
# User can also give the name of the input file as an argument.

# Algorithm determines the quantization based on the number of unique values and number of samples.
# Estimates the conditional probability tables and calculates exogenous variable with minimum entropy in both X->Y direction and X<-Y direction. 
# The score is an indicator of how confident the algorithm is. Larger scores indicate more confidence. Scores are greater than 0.

# INSTRUCTIONS: Put this file in the same folder as a text file input.txt which has two columns of data

import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import IsolationForest
from numpy.linalg import norm
def read_file(file_name):
	df = pd.read_csv(file_name, header = None,delim_whitespace=True)

	# Following is a test to check if random numbers give a causal direction
	#df = pd.DataFrame(np.random.randn(500,2))
	return df

class CausalPair(object):
	def __init__(self,data):
		self.data = data
		self.nofsamples = data.shape[0]
		self.X = data[0]
		self.Y = data[1]
		self.Xmin = np.min(data[0])
		self.Xmax = np.max(data[0])
		self.Ymin = np.min(data[1])
		self.Ymax = np.max(data[1])

def quantize_data(pair):
	# Find number of states, n based on X, Y first
	# We simply select n as large as possible and make sure on average each state has around 10 samples.
	p = pair.nofsamples
	# We also need the number of unique values to make sure we know when inputs are discrete variables
	# A variable is deemed discrete if noofunique values is less than 1/5th of number of samples
	uniqueX = pair.X.unique()
	uniqueY = pair.Y.unique()
	n = decide_quantization(p,uniqueX,uniqueY)
	deltaX = (pair.Xmax-pair.Xmin)/n
	rulerX = [pair.Xmin+i*deltaX for i in range(0,n-1)]
	rulerX.append(pair.Xmax)
	deltaY = (pair.Ymax-pair.Ymin)/n
	rulerY = [pair.Ymin+i*deltaY for i in range(0,n-1)]
	rulerY.append(pair.Ymax)
	Xq = np.digitize( pair.X, bins = rulerX )
	Yq = np.digitize( pair.Y, bins = rulerY )
	return Xq,Yq,n,p

def decide_quantization(p,uniqueX,uniqueY):
	noUniqueX = len(uniqueX)
	noUniqueY = len(uniqueY)
	discreteX = 0
	discreteY = 0
	if 5*noUniqueX < p:
		discreteX = 1
	if 5*noUniqueY < p:
		discreteY = 1
	# 256 is chosen as the upper limit on the number of states
	n = np.min([256, discreteX*discreteY*np.max([noUniqueX,noUniqueY]) + discreteX*(1-discreteY)*np.max([noUniqueX, p/10]) + discreteY*(1-discreteX)*np.max([noUniqueY, p/10]) + (1-discreteY)*(1-discreteX)*p/10])
	return n

def remove_outliers(df):
	outliers_fraction = 0.005
	p = df.shape[0]
	rng = np.random.RandomState(42)
	classifier = IsolationForest(max_samples=p,
                                        contamination=outliers_fraction,
                                        random_state=rng)																																				
	classifier.fit(df)
	labels = 0.5*classifier.predict(df)+0.5
	df = df[labels==1]
	return df

def estimate_conditionals(Xq,Yq,n,p):
	# Mxy is conditional probability transition matrix X given Y: Mxy(i,j) = P(X=i|Y=j)
	Mxy = np.zeros((n,n))
	Myx = np.zeros((n,n))

	for i in range(0,p):
		x = Xq[i]
		y = Yq[i]
		Mxy[x-1,y-1]+=1
		Myx[y-1,x-1]+=1
		
	u = Mxy.sum(axis = 0) # column sums: also marginal for y
	v = Myx.sum(axis = 0) # marginal for x

	for i in range(0,n):
		if u[i] != 0:
			Mxy[:,i] = Mxy[:,i].astype(np.float)/u[i]
		if v[i] != 0:
			Myx[:,i] = Myx[:,i].astype(np.float)/v[i]
	return Mxy,Myx,u/sum(u),v/sum(v)

def remove_zero_columns(M):
	t=(M==0)
	v=np.all(t,axis=0)
	return M[:,~v]

def entropy_minimizer(Myx,Mxy,n):
	# remove all zero columns
	Mxy = remove_zero_columns(Mxy)
	Myx = remove_zero_columns(Myx)

	nYX = Myx.shape[1]
	nXY = Mxy.shape[1]

	flag = 1
	eYX=[]
	while flag:
		# choose min of max per column
		e = np.min(np.max(Myx,axis=0))
		eYX.append(e)
		Myx[np.argmax(Myx,axis=0),range(nYX)] -= e
		flag = sum(eYX) < 1-10**-9
	flag = 1
	eXY = []
	while flag:
		# choose min of max per column
		e = np.min(np.max(Mxy,axis=0))
		eXY.append(e)
		Mxy[np.argmax(Mxy,axis=0),range(nXY)] -= e
		flag = sum(eXY) < 1-10**-9

	return eYX,eXY # eYX is exogenous entropy for X->Y
def calc_entropy(eYX):
	return sum([-np.log2(i**i) for i in eYX])

def main(file_name):
	df = read_file(file_name)
	# Mildly clean up the data by removing outliers here
	# otherwise a few points can mess up the whole quantization
	# Use an isolation forest fit by scikit learn
	df = remove_outliers(df)
	pair = CausalPair(df)
	Xq,Yq,n,p = quantize_data(pair)

	Mxy, Myx, pY, pX = estimate_conditionals(Xq,Yq,n,p) # some columns of conditional probability tables will be zero, this is fine

	eYX,eXY = entropy_minimizer(Myx,Mxy,n) # 

	hYX = calc_entropy(eYX)
	hXY = calc_entropy(eXY)

	hX = calc_entropy(pX)
	hY = calc_entropy(pY)

	if abs(hYX + hX - (hXY + hY )) < 0.02* (hYX + hX + hXY + hY )*0.25:
		print "Test Indecisive: Entropy values are too close"
		return 0
	elif  hYX + hX - (hXY + hY )  > 0:
		 print "Y -> X, Score: %f" % (abs(hYX + hX - (hXY + hY ))/((hYX + hX + hXY + hY )*0.25)-0.02)
		 return -1
	elif hYX + hX - (hXY + hY )  < 0:
		 print "X -> Y, Score: %f" % (abs(hYX + hX - (hXY + hY ))/((hYX + hX + hXY + hY )*0.25)-0.02)
		 return 1
if __name__ == '__main__':
	#file_name = raw_input("Name of data file:")
	if len(sys.argv)<2:
		file_name = raw_input("Name of data file:")
		main(file_name)
	elif len(sys.argv)==2:
		main(sys.argv[1])
	else:
		print "Too many input arguments!"
