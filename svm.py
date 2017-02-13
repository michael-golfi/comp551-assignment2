import numpy as np
#from cvxopt.solvers import qp
import pandas as pd
#from cvxopt import matrix

class svm():
	def __init__(self):
		self.bool=1

	def fit(self,X,Y):
        	y1=pd.DataFrame(np.copy(Y))
        	y1.columns=["Y"]
        	y2=y1.copy()
        	y3=y1.copy()
        	y4=y1.copy()
        	y5=y1.copy()
			y6=y1.copy()
			y7=y1.copy()
			y8=y1.copy()
			##### setting 1 to values for the particular class where it belongs
			y1[y1["Y"]!=1]=-1
        	y2[y2["Y"]!=2]=-1
        	y3[y3["Y"]!=3]=-1
        	y4[y4["Y"]!=4]=-1
			y5[y5["Y"]!=5]=-1
			y6[y6["Y"]!=6]=-1
			y7[y7["Y"]!=7]=-1
			y8[y8["Y"]!=8]=-1
       		y2[y2["Y"]==2]=1
        	y3[y3["Y"]==3]=1
        	y4[y4["Y"]==4]=1
			y5[y4["Y"]==5]=1
			y1 = (y1.as_matrix())
        	y2 = (y2.as_matrix())
        	y3 = (y3.as_matrix())
        	y4 = (y4.as_matrix())
			y5 = (y5.as_matrix())
        	y6 = (y6.as_matrix())
        	y7 = (y7.as_matrix())
        	y8 = (y8.as_matrix())
	        self.w1=self.findw(X,y1)
        	self.w2=self.findw(X,y2)
        	self.w3=self.findw(X,y3)
        	self.w4=self.findw(X,y4) 
			self.w5=self.findw(X,y5)
        	self.w6=self.findw(X,y6)
        	self.w7=self.findw(X,y7)
        	self.w8=self.findw(X,y8) 
		
	
	def findw(self,X,Y):
		#number of examples
		n = X.shape[0]
		q_coef = np.ones((n,n))
		q_coef=np.multiply(q_coef,Y).transpose()
		q_coef=np.multiply(q_coef,Y)
		for i in range(0,n):	
		
			for j in range(0,n):
				factor = X[i].dot(X[j])
				q_coef[i][j] = q_coef[i][j]*factor
		
		P = matrix(q_coef,tc='d')
		q = matrix(-np.ones(n),tc='d')
		G = matrix(-np.identity(n),tc='d')
		h = matrix(np.zeros(n),tc='d')
		A = matrix(Y.transpose(),tc='d')
		b = matrix(0,tc='d')
		sol = qp(P,q,G,h,A,b)		
		alpha=sol['x']
		w=alpha[0]*Y[0]*X[0]
		for i in range(1,n):
			w=w+alpha[i]*Y[i]*X[i]
		return w


	def predict(self,x):
        	w1=self.w1
        	w2=self.w2
        	w3=self.w3
        	w4=self.w4
			w5=self.w5
			w6= self.w6
			w7= self.w7
			w8=self.w8
        	pred = list()
		#print(w1.shape)
		#print(x.shape)
		
        	p1=x.dot(np.array([w1]).transpose())
        	p2=x.dot(np.array([w2]).transpose())
        	p3=x.dot(np.array([w3]).transpose())
        	p4=x.dot(np.array([w4]).transpose())
			p5=x.dot(np.array([w5]).transpose())
        	p6=x.dot(np.array([w6]).transpose())
        	p7=x.dot(np.array([w7]).transpose())
        	p8=x.dot(np.array([w8]).transpose())
        	pred=1+np.argmax(np.concatenate((p1,p2,p3,p4,p5,p6,p7,p8),axis=1),axis=1)

		return np.array(pred)