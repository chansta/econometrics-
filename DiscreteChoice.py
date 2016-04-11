import numpy as np 
import scipy as sp
from scipy.stats import norm, logistic
import math

class DiscreteChoice:
	"""
		A class for modelling and estimating discrete choice model
		Input:
			data= K+1 X N array. Data arranaged row wise ie each row contains data for each variable. 
	"""

	def __init__(self, data, dtype="logistic", intercept=True):
		self.data = np.array(data)
		self.Var,self.N = self.data.shape
		self.Y = self.data[0] 
		self.Jtype = list(set(self.Y))
		self.J = len(self.Jtype)
		if (self.J) > 2:
			self.binary = False
		else:
			self.binary = True
		self.intercept = intercept
		if (self.binary==True)&(intercept == True):
			self.X = np.r_[np.ones((1,self.N)), self.data[1:]]
		else:
			self.Var = self.Var-1
			self.X = self.data[1:]
		self.dtype = dtype 
		self.b = np.zeros(self.Var)
		self.H = np.zeros((self.Var, self.Var))
		self.tstat = np.zeros(self.Var)
		self.std = np.zeros(self.Var)
		self.Yfull = self.Y
		self.Xfull = self.X
		self.lagy = np.zeros(len(self.Y)-1)
		self.warnflag = -1

	def __loglike__(self, b0):
		"""
			Create the log-likelihood function for the discrete choice model.
			Input:
				b0: a k+1 X 1 vector of parameters in case of binary choice. 
				    a (J-1) X K array of parameters in case of discrete choice. 
		"""
		if (self.binary == True):
			b = np.array(b0).reshape((self.Var, 1))
			up = np.dot(b.transpose(), self.X)
			if (self.dtype == "probit"):
				p = sp.stats.norm.cdf(up)
			elif (self.dtype == "logistic"):
				p = np.exp(up)/(1+np.exp(up))
			p = p[0] 
			l = sum(self.Y[(p>0)&(p<1)]*np.log(p[(p>0)&(p<1)]) + (1-self.Y[(p>0)&(p<1)])*np.log(1-p[(p>0)&(p<1)]))
			return float(l)
		else: 
			#b = np.array([b0[i:i+self.Var] for i in range(0,len(b0),self.Var)])
			b = np.array(b0).reshape(self.J-1, self.Var)
			wx = np.array([np.exp(np.dot(j,self.X)) for j in b]) #(J-1) X N vector of explanatory variables
			wx=np.array([ j/(1+sum(j)) for j in wx.transpose()])
			p = np.array([1-sum(k) for k in wx])
			#print "The length of P is %i\n" %len(p)
			wx = np.c_[p,wx] 
			llist = [wx[i][self.Jtype.index(self.Y[i])] for i in range(0,self.N)] 
			l = sum(np.log(llist))
			return l



	def __dloglike__(self, b0, get_cov=False):
		"""
			Calculate the first derivatives of the log-likelihood function.
			Input: 
				b: a k+1 X vector of parameters
		"""
		b = np.array(b0).reshape((self.Var,1))
		xb = np.dot(b.transpose(), self.X) 
		if (self.dtype == "probit"):
			pdf = sp.stats.norm.pdf(xb)
			cdf = sp.stats.norm.cdf(xb)
			pdf = pdf[0] 
			cdf = cdf[0] 
			pdf = pdf[(cdf>0)&(cdf<1)]
			Y = self.Y[(cdf>0)&(cdf<1)]
			scdf = cdf[(cdf>0)&(cdf<1)]
			X = self.X[:, (cdf>0)&(cdf<1)]
			lam = -pdf/(1-scdf)*(1-Y) + pdf/scdf*Y
		elif (self.dtype == "logistic"):
			xbb = sp.stats.logistic.cdf(xb)
			xbb = xbb[0]
			lam = (self.Y - xbb)
			X = self.X
		N = len(lam)
		dL = np.array([lam[i]*X.transpose()[i] for i in range(0,N)]).transpose()
		if dL.shape[0] == 0:
			sdL="Fail"
		else:
			sdL = np.array([sum(dL[i]) for i in range(0,self.Var)])
		if (get_cov == False):
			return sdL
		else:
			return dL

	def __dloglikeDC__(self, b0, get_cov=False):
		b = np.array(b0).reshape(self.J-1, self.Var)
		#b = np.array([b0[i:i+self.Var] for i in range(0,len(b0),self.Var)])
		wx = np.array([np.exp(np.dot(j,self.X)) for j in b]) #(J-1) X N vector of explanatory variables
		wx=np.array([ j/(1+sum(j)) for j in wx.transpose()])
		p = np.array([1-sum(k) for k in wx])
		wx = np.c_[p,wx]
		dij = np.array([[int(i==self.Jtype.index(s)) for i in range(0,self.J)] for s in self.Y])
		dij = (dij - wx).transpose()
		dij = dij[1:].transpose()
		#der = np.kron(dij[1:self.J], self.X)
		der = np.array([np.kron(dij[j],self.X.transpose()[j]) for j in range(0,self.N)]).transpose() 
		sder = np.array([sum(s) for s in der])
		if get_cov==False:
			return sder
		else:
			return der

	def __HessianProbit__(self, b0):
		"""
			Compute the Hessian of the Loglihood function for Probit.
		"""
		q = 2*self.Y-1
		b = b0.reshape(self.Var)
		phi = [q[i]*sp.stats.norm.pdf(q[i]*np.dot(self.X.transpose()[i],b)) for i in range(0,len(self.Y))]
		phi = np.array(phi)
		Phi = [q[i]*sp.stats.norm.cdf(q[i]*np.dot(self.X.transpose()[i],b)) for i in range(0,len(self.Y))]
		Phi = np.array(Phi)
		lam = q*phi/Phi
		lam = lam.reshape(len(self.Y))
		#subh = [np.outer(self.X.transpose()[i], self.X.transpose()[i]) for i in range(0,self.N)]
		#coff = [-lam[i]*(lam[i]+np.dot(self.X.transpose()[i],b)) for i in range(0,self.N)]
		#h = [coff[i]*subh[i] for i in range(0,self.N)]

		h = [-lam[i]*(lam[i]+np.dot(self.X.transpose()[i], b))*np.outer(self.X.transpose()[i], self.X.transpose()[i]) for i in range(0,len(self.Y))]
		H = np.zeros((self.Var, self.Var))
		for i in range(0,len(self.Y)):
			H = H+h[i]
		return H

	def __OPG__(self, b0):
		dL = self.__dloglike__(self, b0, get_cov=True)
		return np.dot(dL, dL.transpose())/self.N
			
	def estimate(self, b0, fprime=False):
		"""
			Estiamte the parameters of the Discrete Choice model as defined by the object. 
			Input:
				b0: a self.VarX1 vector of initial parameters. 
			Output; 
				self.b: parameter estimates
				self.LL: the log-likelihood value at the optimum.
				self.H: the Hassian matrix
				self.std: vector of standard errors of the parameter estimates
				self.tstat: vector of t-statistics of the parameter estimates
		"""
		f = lambda b: (-1)*self.__loglike__(b)
		if (self.binary == True):
			g = lambda b: self.__dloglike__(b)
		else:
			g = lambda b: (-1)*self.__dloglikeDC__(b)
		if fprime==True:
			output = sp.optimize.fmin_bfgs(f, b0, fprime=g, maxiter=10000, full_output=True, retall=True)
		else:
			output = sp.optimize.fmin_bfgs(f, b0, maxiter=10000, full_output=True, retall=True)
		self.b = output[0]
		self.LL = -output[1]
		self.H = output[3]
		self.std = np.power(np.array([abs(self.H[i,i]) for i in range(0,len(self.b))]), 0.5)
		self.tstat = self.b/self.std
		self.warnflag = output[6]	
	
	def probability_fit(self, b0):
		"""
			Fitting probability given the parameter vector, b. 
		"""
		b = np.array(b0).reshape((self.Var,1))
		xb = np.dot(b.transpose(), self.X)
		if self.dtype == "logistic":
			cdf = exp(xb)/(1+exp(xb)) 
		elif self.dtype == "probit":
			cdf = sp.stats.norm.cdf(xb).reshape(self.N) #need double check on the norm function. 
		self.fit = np.array([(1-self.Y[i])*cdf[i] + self.Y[i]*(1-cdf[i]) for i in range(0,self.N)])
		return self.fit
	
	def estimate_serial(self, b1):
		"""
			Attempt to estimate the parameter in a probit model with serial correlation in the error term.
		"""
		self.estimate(b1)
		print "The initial estimate of b is {0}".format(self.b[1])
		self.lagy = self.probability_fit(self.b)
		self.X = np.r_[self.Xfull.transpose()[1:self.N].transpose(), self.lagy[0:self.N-1].reshape(1,self.N-1)]
		self.Y = self.Yfull[1:self.N] 
		bp = np.r_[b1,0.5]
		self.Var = self.Var+1
		self.estimate(bp)
		self.Var = self.Var-1
		self.X = self.Xfull
		self.Y = self.Yfull

	def LR_test_Stability(self, b0, pi, serial=False):
		"""
			Conduct Log-ratio test for structural stability. 
			Input: 
				b0: initial values for purposes of estimation
				pi: location of break point in percentage term pi \in (0,1). The actual break point is calculated as math.floor(pi*T) where T is the total number of time series observations. 

		"""
		if (serial == False):
			estimate = lambda b: self.estimate(b)
		elif (serial == True):
			estimate = lambda b: self.estimate_serial(b) 
		count = 0 
		bp = math.floor(pi*self.N)
		estimate(b0)
		count = self.warnflag
		LL = self.LL 
		self.Y = self.Yfull[0:bp]
		self.X = self.Xfull.transpose()[0:bp].transpose()
		self.N = len(self.Y)
		estimate(b0)
		count = self.warnflag + count
		L0 = self.LL
		self.Y = self.Yfull[bp:]
		self.X = self.Xfull.transpose()[bp:].transpose()
		self.N = len(self.Y)
		estimate(b0)
		count = self.warnflag + count
		L1 = self.LL
		self.Y = self.Yfull
		self.X = self.Xfull
		self.N = len(self.Y)
		if (count > 0):
			return -10000
		else:
			return 2*(L0+L1-LL)

	def LM_test_Stability(self, b0, pi, serial=False): 
		"""
			Compute the LM test of structural stability.  
		"""
		if (serial == False):
			estimate = lambda b: self.estimate(b)
		elif (serial == True):
			estimate = lambda b: self.estimate_serial(b)
		count = 0 
		bp = math.floor(pi*self.N)
		estimate(b0)
		count = self.warnflag
		Nbackup = self.N
		if (serial == True):
			self.X = np.r_[self.Xfull.transpose()[1:self.N].transpose(), self.lagy[0:self.N-1].reshape(1,self.N-1)]
			self.Y = self.Yfull[1:self.N]
			self.Var = self.Var+1
		H = self.__HessianProbit__(self.b)
		invH = -np.linalg.pinv(H)
		self.Y = self.Y[0:bp]
		self.X = self.X.transpose()[0:bp].transpose()
		self.N = bp 
		dl = self.__dloglike__(self.b)
		self.Y = self.Yfull
		self.X = self.Xfull
		self.N = Nbackup
		if (dl == "Fail")|(count>0):
			return -10000 
		else:
			LM = np.dot(np.dot(dl.reshape((1,self.Var)),invH), dl)/(pi*(1-pi))
			return LM[0]
		if (serial == True):
			self.Var = self.Var-1
	
	def Wald_test_Stability(self, b0, pi, serial=False):
		"""
			Compute the Wald Statistics for structural stability.
		"""
		if (serial == False):
			estimate = lambda b: self.estimate(b)
		elif (serial == True):
			estimate = lambda b: self.estimate_serial(b)
		count = 0 
		bp = math.floor(pi*self.N)
		self.X = self.Xfull.transpose()[0:bp].transpose()
		self.Y = self.Yfull[0:bp]
		self.N = len(self.Y) 
		estimate(b0)
		b1 = self.b
		v1 = self.H
		self.X = self.Xfull.transpose()[bp:].transpose()
		self.Y = self.Yfull[bp:]
		self.N = len(self.Y)
		estimate(b0)
		b2 = self.b
		v2 = self.H
		self.Y = self.Yfull
		self.X = self.Xfull
		self.N = len(self.Y)
		diffb = (b1-b2)
		v = v1+v2
		return np.dot(np.dot(diffb, v),diffb)

	def summary(self):
		if self.H.all==0:
			print "You are yet to estimate the model.\n"
		else:
			alls = np.c_[self.b, self.std, self.tstat]
			s = [[str(j) for j in ss] for ss in alls]
			s.insert(0,['Estimate', 'Standard Error', 't-statistics'])
			for i in range(0, len(self.b)+1):
				print "{0:18} {1:18} {2:18}".format(*tuple(s[i])) 
			
