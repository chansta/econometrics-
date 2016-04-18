######################################################################################
"""
Name:       garchpq.py
                An object for purposes of simulating and estimating GARCH process with various conditional mean including nonlinear time series such as STAR. 
Created by:     Felix Chan
Email:          fmfchan@gmail.com
Created on:     2011.12.24
Last Updated:   2013.12.18 - Replacing print statement with print() function. 

"""
######################################################################################
import numpy as np
import scipy as sp
from scipy.stats import norm, t
import numdifftools as nd
import math
from cffi import FFI

class garchpq:

	def __init__(self, p, mean="ar", morder=[1,0], vorder=[1,1], rep=False, data=[1000], dist="norm", fullsample=True, useC=False):
		"""
			An General GARCH(r,s) object with different conditional mean. 
			Input: 
			p : mX1 vector of parameters where m =r+s+q+1 where q is the total number of parameters in the conditional mean
			mean_info: a 2X1 list. First element indicates the type of conditional mean (see extension) and the second element indicates the order. 
			vorder: a 2X1 list for r and s, respectively. 
			data: a TX1 array of observations. Leave the option as default if object is created for simulation purposes. 
			dist: Distribution for the conditional residuals. 
			fullsample: True-using full sample for estimation. Int: use a sub-sample for estimation. Sample range from 0 up to the date specified by fullsample. 
			useC: If True then the routine uses a C function to compute the conditional variance. 
		"""
		self.r = vorder[0]
		self.s = vorder[1]
		self.mr = morder[0]
		self.ms = morder[1]
		self.useC = useC 
		self.mean = mean
		self.base = max(max(vorder, morder))
		self.p = np.array(p)
		self.a = self.p[0:self.r]
		self.b = self.p[self.r:self.r+self.s]
		self.w = self.p[self.r+self.s]
		self.mp = self.p[self.r+self.s+1:] 
		self.m = len(p)
		if len(data)!=1:
			if fullsample == True:
				self.data = data
			else:
				self.data = data[0:fullsample]
			self.T = len(self.data)
			self.allT = len(data)
			self.alldata = data
		else:
			self.T = data[0]
			self.allT = self.T
			self.data = np.zeros(self.T)
		self.e = np.zeros(self.allT)
		self.h = np.ones(self.allT)*sp.stats.tvar(self.data)
		self.eta = np.zeros(self.allT)
		self.fitmean = np.zeros(self.allT)
		self.dist = dist
		self.rep = rep
		self.ll = -1
		self.iteration = 0
		self.function_call = 0
		self.warnflag = 0
		self.J = np.zeros((self.T,self.m))
		self.H = np.zeros((self.m,self.m))
		self.cov = np.zeros((self.m, self.m))
		self.didestimate = 0
		self.std = np.zeros(len(self.p))
		self.tstats = np.zeros(len(self.p))
		f = open("get_hpq.c")
		s = f.read()
		f.close()
		self.ffi = FFI()
		self.ffi.cdef("void get_hpq(double h[], double w[], double a[], double b[], double e2[], double h0[], int order[]);")	
		self.C = self.ffi.verify(s)

	
	def simulate(self, initial=[0.3,0,0.075]):
		"""
			Simulate a AR(1)-GARCH(r,s) process. Given the information provided in the garchpq object
			Input:
				initial: a r+s+1 array indicating the initial values for the observed time series, the unconditional residuals and the conditional variance. 
			Output:
				self.y, self.e, self.h
		"""
		base = max(self.r, self.s)
		trange = np.arange(base,self.T)
		self.e[base-self.r:base] = initial[0:self.r]
		self.h[base-self.s:base] = initial[self.r:self.r+self.s]
		self.data[base-1] = initial[self.r+self.s]
		if self.dist=="norm":
			self.eta = sp.stats.norm.rvs(size=self.T)
		elif self.dist=="t":
			self.eta = sp.stats.t.rvs(size=self.T)
		else:
			print("We have not yet implememted this distribution yet")
			return -1
		for t in trange:
			self.h[t] = self.w + np.dot(self.e[t-self.r:t]**2, self.a) + np.dot(self.h[t-self.s:t], self.b)
			self.e[t] = self.eta[t]*pow(self.h[t],0.5)
			self.data[t] = self.mp[0] + self.mp[1]*self.data[t-1]+self.e[t]

	def __transferf__(self, pa, x, Gtype):
		"""
			Calculate the transfer (transistion) function - either one point at a time or from a set of observation
			g,c = pa in that order
		"""
		g = pa[0]
		c = pa[1]
		if (Gtype == "logistic"):
			return np.power((1+np.exp(-g*(x-c))), -1)
		elif (Gtype == "exponential"):
			return 1 - np.exp(-g*np.power(x-c,2))
		else:
			return 0

	def __star__(self, mp, Gtype):
		"""
			Calculate the condtional mean with a STAR specification. Specification follows Chan, McAleer and Medeiros (2012). 

		"""
		ly = self.data[0:self.T-1]
		fy = self.data[1:self.T]
		x = np.c_[np.ones(self.T-1), ly].transpose()
		phi1 = np.dot(mp[0:2], x)
		phi2 = np.dot(mp[2:4],x)
		pa = mp[4:]
		G = self.__transferf__(pa,ly, Gtype=Gtype) 
		self.fitmean[0] = self.data[0]
		self.fitmean[1:]=phi1.reshape(self.T-1)+phi2.reshape(self.T-1)*G
		self.e = self.data - self.fitmean
		return 0
	
	def __arma__(self, mp):
		"""
			Calculate the conditional mean with a ARMA(r,s) specification. 
		"""
		base = self.base 
		if (self.mr != 0):
			arp = mp[0:self.mr]
		if (self.ms !=0):
			mapp = mp[self.mr:self.mr+self.ms]
		mu = mp[self.mr+self.ms]  
		if (self.mr!=0)&(self.ms!=0): 
			for t in np.arange(base, self.T):
				self.fitmean[t] = mu + np.dot(self.data[t-self.mr:t], arp) + np.dot(mapp, e[t-self.ms:t]) 
				self.e[t] = self.data[t]-self.fitmean[t] 
		elif (self.mr==0): 
			for t in np.arange(base, self.T):
				self.fitmean[t] = mu + np.dot(mapp, e[t-self.ms:t])
				self.e[t] = self.data[t]-self.fitmean[t]
		else:
			for t in np.arange(base, self.T):
				self.fitmean[t] = mu + np.dot(self.data[t-self.mr:t], arp) 
				self.e[t] = self.data[t]-self.fitmean[t]
		return 0 

	def likelihood(self,p, rep, get_opg):
			"""	
				Return the conditional log-likelihood value for AR(1)-GARCH(1,1) model given the parameter p and data y. Rep is used for reparameterization purposes.
				Input: 
					p: 5X1 list of parameters
					y: a TX1 list of observations
					rep: 1. to reparameterize; otherwise no reparameterization
					get_opt: 0: Returns likelihood as a scalar; 1: return a Tx1 vector with likelihood scores for every observation
				Output:
					ll: scalar. The log-likelihood value
				Extension: Hoping to extend this AR(p)-GARCH(r,s) soon.
			"""
			base = self.base 
			a = p[0:self.r]
			b = p[self.r:self.r+self.s]
			w = p[self.r+self.s]
			mp = p[self.r+self.s+1:]
			self.h[base-self.s:base] = sp.stats.tvar(self.data)*np.ones(self.s)
			self.e[base-self.r:base] = sp.stats.tmean(self.data)*np.ones(self.r)
			if self.mean=="ar":
				self.__arma__(mp)
			else:
				self.__star__(mp,Gtype=self.mean)
			trange = np.arange(base,self.T)
			if self.useC == False:
				for t in trange:
					if (rep==True):
						self.h[t] = np.exp(w) + np.dot(np.power(self.e[t-self.r:t],2),np.power(a,2)) + np.dot(self.h[t-self.s:t], np.power(b,2))
					else:
						self.h[t] = w + np.dot(np.power(self.e[t-self.r:t],2),a) + np.dot(self.h[t-self.s:t], b)
			else:
				self.h = self.__getH__(w,a,b)
			self.h = np.abs(self.h)
			l = -0.5*(np.log(2*math.pi)+np.log(self.h[base:self.T]) + np.power(self.e[base:self.T],2)/self.h[base:self.T])
			if get_opg==0:
				ll = np.sum(l)
			else:
				ll = l
			return ll
	
	def __getH__(self, lw, la, lb):
		w = self.ffi.new("double[]", [lw])
		a = self.ffi.new("double[]", list(la))
		b = self.ffi.new("double[]", list(lb))
		h = self.ffi.new("double[]", list(self.h))
		h0 = self.ffi.new("double[]", list(self.h[0:self.s]))
		e = self.ffi.new("double[]", list(np.power(self.e,2)))
		order = self.ffi.new("int[3]", [int(self.r), int(self.s), self.T])
		self.C.get_hpq(h,w,a,b,e,h0,order)
		return np.array([float(h[i]) for i in range(0,self.T)])


	def estimate(self, getcov=False):
		"""
			Maximize the log-likelihood function given information from initialization.
		"""
		if len(self.data)<2:
			print("You do not have enough data to estimate the parameters.")
			return -1
		else: 
			fobj = lambda b0: -self.likelihood(b0, rep=self.rep, get_opg=0)
			dfobj = nd.Gradient(fobj)
			p0 = self.p
			#self.p, self.ll, self.J, self.function_call, gradientcall, self.warnflag, allvecs = sp.optimize.fmin_bfgs(fobj, p0, fprime=dfobj, full_output=1, retall=1) 
			self.p,self.ll,self.iteration,self.function_call,self.warnflag = sp.optimize.fmin(fobj, p0, maxiter=10000, maxfun=100000, full_output=1)  
		if (self.rep==True):
			self.p[0:r+s] = self.p[0:r+s]**2
			self.p[r+s] = np.exp(p[r+s])
		if getcov==True:
			self.get_cov()
		self.didestimate = 1

	def get_cov(self):	
		"""
			Generate robust variance-covariance matrix. As by-products, once can also obtain OPG and Hessian matrices.
		"""
		dl = lambda p0: self.likelihood(p0,False,1)
		dlh = lambda p0: self.likelihood(p0,False,0)
		edl = nd.Jacobian(dl, stepMax=float(0.000001), vectorized=True)
		edlh = nd.Hessian(dlh, stepMax=float(0.000001)) 
		J = edl(self.p)
		self.H = np.linalg.inv(edlh(self.p))
		self.J = np.dot(J.transpose(),J)
		self.cov = np.dot( np.dot(self.H, self.J), self.H)
		self.std = np.power(np.abs(np.diag(self.cov)), 0.5)
		self.tstats = self.p/self.std
	
	def fit(self, showgraph=False):
		if len(self.data)!=len(self.alldata):
			temp = np.array(self.data)
			self.data = np.array(self.alldata)
			self.T = self.allT
		self.likelihood(self.p, 0, 0)
		self.eta = self.e/pow(self.h, 0.5)
		titlename = ["Conditional Mean", "Unconditional Residuals", "Conditional Variance", "Conditional Residuals"]
		fitdata = np.c_[self.data, self.fitmean, self.e, self.h, self.eta]
		if (showgraph==True):
			import matplotlib.pyplot as plt
			trange = np.arange(0,len(self.data))
			for i in np.arange(1,5):
				plt.subplot(2,2,i)
				if (i==1):
					plt.plot(trange, fitdata[:,i-1], 'b-', trange, fitdata[:,i], 'r-')
				else:
					plt.plot(trange, fitdata[:,i], 'b-')
				plt.xlabel("Time")
				plt.ylabel("Returns")
				plt.title("Plot of "+titlename[i-1]) 
			plt.show()
		if len(self.data)!=len(self.alldata):
			self.data = temp
			self.T = len(self.data)	
	def summary(self, savefile=False):
		if self.cov.all == 0:
			print("You are yet to estimate the model. \n")
		else:
			result = np.c_[self.p, self.std, self.tstats]
			s = [["{0:.3f}".format(i) for i in l] for l in result]
			s.insert(0, ["Estimates","Standard Errors", "t statistics"])
			for i in s:
				print("{0:18} {1:18} {2:18}".format(*tuple(i)))
		if savefile != False:
			ss = ","+",".join(s[0])+"\n"+"\n".join([",".join([j for j in l]) for l in s[1:]])
			f = open(savefile, "w")
			f.write(ss)
			f.close()
