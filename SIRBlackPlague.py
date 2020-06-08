#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:28:17 2020

@author: Javi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:48:18 2020

@author: Javi
"""


import numpy as np
import scipy
import scipy.stats
import scipy.integrate
from scipy.integrate import odeint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numba import jit
from matplotlib import cm
import pytwalk
import emcee
import pystan
import time
import corner
import arviz as az
import timeit


''' Second black plague outbreak in Eyam, UK 
    June 19, 1666 to Nov 1, 1666 (Massel et al 2004)
    114 days
    The village had been quarantined so the population
    is fixed to N=261. 
    
    States: 
        S(t): susceptible
        I(t): infected
        R(t): removed
        N=S(t)+I(t)+R(t)
    obs: once you get the plague you never recover, you die 
    
    '''
    
N=261    

def SIR( X, t,  alpha, beta, I0  ):
    dSdt= -beta* X[0] * X[1]
    dIdt= beta*X[0] * X[1]- alpha*X[1]
    dRdt= alpha*X[1]
    return np.array( [   dSdt, dIdt, dRdt   ]   )
    

data = pd.read_csv("Eyam_time_SIR.csv") 
data.index
y=np.append(0, pd.Series.to_numpy(data.iloc[:, 3])) #cumulative deaths 
x=pd.Series.to_numpy(data.iloc[ 111:114 , 2 ])
n=len(y)        
ts=np.arange(0,len(y),1 )


sns.set()
plt.scatter( ts,y, s=5)
plt.title('Cumulative deaths Elyam, UK 1666')

def logprior( Theta):
    alpha=Theta[0]
    beta=Theta[1]
    I0=Theta[2]
    if( alpha>0 and beta>0 and I0>=1 and I0<=N  ):
         a= -alpha-beta+I0*np.log( 5/N )+(N-I0)*np.log( 1- 5/N   )
         b=scipy.special.loggamma(N+1)- scipy.special.loggamma(I0+1)- scipy.special.loggamma(N-I0+1)
         return a+b
    else:
         return -np.inf
    
    
def logL(Theta):
    alpha=Theta[0]
    beta=Theta[1]
    I0=Theta[2]
    n=len(y)
    X0=np.array( [N-I0, I0, 0] )
    Xt = odeint( SIR, X0, ts, args=(alpha,beta,I0,))  
    It= Xt[:,1]
    It[It<0]=0
    Rt=Xt[:,2]
    Rt[Rt>261]=261
    ll=0
    #print(Theta)
    ll=ll+np.log( It[n-2]/N +1e-300 )+(N-1)*np.log( 1- It[n-2]/N    )+N*np.log( 1- It[n-1]/N  )
    ll=ll+ np.sum(y[1:n]* np.log( Rt[1:] /N ))+np.sum( (N-y[1:])  *np.log(   1- Rt[1:]/N +1e-300 ))
    return ll

def lpost(Theta):
    alpha=Theta[0]
    beta=Theta[1]
    I0=Theta[2]
    if( alpha>0 and beta>0 and I0>=1 and I0<=N  ):
         return -logprior(Theta)-logL(Theta) 
    else:
         return np.inf
    

def fpoints(f, args, theta,  pars , lb, upb, m=100):
    '''f is the objetcive function to be plotted
       args are the arguments of f as a list
       theta is the vector of true parameters
       pars are the parameters to be plotted (passed as an array)
       lb: array of lower bounds
       upb:array of upper bounds
       '''
       
    xx = np.linspace(lb[0], upb[0], m)
    yy = np.linspace(lb[1], upb[1], m)
    xx, yy = np.meshgrid(xx, yy)
    Z = np.zeros(shape=[m, m])
    mask = np.zeros( theta.shape, dtype=bool )
    mask[pars] = True
    temp=np.zeros(theta.shape)
    for i in range( m ):
        for j in range( m ):
            temp[mask]=np.array( [xx[i][j], yy[i][j]] )
            temp[~mask]= theta[~mask]
            #Z[i][j] = f( temp , args)
            Z[i][j] = f( temp )
    return xx,yy,Z 

def pContour(xx,yy,Z,c=100):
    plt.contour( xx, yy, Z, c, colors='b' )
    plt.xlabel( r'$\theta_1$' )
    plt.ylabel( r'$\theta_2$' )
    plt.title( r'Objective Function Contours' )
    plt.show()

def pSurface(xx,yy, Z):
    fig = plt.figure()
    ax = plt.axes( projection='3d' )
    ax.plot_surface( xx, yy, Z, cmap='Spectral')
    # ax.contour3D(xx, yy, Z, 30, cmap='binary')
    ax.set_xlabel( r'$\theta_1$' )
    ax.set_ylabel( r'$\theta_2$' )
    ax.set_zlabel( r'$Posterior Surface' )
    ax.view_init( 60, 35 )
    fig


xx,yy,Z= fpoints(f= lpost,args=1, theta= np.array([ 0.09 , 0.0007 , 5 ]) ,
                  pars= np.array([ 0,1]),
                  lb=np.array([ 0.080, 0.00043 ])  , upb= np.array([ .120 , 0.00075  ])  , m=100 )

pContour( xx, yy, Z , c=100) 



# are there several minimums?



from scipy import optimize

a=scipy.stats.uniform.rvs( 0.085, 0.12-0.085, 1  )
b=scipy.stats.uniform.rvs( 0.0004, 0.00075-0.0004, 1  )

xopt=optimize.minimize( lpost ,  np.array([a[0],b[0],5 ])  ,method='bfgs', tol=1e-5)
xopt['x']

#two minimum points at least !!!




#-----------------------------emcee

''' emcee uses -U=log posterior

#Observations:
Make a posterior function for the emcee as you will need -np.inf 
when out of bounds. emcee doesn't use a support auxiliary function. 
You should check support within the posterior!!! '''

def post_emcee(Theta):
    alpha=Theta[0]
    beta=Theta[1]
    I0=Theta[2]
    if( alpha>0 and beta>0 and I0>=1 and I0<=10 and alpha<1 and beta<1  ):
         return logprior(Theta)+logL(Theta) 
    else:
         return -np.inf
    

def p0_emcee(nwalkers=1):
    p0=np.zeros([nwalkers,3])
    a=scipy.stats.uniform.rvs(0, 1, nwalkers)
    b= scipy.stats.uniform.rvs(0, 1 , nwalkers)
    c=scipy.stats.uniform.rvs(1,9, nwalkers)
    p0[:,0]=a
    p0[:,1]=b
    p0[:,2]=c
    return p0


nwalkers = 10
T=100000
bi=int(0.10*T)
mc_x0=p0_emcee(nwalkers)

ndim=mc_x0.shape[1]

sampler =  emcee.EnsembleSampler(nwalkers, ndim, post_emcee )

start=time.time()
sampler.run_mcmc(mc_x0, T, progress=True)
end=time.time()

total_time=end-start
print(total_time/60)


samples = sampler.get_chain(discard=bi, flat=True)
print('Shape of samples: ',samples.shape)

emcee_chains = sampler.get_chain(discard=bi)



# Set theme


#run all this section at once 
plt.plot(emcee_chains[:,0,0 ])
plt.plot(emcee_chains[:,1,0 ])
plt.plot(emcee_chains[:,2,0 ])
plt.plot(emcee_chains[:,3,0 ])
plt.plot(emcee_chains[:,4,0 ])
plt.plot(emcee_chains[:,5,0 ])
plt.plot(emcee_chains[:,6,0 ])
plt.plot(emcee_chains[:,7,0 ])
plt.plot(emcee_chains[:,8,0 ])
plt.plot(emcee_chains[:,9,0 ])


#run all this section at once 
plt.plot(emcee_chains[:,0,1 ])
plt.plot(emcee_chains[:,1,1 ])
plt.plot(emcee_chains[:,2,1 ])
plt.plot(emcee_chains[:,3,1 ])
plt.plot(emcee_chains[:,4,1 ])
plt.plot(emcee_chains[:,5,1 ])
plt.plot(emcee_chains[:,6,1 ])
plt.plot(emcee_chains[:,7,1 ])
plt.plot(emcee_chains[:,8,1 ])
plt.plot(emcee_chains[:,9,1 ])


#run all this section at once 
plt.plot(emcee_chains[:,0,2 ])
plt.plot(emcee_chains[:,1,2 ])
plt.plot(emcee_chains[:,2,2 ])
plt.plot(emcee_chains[:,3,2 ])
plt.plot(emcee_chains[:,4,2 ])
plt.plot(emcee_chains[:,5,2 ])
plt.plot(emcee_chains[:,6,2 ])
plt.plot(emcee_chains[:,7,2 ])
plt.plot(emcee_chains[:,8,2 ])
plt.plot(emcee_chains[:,9,2 ])



sns.distplot( np.log(samples[:,0]) )
sns.distplot( np.log(samples[:,1]) )
plt.title(r'$\alpha, \beta $'+' log Posterior densities ')
plt.savefig('alphabetaBlackPlague.pdf', dpi=500)





def predobs( alpha, beta, I0s, tup=136  ,sample_size=150 ):
    #predictive distribution function
    L=len(alpha)
    obs=np.zeros([ sample_size  , tup]) 
    
    SSts=np.zeros([ sample_size  , tup]) 
    IIts=np.zeros([ sample_size  , tup]) 
    RRts=np.zeros([ sample_size  , tup]) 
    
    ts=np.arange(0,tup, 1)
    
    #sample of size sample_size of betas and I0s. 
    #which observations to take into account for the predictive
    indexes=np.random.choice(a = np.arange( 0, L )   , size = sample_size, replace = False)
    
    salphas=alpha[indexes]
    sI0s= I0s[indexes]
    sbetas= beta[indexes]
    for i in range(sample_size ):
        X0=np.array( [N-I0s[i], I0s[i], 0] )
        Xt=scipy.integrate.odeint( SIR, X0, ts, args=(salphas[i],sbetas[i],sI0s[i],) ) 
        It=Xt[:,1]
        It[It<0]=0
        Rt=Xt[:,2]
        Rt[Rt>261]=261
        
        SSts[i]=Xt[ :,0]
        IIts[i]=Xt[ :,1]
        RRts[i]=Xt[ :,2]
        obs[i]=scipy.stats.binom.rvs( n=N, p= Rt/N )
        
    return SSts, IIts, RRts,  obs
        
        
  
#### Predictive distribution



SSts, IIts, RRts,  obspred=predobs( samples[:,0] , samples[:,1] , samples[:,2]   , tup=114, sample_size=300)

#subsample 
indexes=np.random.choice(a = np.arange( 0,  obspred.shape[0] ), size =100, replace = False)
obs_samples=obspred[indexes]

SSts_samples=SSts[indexes]
IIts_samples=IIts[indexes]


for i in range( len(obs_samples )  ):
    plt.plot(obs_samples[i], color='red' , alpha=0.07 )


plt.scatter(ts,y, s=10, color='blue', zorder=1)
plt.xlabel( r'$t$' )
plt.ylabel( r'R(t)' )
plt.title( 'Posterior predictive curves' )
plt.savefig( 'RtBlackPlague.pdf', dpi=500  )
plt.show()


#S(t)
for i in range( len(obs_samples )  ):
    plt.plot(SSts_samples[i], color='red' , alpha=0.07 )


plt.xlabel( r'$t$' )
plt.ylabel( r'S(t)' )
plt.title( 'Posterior predictive curves S(t)' )
plt.savefig( 'STBlackPlague.pdf', dpi=500  )
plt.show()

#I(t)

for i in range( len(obs_samples )  ):
    plt.plot(IIts_samples[i], color='red' , alpha=0.07 )


plt.xlabel( r'$t$' )
plt.ylabel( r'I(t)' )
plt.title( 'Posterior predictive curves I(t)' )
plt.savefig( 'ITBlackPlague.pdf', dpi=500  )
plt.show()





