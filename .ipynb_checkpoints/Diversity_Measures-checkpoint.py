

# ## Jost    
# All diversity index functions below will take equilibrium vector $N$ and return raw index, not the numbers equivalent.

# $H_A$, $H_B$ are any orthogonal quantities/qualities of a diversity index H (EX: alpha and beta diversity, evenness and richness).

import numpy as np
from numpy.linalg import eigvals
import math as mt

def normalize(N):
    '''
    Makes column vector(s) length 1.
    '''
    if N.ndim == 2:
        N = N.reshape(1,N.shape[0],N.shape[1])
    norms = np.sqrt(np.sum(np.square(N),1)).reshape(N.shape[0],1,1)
    N = N/norms
    
    return N

def sum_to_one(N):
    if np.all(N==0):
        return N
    if N.ndim == 2:
        N = N.reshape(1,N.shape[0],N.shape[1])
    return N/np.sum(N,1).reshape(N.shape[0],1,1)

def simpson_conc(N):
    H = np.sum(np.square(N),1)
    # D = 1/H
    return H

def shannon_entropy(N): 
    if (np.all(N>0)):
        H = -np.sum(N*np.log(N), 1)
    else:
        H = -999 # Error value, to point out unviable equilibria in gLV.
    # D = np.exp(H).reshape(N.shape[0],1,1)
    return H

def gini_simpson_index(N):
    H = 1 - np.sum(np.square(N), axis = 1)
    # D = 1/(1-H)
    
    return H

def hcdt_entropy(N, q=2):
    p = sum_to_one(N)
    if q==1:
        return shannon_entropy(p) # Degenerates to Shannon entropy
    else:
        return (1 - np.sum(np.power(p,q), 1) / (q-1) )

def renyi_entropy(N, q=2):
    p = sum_to_one(N)
    if q==1:
        return shannon_entropy(p) 
    else:
        return -np.log(np.sum(np.power(p,q), 1)) / (q-1)
    
    
def D_1(p):
    if p.ndim == 1:
        p = p.reshape(1,p.size,1)

    return np.exp(- np.sum(p*np.log(p), axis=1))


# Universal numbers equivalent
def numbers_equiv_alpha(w,p,q):
    '''
    There are C communities and S species.
    
    Takes community weights w (vector of length C), species probabilities p of size (S,C), and order q of the diversity index.
    Returns numbers equivalent of diversity index's alpha component.
    '''
    w = w.reshape(p.shape[0],1)
    
    if (q==1): # undefined case, need formula that takes the limit as q->1
        D_alpha = np.exp(np.sum(-w * np.sum(p*np.log(p),1) ))

    else:
        num = np.sum(np.power(w,q)*np.sum(np.power(p,q),1) ) # take the species sums of p with axis=0
        den = np.sum(np.power(w,q))
        
        D_alpha = np.power(num/den, 1/(1-q))
        
    return D_alpha

def numbers_equiv_gamma(w,p,q):
    w = w.reshape(p.shape[0],1,1)
    
    weight_sum = np.sum(w*p)
    
    if (q==1):
        D_gamma = np.exp(np.sum(-weight_sum*np.log(weight_sum)))
        
    else:
        # Really, Jost only explicitly gives formula for even weights. Should work like this, right?
        D_gamma = np.power(np.sum(w*p), q)
        D_gamma = np.power(D_gamma, 1/(1-q))
        
    return D_gamma 