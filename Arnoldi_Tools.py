from numpy.linalg import eigvals
import numpy as np

def dom_eigvals(A): # for A shape (k,2*n,2*n) ; returns entire eigval (complex, with sign)
    if A.ndim == 2:
        A = A.reshape(1,A.shape[0],A.shape[1])    
    k = A.shape[0]
    
    evs = eigvals(A)
    
    dom_ev_ind = (np.argmax(evs.real,-1)) 
    dom_ev = evs[np.arange(0,k), dom_ev_ind]
    
    return dom_ev

def dom_eigval_single(A): # Just for spectral_norm()
    evs = eigvals(A)
    
    dom_ev_ind = (np.argmax(evs.real,-1)) 
    dom_ev = evs[dom_ev_ind]
    
    return dom_ev

def asym_res(A):
    if A.ndim == 2:
        A = A.reshape(1,A.shape[0],A.shape[1])
    k = A.shape[0]
    dom_ev = dom_eigvals(A)
    
    R_inf = -dom_ev.real
    
    return R_inf

from scipy.optimize import minimize
from scipy.optimize import brent
from scipy.optimize import minimize_scalar # which Brent?
def spectral_norm(A): # only takes 1 matrix
    dom_ev = dom_eigval_single(np.dot((A.T).conjugate(), A))

    return np.emath.sqrt((dom_ev.real)) # np.emath.sqrt returns complex if takes negative # Take the real part?

# Find omega that maximizes ||inverse(i*om-A)||_{spectral}
# Objective function:
def determ_invar_OBJ(om, A):
    B = np.linalg.inv(np.eye(A.shape[-1])*complex(0,om) - A)  # call B the matrix whose spectral norm we aim to maximize    
    return -spectral_norm(B) # negative because we are maximizing

# Deterministic Invariability
def determ_invar(om, A): # assumes you have found argmin omega    
    B = np.linalg.inv(np.eye(A.shape[-1])*complex(0,om) - A)
    
    return 1 / spectral_norm(B)

def stoch_invar(A):
    # NOT vectorized bc Kronecker product (at the moment)
    I = np.identity(A.shape[-1])
    A_hat = np.kron(A,I) + np.kron(I,A)
    
    temp = -np.linalg.inv(A_hat)
    spec_norm = spectral_norm(temp)
    
    SI = 0.5 / spec_norm
    
    return SI



def init_res(A):
    
    if A.ndim == 2:  # to account for when passing a single community matrix A, shape (2*n,2*n)
        A = A.reshape(1,A.shape[0],A.shape[1])
    k = A.shape[0]
    
    temp = A + np.transpose(A, (0,2,1)) # symmetric matrices have real eigvals
    
    dom_evs = dom_eigvals(temp).real
    
    R0 = -0.5* dom_evs
    
    return R0

