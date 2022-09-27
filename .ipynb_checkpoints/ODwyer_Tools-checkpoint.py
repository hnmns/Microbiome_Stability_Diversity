# Original P functions
import numpy as np
from numpy.random import normal, uniform
from numpy.linalg import eigvals
import math as mt

## Asymmetric P matrix ##
def asymP(sig, n):

    return abs(normal(0, sig, (n, n))) ### half-normal entries

## Symmetric P matrix ##
def symP(sig, n):
    
    P = np.empty((n, n))
    for i in range(n):
        for j in range(i + 1):
            P[j][i] = P[i][j] = abs(normal(0, sig))
    
    return P

## Vectorized P functions

### Vectorized Asymmetric P matrix
def asymPvec(sig, n, k=1):
    return abs(normal(0, sig, (k, n, n))) # again, a half-normal with mean=1

### Vectorized Symmetric P matrix
def symPvec(sig, n, k=1): # k, number of matrices to generate
    P = abs(normal(0,sig,(k,n,n)))
    P = np.maximum(P,np.transpose(P,(0,2,1))) #transpose each sub-P (only transpose last two dimensions)
    ### Is max an issue? Favors less probable p-values, I guess. Not QUITE normally dist'd 
                            ### after abs and this maximum. Effectively increases sigma?
    
    return P

def symPvec2(sig,n,k=1): ### This way does not use max, but slightly slower than symPvec
    P = abs(normal(0,sig,(k,n,n)))
    inds = np.tril_indices(n, k=-1)
    P[:,inds[1],inds[0]] = P[:,inds[0],inds[1]]
    
    return P


## Setup for new approach

def extract_block_diags(A_,block_size):
    '''
    A, square matrix whose (square) block diagonals of shape (block_size,block_size) are to be extracted
    
    Previously needed for S_star_gen()

    '''
    block_diags = np.lib.stride_tricks.sliding_window_view(A_,(block_size,block_size))[::block_size,::block_size]
    num_blocks = int(A_.shape[0]/block_size)
    block_inds = list(range(num_blocks))
    
    return block_diags[block_inds,block_inds,:,:]

### According to Butler/O'Dwyer, for any CHOICE of (+) mu and rho, the formulae below will produce feasible Steady States


# Why are r_i values as big as 100??? Any way to normalize/nondimensionalize the model?

def R_star_gen(C_,P_,eps_,mu_): # P_ is actually 3-dimensional matrix (k nxn sub-P-matrices)
    '''
    Generates stack of steady state vectors R_star, where 0th dimension selects the sub-P to which 
    that particular R_star corresponds (i.e. R_star is the equilibrium for the Jacobian formed with that sub-p).
    
    No argument for number of sub-R_stars because that is inferred from 0th dimension of 'P_' argument.
    '''
    k_ = P_.shape[0]
    n_ = P_.shape[-1]
    # mu_ = np.tile(mu_,(k_,1,1))
    
    A_ = np.linalg.inv(C_.T)*(1/eps_) # A is just (k_,n_,n_)
    # print('P:',P.shape)
    # print('mu:',mu.shape)
    # print('k:',k_)
    # print('n:',n_)
    v_ = np.sum(np.transpose(P_,(0,2,1)), 2).reshape(k_,n_,1) + mu_.reshape(-1,n_,1) #take row sums of P_ transpose, reshape to add 
    # Please do not blow up because mu_ is reshaped with a -1 now.
    
    R_star_ = np.tensordot(A_,v_,axes=([1],[1]))
    R_star_ = R_star_.reshape(n_,k_).T.reshape(k_,n_,1)
    return R_star_

# ### Resource-Competition Steady State vectors (R_star and S_star)
# # Why are Rstars in the range of r_i in [1,100]??? Any way to normalize/nondimensionalize the model?

# def R_star_gen(C,P,eps,mu): # P is actually 3-dimensional matrix (k nxn sub-P-matrices)
#     '''
#     Generates stack of steady state vectors R_star, where 0th dimension selects the sub-P to which 
#     that particular R_star corresponds (i.e. R_star is the equilibrium for the Jacobian formed with that sub-p).
    
#     No argument for number of sub-R_stars because that is inferred from 0th dimension of P argument.
#     '''
    
#     k = P.shape[0]
#     n = P.shape[-1]
#     mu = np.tile(mu,(k,1,1))
    
#     #C = np.tile(C,(k,1,1)) # OLD: stack C k times so that it can be dotted with each sub-Ps
#                                 # np.dot() does not broadcast, I think. Have to manually add 3rd dimension
#     #A = np.linalg.inv(np.transpose(C,(0,2,1)))*(1/eps) # split up into A,v for readability
#     A = np.linalg.inv(C.T)*(1/eps) # A is just (k,n,n)
#     v = np.sum(np.transpose(P,(0,2,1)), 2).reshape(k,n,1) + mu #take row sums of P transpose, reshape to add 
    
#     # How to pairwise dot an array of 2D matrices with array of column vectors?
    
#     R_star = np.tensordot(A,v,axes=([1],[1]))
#     R_star = R_star.reshape(n,k).T.reshape(k,n,1)
#     #np.tensordot(I,b,axes=([1],[1])).reshape(2,3).T.reshape(3,2,1)
#     return R_star

# $\vec{S^*} = [(R^{*}_{diag})C - P]^{-1} \vec{\rho}$
def dot_across(A, x): # Wow, x can be vectors OR more matrices
    k_ = A.shape[0]
    n_ = A.shape[1]
    DI_k = np.diag_indices(k_)[0]
    prod = np.dot(A,x.reshape(k_,n_,-1)) # Do a better check for dimensions, doofus.
    return prod[DI_k,:,DI_k,:]

def S_star_gen(C_,P_,R_star_,rho_):
    k_ = P_.shape[0]
    n_ = P_.shape[1]
    
    R_diag_ = np.zeros((k_,n_,n_))
    d_inds_ = np.diag_indices(n_)
    R_diag_[:,d_inds_[0],d_inds_[1]] = R_star_.reshape(k_,n_)
        
    RC = np.dot(R_diag_, C_)

    A_ = np.linalg.inv(RC - P_) # "Subtract across" RC and P
    
    # This time, we have many different sub-A right-multiplied by constant vector rho
    if rho_.ndim == 2:
        S_star_ = np.dot(A_, rho_)
    else:
        S_star_ = dot_across(A_, rho_)
    
    return S_star_.reshape(k_,n_,1)

# def S_star_gen(C,P,R_star,rho):
#     k = P.shape[0]
#     n = P.shape[1]

#     R_diag = np.zeros((k*n,k*n))
#     np.fill_diagonal(R_diag, R_star.reshape(k*n,))
#     # now, R_diag is big array
#     R_diag = extract_block_diags(R_diag, n) # get each R_diag in (k,n,n)-shaped array
    
#     ### FIX THIS LATER FOR NON-IDENTITY C ################################
#     #  RC = np.dot(R_diag, C) # real way would be sth like this
#     RC = R_diag
    
#     A = np.linalg.inv(RC - P) # pairwise subtract RC and P
#     #S_star = np.dot(A, rho)  # in a perfect world, this is the answer
    
#     # this time, we have many different sub-A and constant rho
#     S_star = np.tensordot(A,rho.reshape(1,n,1),axes=([2],[1])) #.reshape(n,k).T.reshape(k,n,1) # we do not live in a perfect world

#     return S_star.reshape(k,n,1)

## Function for eigvals of our random systems

def eigval_gen(P,eps,mu,rho): # For now, just identity C, ADD AS ARGUMENT LATER
    '''
    Gets eigvals of system given P and parameters.
    
    Needs to be updated to take C as argument. For now, just uses identity matrix for C (perfect specialists).
    Also, mu, rho, and epsilon.
    '''
    EVs = np.array([])

    ## Parameters ##
    n = P.shape[-1]
    k = P.shape[0]

    ## Consumption Matrix ##
    c = 1.0 # Keeps cool curve from 0-1, chaotic-looking when > 1
    I = np.identity(n)
    C = c * I
    
    ## Linearized System ##
       ### Jacobian, L, eval'd at steady state
    #     [ LA | aLB]
    #aL = [---------]
    #     [ LC | LD ]
    LD = np.zeros((k, n, n))

    ## Equilibrium Abundances ##
    R_star = R_star_gen(C, P, eps, mu)
    S_star = S_star_gen(C, P, R_star, rho)

    # Community Matrix Partitions
    LA  = -c * S_star * I
    LB = P - c * R_star * I
    LC = eps * c * S_star * I

    L  = np.concatenate((np.concatenate((LA, LB),2), np.concatenate((LC, LD),2)), 1)

    return eigvals(L)

# Old rand_eigval_gen() is minimized below this cell.
# New eigval_gen() above just requires that you make your own P-matrices now.
    # That way, you can keep track of the P-matrices that led to the eigvals that you get.

#### OLD ####
def rand_eigval_gen(n, k, sig = 1, symmetric=True): # For now, just identity C, ADD AS ARGUMENT LATER
    '''
    Takes same arguments as (a)symPvec(), but performs all of the eigenvalue calcs for each of the k matrices
    
    Needs to be updated to take C as argument. For now, just uses identity matrix for C (perfect specialists).
    Also, mu, rho, and epsilon.
    '''
    
    EVs = np.array([])

    ## Parameters ##

    # n   = 20      # number of species AND number of resrcs
    # k = 750        # number of systems to generate
    eps = 0.25    # efficiency, WHY SAME FOR EACH SPECIES, scales consumption
    # sig = 25      # std deviation of production rates? Yes, also for abundances?
    c = 1       # equal consumption rate, WHY SAME FOR EACH SPECIES
    #mu = np.random.unif(0,1,n) 
    mu = (np.ones(n)*0.5).reshape(n,1) # R_star_gen takes 2D mu
    #rho = np.random.unif(0,1,n)
    rho = (np.ones(n)*0.5).reshape(n,1)
    ### SEE METHODS FOR mu AND rho


    ## Consumption Matrix ##
    I = np.identity(n)
    C = c * I
    # C = np.identity(n)

    
    # only difference between symmetric=True and otherwise is the call to symPvec2() or asymPvec()
        
    ## Choice of Production matrices ##
    if (symmetric):
        P = symPvec2(sig, n, k)
    else:
        P = asymPvec(sig, n, k)
    
    ## Linearized System ##
       ### Jacobian, L, eval'd at steady state
    #     [ LA | aLB]
    #aL = [---------]
    #     [ LC | LD ]
    LD = np.zeros((k, n, n))


    ## Equilibrium Abundances ##
    R_star = R_star_gen(C, P, eps, mu)
    S_star = S_star_gen(C, P, R_star, rho)

    # Community Matrix Partitions
    LA  = -c * S_star * I
    LB = P - c * R_star * I
    LC = eps * c * S_star * I


    L  = np.concatenate((np.concatenate((LA, LB),2), np.concatenate((LC, LD),2)), 1)


    # EVs = np.append(EVs, eigvals(L)) # old
    # print(P)
    return eigvals(L)