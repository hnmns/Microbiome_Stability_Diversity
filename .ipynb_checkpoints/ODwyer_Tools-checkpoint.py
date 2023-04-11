# Original P functions
import numpy as np
from numpy.random import normal, uniform
from numpy.linalg import eigvals
import math as mt

from Arnoldi_Tools import *
from Diversity_Measures import *

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


def community_matr_gen(C_,P_,eps_,mu_,rho_):
    '''
    Returns L, the community matrix (Jacobian) of an O'Dwyer system described by given:
        Consumption matrix C,
        Production matrix P, 
        Exogenous mu,
        rho
    '''
    n = P_.shape[-1]
    k = P_.shape[0]
    I = np.identity(n)
        
    LD = np.zeros((k, n, n))

    ## Equilibrium Abundances ##
    R_star = R_star_gen(C_, P_, eps_, mu_)
    S_star = S_star_gen(C_, P_, R_star, rho_)
    
    # Community Matrix Partitions
    CS = np.dot(C_, S_star).T.reshape(k,n,1) # can use regular .T because first dimension is 1
    CR = np.dot(C_, R_star).T.reshape(k,n,1)
    
    LA  = -CS * I
    LB = P_ - CR * I
    # LC = eps_ * CS * I # WRONG
    d_idx = np.diag_indices(n)[0]
    S_diag = np.zeros((k,n,n))
    S_diag[:,d_idx,d_idx] = S_star.reshape(k,n)
    LC = eps_ * np.dot(S_diag, C_.T)

    L  = np.concatenate((np.concatenate((LA, LB),2), np.concatenate((LC, LD),2)), 1)
    
    return L


def community_matr_gen_noP(n_, k_, C_, eps_, R_star_, S_star_):

    I = np.identity(n_)
        
    LD = np.zeros((k_, n_, n_)) # Bottom right
    
    # Community Matrix Partitions
    d_idx = np.diag_indices(n_)[0]
    R_diag = np.zeros((k_,n_,n_))
    R_diag[:,d_idx,d_idx] = R_star_.reshape(k_,n_)
    
    S_diag = np.zeros((k_,n_,n_))
    S_diag[:,d_idx,d_idx] = S_star_.reshape(k_,n_)
    
    A_temp = - np.dot(C_, S_star_)
    A_temp = np.transpose(A_temp, (1,0,2))
    
    LA = np.zeros((k_,n_,n_))
    LA[:,d_idx,d_idx] = A_temp.reshape(k_,n_) # Top left
    LB = - np.dot(R_diag, C_) # Top right

    LC = eps_ * np.dot(S_diag, C_.T) # Bottom left

    L  = np.concatenate((np.concatenate((LA, LB),2), np.concatenate((LC, LD),2)), 1)
    
    return L


# Add lo(w)ess to stability-diversity plots
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

class ODsystem:
    '''
    O'Dwyer systems + their community matrices and equilibria + Arnoldi Measures.
    VERY slow when n>~6, quadratic time from needing to for-looping I_S and I_D.
    -------------------------------------------------------
    Specify:
    k, number of systems
    n, number of species and number of resources
    mu, vector of mortality rates
    rho, vector of resource influx rates
    eps, resource consumers' conversion efficiency of biomass
    sig_P, standard deviation of entries in production matrix P
    C, consumption matrix
    '''
    def __init__(self, k_=1000,n_=5, eps_=0.25, alpha_EQ_ = None, C_ = None, seed_=667):
        self.seed = seed_
        np.random.seed(self.seed)
        self.k = k_
        self.n = n_
        self.eps = eps_
        self.alpha_EQ = alpha_EQ_
        
        if (alpha_EQ_ == None):
            self.R_star = np.random.uniform(0,1, (self.k, self.n, 1))
            self.S_star = np.random.uniform(0,1, (self.k, self.n, 1))
        else:
            self.R_star = np.random.dirichlet(self.alpha_EQ*np.ones(self.n), size=self.k).reshape(self.k,self.n,1)
            self.S_star = np.random.dirichlet(self.alpha_EQ*np.ones(self.n), size=self.k).reshape(self.k,self.n,1)
        if C_ is None:
            self.C = np.identity(self.n)
        else:
            self.C = C_
        
        
        self.s_star = sum_to_one(self.S_star)
        self.r_star = sum_to_one(self.R_star)
        
        blankones = np.ones((self.k,self.n,self.n))
        self.mu = dot_across(np.dot(self.eps * np.eye(self.n), np.transpose(self.C)) * blankones, self.R_star)
        self.rho = dot_across(dot_across(np.eye(self.n)*self.R_star, self.C*blankones), self.S_star)
        
        #(n_, k_, C_, eps_, R_star_, S_star_)
        self.J = community_matr_gen_noP(n_=self.n, k_=self.k, C_=self.C, eps_=self.eps, R_star_=self.R_star, S_star_=self.S_star)
        # self.R_star = R_star_gen(self.C,self.P,self.eps,self.mu)
        # self.S_star = S_star_gen(self.C,self.P,self.R_star,self.rho)
        
        ### Arnoldi Measures ###
        self.R_0 = init_res(self.J)
        self.R_inf = asym_res(self.J)
        self.I_S = np.zeros(self.k)
        for i in range(self.k):
            self.I_S[i] = stoch_invar(self.J[i,:,:])
        self.I_D = np.zeros(self.k)
        for i in range(self.k):
            self.I_D[i] = -1/minimize_scalar(determ_invar_OBJ, bracket = (0,3), args=(self.J[i,:,:]), method='brent').fun

        ### Diversity Measures ###
        self.simpson_s = simpson_conc(self.s_star)
        self.shannon_s = shannon_entropy(self.s_star)
        self.gini_simp_s = gini_simpson_index(self.s_star)
        self.hcdt_entropy_s = hcdt_entropy(self.s_star, q=2)
        self.renyi_entropy_s = renyi_entropy(self.s_star, q=2)
        self.D_s = D_1(self.s_star)
        
        
# List div_inds_, whichever diversity measures to include in plots (easier to just use idx)
## How deal with negative space? Leave blank for now? Find way to collapse negative space.
def OD_stab_div_plot(OD_, div_idx_=list(range(6)), fsize=(7,6), res = 250, fontsize=4, q=2, lowess_frac=None, lowess_col='black', title = '', one_col=None, s_=3, save_as=None, img_type='pdf'):
    q_renyi=q
    q_hcdt=q
    # q_renyi = int(input('q (order of diversity) for Renyi entropy?'))
    # q_hcdt = int(input('q for HCDT entropy?'))
    hcdt = hcdt_entropy(s_star, q=q_hcdt)
    renyi = renyi_entropy(s_star, q=q_renyi)

    xlist = np.array([OD_.simpson_s, OD_.shannon_s, OD_.gini_simp_s, hcdt, renyi, OD_.D_s], dtype=object)
    xlist = list(xlist[div_idx_])
    xnames = np.array(['Simpson', 'Shannon', 'Gini-Simpson', r'HCDT, q={}'.format(q_hcdt), 'Renyi, q={}'.format(q_renyi), 'D, Numbers Equivalent'], dtype=object)
    xnames = list(xnames[div_idx_])
    ylist = [OD_.R_0, OD_.R_inf, OD_.I_S, OD_.I_D]
    ynames = [r'$\mathcal{R}_0$', r'$\mathcal{R}_{\infty}$', r'$\mathcal{I}_S$', r'$\mathcal{I}_D$']

    plt.figure(figsize=fsize, dpi=res)
    plt.style.use('ggplot')
    mpl.rcParams.update({'font.size': fontsize})
    thetitle = 'Stability against diversity in resource-competition ($n=${}) \n{}'.format(OD_.n, title)
    if (one_col==None):
        # if lowess_frac:
            # thetitle = thetitle + '\nLOWESS with fraction {}'.format(np.round(lowess_frac,3))
        plt.suptitle(thetitle);

    colors = ['#648FFF','#785EF0','#DC267F','#FE6100','#FFB000', '#00000f']

    dimx = len(xlist)
    dimy = len(ylist)

    for i,stab in enumerate(ylist):
        for j,divind in enumerate(xlist):
            
            plt.subplot(dimy, dimx, i*dimx+j+1)
            if (i!=dimy-1):
                plt.tick_params('x', which='both', bottom=False, labelbottom=False);
            if (one_col==None):
                plt.scatter(divind, stab, s = s_, alpha=0.5, color = colors[j%dimx+len(colors)-2]);
            else:
                plt.scatter(divind, stab, s = s_, alpha=0.5, color = one_col);
            plt.xlim((1,OD_.S_star.shape[1]));
            if lowess_frac:
                w = lowess(stab.reshape(-1), divind.reshape(-1), frac=lowess_frac)
                if j != len(div_idx_)-1:
                    plt.plot(w[:,0], w[:,1], c=lowess_col, linestyle='dashed', linewidth=0.5, alpha=1.00)
                else:
                    if (one_col==None):
                        plt.plot(w[:,0], w[:,1], c='black', linestyle='dashed', linewidth=0.5, alpha=1.00)
                    else:
                        plt.plot(w[:,0], w[:,1], c=lowess_col, linestyle='dashed', linewidth=0.5, alpha=1.00)
            if j%dimx == 0:
                plt.ylabel(ynames[i]);
            if i == dimy-1:
                plt.xlabel(xnames[j]);

    
    plt.tight_layout();
    
    if (save_as != None):
        plt.savefig(save_as+'.{}'.format(img_type), format=img_type)
        
        
        
        

####### No Resource Production ########
        
def dot_across(A, x): # Overkill, slower than for-looping, really.
    k_ = A.shape[0]
    n_ = A.shape[1]
    DI_k = np.diag_indices(k_)[0]
    prod = np.dot(A,x.reshape(k_,n_,-1)) 
    return prod[DI_k,:,DI_k,:]

def mu_gen(eps_, C_, R_star_):
    n_ = C_.shape[-1]

    lhs = np.dot(eps_*np.eye(n_), C_)
    return np.transpose(np.dot(lhs, R_star_), (1,0,2))

def rho_gen(R_star_, C_, S_star_):
    k_ = R_star_.shape[0]
    n_ = C_.shape[-1]
    diag_idx = np.diag_indices(n_)[0]
    R_diag = np.zeros((k_,n_,n_))
    R_diag[:, diag_idx, diag_idx] = R_star_.reshape(k_, -1)
    
    lhs = np.dot(R_diag, C_)
    
    return dot_across(lhs, S_star_)

class ODsystem_noP:
    '''
    O'Dwyer systems + their community matrices and equilibria + Arnoldi Measures.
    VERY slow when n>~6, quadratic time from needing to for-loop I_S and I_D.
    -------------------------------------------------------
    Specify:
    k, number of systems
    n, number of species and number of resources
    R_star_bounds, resource equilibria random unif bounds (DEFUNCT, Need to sum to 1 anyway bc densities!!!)
    S_star_bounds, species equilibria random unif bounds
    eps, resource consumers' conversion efficiency of biomass
    sig_P, standard deviation of entries in production matrix P
    C, consumption matrix
    '''
    def __init__(self, k_=1000,n_=5, R_star_bounds=(1,2), S_star_bounds=(10,20), eps_=0.25, C_ = None, seed_=667):
        self.k = k_
        self.n = n_
        self.eps = eps_
        self.seed = seed_
        np.random.seed(self.seed)
        
        # self.R_star = np.random.uniform(R_star_bounds[0], R_star_bounds[1], (k_,n_,1))
        self.R_star = np.random.uniform(0,1, (k_,n_,1))
        self.R_star = self.R_star / np.sum(self.R_star, axis=1).reshape(self.k,1,1)
        # self.S_star = np.random.uniform(S_star_bounds[0], S_star_bounds[1], (k_,n_,1))
        self.S_star = np.random.uniform(0,1, (k_,n_,1))
        self.S_star = self.S_star / np.sum(self.S_star, axis=1).reshape(self.k,1,1)
        
        if C_ is None:
            self.C = np.identity(self.n)
        else:
            self.C = C_
        
        self.mu = mu_gen(self.eps, self.C, self.R_star)
        self.rho = rho_gen(self.R_star, self.C, self.S_star)
            
        self.J = community_matr_gen_noP(self.n, self.k, self.C,self.eps,self.R_star,self.S_star)
        
        
        ### Arnoldi Measures ###
        self.R_0 = init_res(self.J)
        self.R_inf = asym_res(self.J)
        self.I_S = np.zeros(self.k)
        for i in range(self.k):
            self.I_S[i] = stoch_invar(self.J[i,:,:])
            
        self.I_D = np.zeros(self.k)
        for i in range(self.k):
            self.I_D[i] = -1/minimize_scalar(determ_invar_OBJ, bracket = (0,3), args=(self.J[i,:,:]), method='brent').fun

            
        ### Diversity Measures ###
        self.s_star = sum_to_one(self.S_star)
        self.r_star = sum_to_one(self.R_star)

        self.simpson_s = simpson_conc(self.s_star)
        self.shannon_s = shannon_entropy(self.s_star)
        self.gini_simp_s = gini_simpson_index(self.s_star)
        self.hcdt_entropy_s = hcdt_entropy(self.s_star, q=2)
        self.renyi_entropy_s = renyi_entropy(self.s_star, q=2)
        self.D_s = 1/(1-self.gini_simp_s)
        