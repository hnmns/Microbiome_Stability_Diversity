import numpy as np

from scipy.linalg import solve as slv 


def lv_gen(k_, n_, sig_A, C_=1., diag_ = 'random', interaction_type_='random', is_sym_=False):
    ''' Symmetric system
    k, number of matrices
    n, number of species
    sig_A, standard deviation of interaction rates
    '''
    dinds = np.diag_indices(n_)
    A = np.random.normal(loc=0, scale=sig_A, size=(k_,n_,n_)) # 0 in the diagonal
    if diag_ == 'random':
        A[:,dinds[0],dinds[0]] = -np.abs(A[:,dinds[0],dinds[0]])
    elif diag_ == 'uniform':
        A[:,dinds[0],dinds[0]] = np.random.uniform(low=-1, high=0, size=(k_,n_))
    elif diag_ == -1:
        A[:,dinds[0],dinds[0]] = -1
    else:
        print('Invalid diagonal type, defaulting to negative half normal (random)')
        A[:,dinds[0],dinds[0]] = -np.abs(A[:,dinds[0],dinds[0]])
    
    if is_sym_:
        temp_ = np.triu(A, k=1)
        A = np.triu(A) + np.transpose(temp_, (0,2,1))
    
    if interaction_type_ == 'pred_prey':# For predator-prey-like opposite signs across diagonal
        # Will increase likelihood of endemic equ. stability? Only for closely related pred-prey pairs, though.
        A = np.triu(A) - np.abs(np.tril(A))*np.transpose(np.sign(np.triu(A,1)), (0,2,1))
    elif interaction_type_ == 'mutualism':
        Adiag = A[:,dinds[0],dinds[0]]
        A = np.abs(A)
        A[:,dinds[0],dinds[0]] = Adiag
    elif interaction_type_ == 'competition':
        Adiag = A[:,dinds[0],dinds[0]]
        A = -np.abs(A)
        A[:,dinds[0],dinds[0]] = Adiag
    elif interaction_type_ == 'random': # Just leave regular normal off-diag entries
        A = A
    else: # No Match case
        return "Invalid interaction type."
        
    # sample from bernoulli to replace entries in matrix of ones with 0
    connect = np.random.binomial(1, C_, (k_, n_, n_))
    connect[:,dinds[0],dinds[0]] = 1
    connect = np.triu(connect,k=1) + np.transpose(np.triu(connect), (0,2,1))
    A = A * connect

    return A

def lv_equilibrium(A_,r_):
    # Need to solve system -r_i = sum_over_j(beta_{i,j}*N_star_j)
    N_star = np.zeros((A_.shape[0], A_.shape[1], 1))
    for i in range(A_.shape[0]):
        N_star[i,:,:] = slv(a=A_[i,:,:], b=r_[i,:,:])
    
    return N_star

def lv_Jacobian(A_, r_, N_=None):
    if A_.ndim<3:
        A_ = np.expand_dims(A_, 0)
    if r_.ndim<3:
        r_ = np.expand_dims(r_, 0)
    if N_.ndim<3:
        N_ = np.expand_dims(N_, 0)
    k_ = A_.shape[0]
    n_ = A_.shape[1]
    
    if np.any(N_==None):
        N_star_ = lv_equilibrium(A_,r_)
    else:
        N_star_ = N_
        
    D_ = np.eye(n_) * r_.reshape(k_,n_,1)
    
    # Dotting k_ matrices with k_ vectors
    dinds_k = np.diag_indices(k_)[0]
    dot = np.dot(A_, N_star_.reshape(k_,n_,1))[dinds_k,:,dinds_k,:]
    D_ = D_ + np.eye(n_)*dot
    
    # There is an extra beta_{i,i}*N_i in each diag entry
    dinds_n = np.diag_indices(n_)[0]
    D_[:,dinds_n,dinds_n] = D_[:,dinds_n,dinds_n] + A_[:,dinds_n,dinds_n]*N_star_.reshape(k_,n_)
    
    # Off-diagonal entries
    temp_ = A_*N_star_.reshape(k_,1,n_)
    temp_[:,dinds_n,dinds_n] = 0
    
    D_ = D_ + temp_
    
    return D_



class LVsystem: # MUST CAST OUT UNSTABLE SYSTEMS!!! or just choose params better
    '''
    Lotka-Volterra systems + their community matrices and equilibria + Arnoldi Measures.
    -------------------------------------------------------
    Specify:
    k, number of systems
    n, number of species
    r, vector of intrinsic birth rates
    sig_A, standard deviation of entries in interaction matrix A
    C, connectance (as in May), probability that a pair of species interact,
                meaning non-zero a_ij and a_ji
    diag, either -1 or 'random', main diagonal entries of A
    interaction_type, type of A (interaction) matrix entries
                -'pred_prey': sign(a_ij) = -1 * sign(a_ji)
                -'mutualism': sign(a_ij) = sign(a_ji) = (+)
                -'random': no change
                -'competition': sign(a_ij) = sign(a_ji) = (-)
    is_sym, |a_ij|=|a_ji|
    '''
    def __init__(self, k_=1000,n_=5, r_=None, sig_A_=1, C_=1., diag_='random', interaction_type_='pred_prey', is_sym_=False):
        # Add list:
            # Replace pred_prey with interaction_type string parameter (pred_prey, mutual, competition, mixture)
            # Double check the units/magnitudes/funny business on O'Dwyer values (equilbria are crazy)
            # Use May/Allesina complexity bounds to check what is wrong with gLV community matrices.
        self.k = k_
        self.n = n_
        self.C = C_
        self.sig_A = sig_A_
        self.diag = diag_
        self.interaction_type = interaction_type_
        self.is_sym = is_sym_

        if r_ is None:
            self.r = np.random.uniform(0,1,(self.k,self.n,1))
        else:
            self.r = r_
       
        self.A = lv_gen(self.k, self.n, self.sig_A, self.C, diag_=self.diag, interaction_type_ = self.interaction_type, is_sym_=self.is_sym)
        
        self.J = lv_Jacobian(self.A, self.r)
        
        self.N_star = lv_equilibrium(self.A, self.r).reshape(self.k,self.n,1)

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
        self.n_star = sum_to_one(self.N_star)

        self.simpson = simpson_conc(self.n_star)
        self.shannon = shannon_entropy(self.n_star)
        self.gini_simp = gini_simpson_index(self.n_star)
        self.hcdt_entropy = hcdt_entropy(self.n_star, q=2)
        self.renyi_entropy = renyi_entropy(self.n_star, q=2)
        self.D = 1/(1-self.gini_simp)
        
        # DON'T have working stable gLV systems yet, so can't do stability-diversity plots.
#     def stab_div_plot(self, fsize=(7,6), res = 250, fontsize=4, q=2, lowess_frac=None):
#         q_renyi=q
#         q_hcdt=q
#         # q_renyi = int(input('q (order of diversity) for Renyi entropy?'))
#         # q_hcdt = int(input('q for HCDT entropy?'))
#         hcdt = hcdt_entropy(self.n_star, q=q_hcdt)
#         renyi = renyi_entropy(self.n_star, q=q_renyi)
                
#         xlist = [self.simpson, self.shannon, self.gini_simp, hcdt, renyi, self.D]
#         xnames = ['Simpson', 'Shannon', 'Gini-Simpson', r'HCDT, q={}'.format(q_hcdt), 'Renyi, q={}'.format(q_renyi), 'Numbers Equivalent']
#         ylist = [self.R_0, self.R_inf, self.I_S, self.I_D]
#         ynames = [r'$\mathcal{R}_0$', r'$\mathcal{R}_{\infty}$', r'$\mathcal{I}_S$', r'$\mathcal{I}_D$']
        
#         plt.figure(figsize=fsize, dpi=res)
#         plt.style.use('ggplot')
#         mpl.rcParams.update({'font.size': fontsize})
#         thetitle = 'gLV Stability Measures vs. Diversity Indices ($n=${})'.format(self.n)
#         if lowess_frac:
#             thetitle = thetitle + '\nLOWESS with fraction {}'.format(np.round(lowess_frac,3))
#         plt.suptitle(thetitle);

        
#         colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4', '#000000']
        
#         dimx = len(xlist)
#         dimy = len(ylist)
        
#         for i,stab in enumerate(ylist):
#             for j,divind in enumerate(xlist):
#                 plt.subplot(4,6,i*dimx+j+1)
#                 plt.scatter(divind, stab, s = 1, alpha=0.5, color = colors[j%dimx]);
#                 if lowess_frac:
#                     w = lowess(stab.reshape(-1), divind.reshape(-1), frac=lowess_frac)
#                     if j != 5:
#                         plt.plot(w[:,0], w[:,1], c='black', linestyle='dashed', linewidth=0.5, alpha=1.00)
#                     else:
#                         plt.plot(w[:,0], w[:,1], c='red', linestyle='dashed', linewidth=0.5, alpha=1.00)
#                 if j%dimx == 0:
#                     plt.ylabel(ynames[i]);
#                 if i == dimy-1:
#                     plt.xlabel(xnames[j]);

#         plt.tight_layout();    


