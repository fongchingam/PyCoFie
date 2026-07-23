from operator import mul
import functools
import numpy as np
from streamtracer import StreamTracer, VectorGrid
from scipy.special import gammaln, assoc_legendre_p_all
from scipy.interpolate import interpn
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_ct_p_grids(Nt, Np, eqtheta = False, gridcenter = False):
    if gridcenter == True:
        # if true return value on the center on the grid
        # if false return value on the edge of the cell
        Nt += 1
        Np += 1
    
    if eqtheta == True:
        theta = np.linspace(np.pi, 0, Nt)
        costheta = np.cos(theta)
    else:
        costheta = np.linspace(-1, 1, Nt)
    phi = np.linspace(0, 2*np.pi, Np)
    if gridcenter == True:
        phi_l = (phi[1:] + phi[:-1])/2
        costheta_l = (costheta[:-1] + costheta[1:]) /2
    else:
        phi_l = phi
        costheta_l = costheta
        #costheta_l[0], costheta_l[-1] = (costheta[0]+costheta[1])/2, (costheta[-2] + costheta[-1])/2
    sintheta_l = np.sin(np.arccos(costheta_l))
    costheta, phi = np.meshgrid(costheta_l, phi_l)
    return costheta, phi, costheta_l, sintheta_l, phi_l

def get_allPdP(mbig, lbig, costheta):
    # assoc_legendre_p_all computes all orders up to lbig
    # returns (values, derivatives) 
    P_all, dP_all = assoc_legendre_p_all(mbig, lbig, costheta, diff_n=1)

    # Shapes: (Nt, lbig+1, lbig+1)
    # P_all[i, l, m] = P_l^m(costheta[i])
    # dP_all[i, l, m] = derivative wrt costheta

    # Now slice to match mbig
    all_Plm = P_all[:lbig+1, :mbig+1, :].transpose(2, 1, 0)
    all_dPlm = dP_all[:lbig+1, :mbig+1, :].transpose(2, 1, 0)
    # Replace infinities with 0
    all_dPlm = np.where(np.abs(all_dPlm) == np.inf, 0, all_dPlm) 
    return all_Plm, all_dPlm

def get_gh(Bra, N=85):
    sz = np.shape(Bra)
    X, Y = sz[1], sz[0]
    Nt, Np = Y, X
    costheta, phi, costheta_l, sintheta_l, phi_l = get_ct_p_grids(Nt, Np)
    all_Plm, all_dPlm = get_allPdP(N, N, costheta_l)

    def compute_lm(args):
        l, m = args
        
        ratio = np.exp(gammaln(l-m+1) - gammaln(l+m+1))
        d0 = 1 if m == 0 else 0
        
        norm = (-1)**m * np.sqrt(ratio * (2-d0))
        lpmv = np.ones([Np, Nt]) * all_Plm[:, int(m), int(l)]
        coeff = (2*l+1)/X/Y
        glm = coeff * np.sum(norm * lpmv * np.cos(m*phi) * Bra.T)
        hlm = coeff * np.sum(norm * lpmv * np.sin(m*phi) * Bra.T)
        return np.array([l, m, glm, hlm], dtype = object)

    # Build ordered list of (l,m) pairs
    lm_pairs = [(l, m) for l in range(N+1) for m in range(l+1)]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_lm, lm_pairs))

    # Filter out None entries
    gh_result = np.array([r for r in results if r is not None], dtype = object)
    #gh_result = np.where(np.isnan(gh_result), 0, gh_result)
    del results, lm_pairs, costheta, phi, costheta_l, sintheta_l, phi_l, all_Plm, all_dPlm
    return gh_result

def accumulate_in_place(res, Rl_results, Br, Bt, Bp):
    l, hm_r, hm_t, hm_p = res
    Rl_r, Rl_tp = Rl_results[int(l)]
    Rl_r = np.array(Rl_r, dtype=np.float32)
    Rl_tp = np.array(Rl_tp, dtype=np.float32)

    Br += hm_r * Rl_r[:, None, None]
    Bt += hm_t * Rl_tp[:, None, None]
    Bp += hm_p * Rl_tp[:, None, None]

def process_chunk(args):
    chunk, Rl_results, Nr, Np, Nt = args
    Br = np.zeros([Nr, Np, Nt], dtype=np.float32)
    Bt = np.zeros_like(Br)
    Bp = np.zeros_like(Br)
    for res in chunk:
        if res is None:
            continue
        
        l, hm_r, hm_t, hm_p = res
        Rl_r, Rl_tp = Rl_results[int(l)]
        Rl_r = np.array(Rl_r, dtype=np.float32)
        Rl_tp = np.array(Rl_tp, dtype=np.float32)
        Br += hm_r * Rl_r[:, None, None]
        Bt += hm_t * Rl_tp[:, None, None]
        Bp += hm_p * Rl_tp[:, None, None]
    return Br, Bt, Bp

def rec_Brtp_PFSS(gh_array, r, N = 85, Rs = 2.5, 
             Np = 360, Nt = 180, eqtheta = False, gridcenter = True,
             nproc = 14):
    
    Nr = len(r)

    costheta, phi, costheta_l, sintheta_l, phi_l = get_ct_p_grids(Nt, Np, eqtheta = eqtheta, gridcenter = gridcenter)
    with np.errstate(divide='ignore', invalid='ignore'):
        div_sintheta_l = 1/sintheta_l
        div_sintheta_l = np.where(abs(sintheta_l) < 1e-15, 0, div_sintheta_l)

    if N == 'all' or N > len(gh_array):
        print("Warning: N is set to 'all' or greater than the number of available coefficients. Using all coefficients.")
        N_rows = len(gh_array)
        order = int(gh_array[-1][0])
    else:
        N_rows = int((1+N+1)*(N+1)/2)
        order = N
    
    all_Plm, all_dPlm = get_allPdP(order, order, costheta_l)

    def compute_Rl(r, l, Rs):
        Rl_r = (1/r)**(l+2) * (l+1+l*(r/Rs)**(2*l+1)) / (l+1+l*(1/Rs)**(2*l+1))
        Rl_tp = (1/r)**(l+2) * (1-(r/Rs)**(2*l+1)) / (l+1+l*(1/Rs)**(2*l+1))
        return Rl_r, Rl_tp
    
    def compute_term(entry):
        l, m, g, h = entry

        d0 = 1 if m == 0 else 0
        ratio = np.exp(gammaln(l-m+1) - gammaln(l+m+1))
        
        norm = (-1)**m * np.sqrt(ratio * (2-d0))

        gc, hs, gs, hc = g*np.cos(m*phi), h*np.sin(m*phi), g*np.sin(m*phi), h*np.cos(m*phi)

        lpmv = np.ones([Np, Nt]) * all_Plm[:, int(m), int(l)]
        dpmv = np.ones([Np, Nt]) * all_dPlm[:, int(m), int(l)]

        hm_r = np.array(norm * lpmv * (gc + hs), dtype=np.float32)
        hm_t = np.array(norm * dpmv * (gc + hs) * sintheta_l, dtype=np.float32)
        hm_p = np.array(m * norm * lpmv * div_sintheta_l * (gs - hc), dtype=np.float32)
        return l, hm_r, hm_t, hm_p
    
    # Parallel execution
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_term, gh_array[:N_rows]))

    with ThreadPoolExecutor() as executor:
        Rl_results = list(executor.map(lambda l: compute_Rl(r, l, Rs), range(order+1)))
    
    # Divide results into chunks
    chunksize = max(1, len(results)//nproc)
    chunks = [results[i:i+chunksize] for i in range(0, len(results), chunksize)]

    # Thread-based parallel accumulation
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        args = [(c, Rl_results, Nr, Np, Nt) for c in chunks]
        partials = list(executor.map(process_chunk, args))

    Br = sum(p[0] for p in partials)
    Bt = sum(p[1] for p in partials)
    Bp = sum(p[2] for p in partials)
    del all_Plm, all_dPlm, results, Rl_results

    return Br, Bt, Bp

############ CAUTION ###############
# These params are defined as the factor in front the calculation, NOT just R listed in table in section 5 of pfss.pdf

def HCS_R_L(r, l, a = 1):
    # Below cusp surface (Rc < Rss)
    result = (1+a)**l/(l+1)/(r+a)**(l+1) / r
    return result

def HCS_R_H(r, l, Rc = 1.7, Rs = 2.5, a = 1):
    # CAUTION: There's a 1/r at the end, for B(theta/phi) \propto 1/r*dPhi/d(theta/phi)
    top = 1/(r+a)**(l+1) - (r+a)**l/(Rs+a)**(2*l+1)
    bottom = (l+1)/(Rc+a)**l + l*(Rc+a)**(l+1)/(Rs+a)**(2*l+1)
    result = Rc**2* top/bottom  /r
    return result

def HCS_dR_L(r, l, a = 1):
    # CAUTION: automatically attach ita(r, a) to dR since dR always go with Br, which has ita
    ita = (1+a/r)**2
    result = ita* (1+a)**l/(r+a)**(l+2)
    return result

def HCS_dR_H(r, l, Rc = 1.7, Rs = 2.5, a = 1):
    ita = (1+a/r)**2
    top = (l+1)/(r+a)**(l+2) + l*(r+a)**(l-1)/(Rs+a)**(2*l+1)
    bottom = (l+1)/(Rc+a)**l + l*(Rc+a)**(l+1)/(Rs+a)**(2*l+1)
    result = ita* Rc**2* top/bottom
    return result

############ END CAUTION ########################    


def rec_Brtp_CSSS_low(gh_array, r_all, 
                      phi, costheta_l, sintheta_l, div_sintheta_l,
                      Rc=1.7, a=1, N=85, 
                      nproc = 14):

    r = np.append(r_all[np.where(r_all < Rc)], Rc)
    Nr = len(r)
    Nt, Np = len(costheta_l), len(phi)

    if N == 'all' or N > len(gh_array):
        print("Warning: N is set to 'all' or greater than the number of available coefficients. Using all coefficients.")
        N_rows = len(gh_array)
        order = int(gh_array[-1][0])
    else:
        N_rows = int((1+N+1)*(N+1)/2)
        order = N

    all_Plm, all_dPlm = get_allPdP(order, order, costheta_l)

    def compute_Rl(r, l, a):
        Rl_r = HCS_dR_L(r, l, a=a)
        Rl_tp = HCS_R_L(r, l, a=a)
        return Rl_r, Rl_tp

    def compute_term(entry):
        l, m, g, h = entry
        if l == 0:
            return None  # skip 0 term
        d0 = 1 if m == 0 else 0
        ratio = np.exp(gammaln(l-m+1) - gammaln(l+m+1))
        
        norm = (-1)**m * np.sqrt(ratio * (2-d0))

        gc, hs, gs, hc = g*np.cos(m*phi), h*np.sin(m*phi), g*np.sin(m*phi), h*np.cos(m*phi)

        lpmv = np.ones([Np, Nt]) * all_Plm[:, int(m), int(l)]
        dpmv = np.ones([Np, Nt]) * all_dPlm[:, int(m), int(l)]

        hm_r = np.array(norm * lpmv * (gc + hs), dtype=np.float32)
        hm_t = np.array(norm * dpmv * (gc + hs) * sintheta_l, dtype=np.float32)
        hm_p = np.array(m * norm * lpmv * div_sintheta_l * (gs - hc), dtype=np.float32)

        return l, hm_r, hm_t, hm_p

    # Parallel execution
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_term, gh_array[:N_rows]))

    with ThreadPoolExecutor() as executor:
        Rl_results = list(executor.map(lambda l: compute_Rl(r, l, a), range(order+1)))

    # Divide results into chunks
    chunksize = max(1, len(results)//nproc)
    chunks = [results[i:i+chunksize] for i in range(0, len(results), chunksize)]

    # Thread-based parallel accumulation
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        args = [(c, Rl_results, Nr, Np, Nt) for c in chunks]
        partials = list(executor.map(process_chunk, args))

    Br = sum(p[0] for p in partials)
    Bt = sum(p[1] for p in partials)
    Bp = sum(p[2] for p in partials)

    del all_Plm, all_dPlm, results, Rl_results

    return Br, Bt, Bp

### Construct alpha beta matrix

def Kl(l, Rs = 2.5, Rc = 1.7, a = 1):
    top = ((Rs+a)**(2*l+1) - (Rc + a)**(2*l + 1))*Rc
    bot = ((l + 1)*(Rs + a)**(2*l+1) + l*(Rc + a)**(2*l+1))*(Rc + a)
    return top/bot

### Alpha beta matrix the indexes 

def get_albe_ind(order, Nt, Np):

    abind = np.zeros([int((order+1)**2), Nt*Np*3, 5], dtype = float)
    
    ### Setting the first 2 number as l, m
    lmind = np.array([(0, 0)])
    for i in np.arange(order)+1:
        for j in np.arange(i+1):
            lmind = np.append(lmind, [(i,j)], axis = 0)
    for i in np.arange(order)+1:
        for j in np.arange(i)+1:
            lmind = np.append(lmind, [(i,j)], axis = 0)
    for j in np.arange(Nt*Np*3):
        abind[:int((order+1)**2), j, :2] = lmind
        
    # follows Zhao (1995) paper logic
    
    ct_i, phi_i, i_i = np.arange(Nt), np.arange(Np), np.arange(3)+1
    i_i, ct_i, phi_i = np.meshgrid(i_i, ct_i, phi_i)
    f_ci, f_pi, f_i = ct_i.flatten(), phi_i.flatten(), i_i.flatten()

    ct_p_i_i = np.zeros([Nt*Np*3, 3])
    ct_p_i_i[:, 0] = f_ci
    ct_p_i_i[:, 1] = f_pi
    ct_p_i_i[:, 2] = f_i

    for row in abind:
        row[:, 2:] = ct_p_i_i
    return abind

### Alpha function

def alpha_byind(inds, all_Plm, all_dPlm, 
                phi_l, sintheta_l, div_sintheta_l,
                Rs=2.5, Rc=1.7, a=1):
    l     = inds[:, :, 0].astype(int)
    m     = inds[:, :, 1].astype(int)
    ct_i  = inds[:, :, 2].astype(int)
    phi_i = inds[:, :, 3].astype(int)
    rtp   = inds[:, :, 4].astype(int)

    p  = phi_l[phi_i]
    st = sintheta_l[ct_i]
    dst = div_sintheta_l[ct_i]

    d0   = (m == 0).astype(int)
    ratio = np.exp(gammaln(l-m+1) - gammaln(l+m+1))
    norm = (-1)**m * np.sqrt(ratio * (2-d0))

    cosmp  = np.cos(m*p)
    sinmp  = np.sin(m*p)
    Plm    = all_Plm[ct_i, m, l]
    dPlm   = all_dPlm[ct_i, m, l]
    Kl_val = Kl(l, Rs=Rs, Rc=Rc, a=a)

    result = np.zeros(np.shape(cosmp), dtype=np.float64)

    mask1 = (rtp == 1)
    mask2 = (rtp == 2)
    mask3 = (rtp == 3)

    # Each of these is 1D, so just use one mask
    result[mask1] = cosmp[mask1] * norm[mask1] * Plm[mask1]
    result[mask2] = -Kl_val[mask2] * cosmp[mask2] * norm[mask2] * dPlm[mask2] * (-st[mask2])
    result[mask3] = m[mask3] * Kl_val[mask3] * sinmp[mask3] * norm[mask3] * Plm[mask3] *dst[mask3] #/ st[mask3]

    return result

def beta_byind(inds, all_Plm, all_dPlm, 
               phi_l, sintheta_l, div_sintheta_l,
               Rs=2.5, Rc=1.7, a=1):
    l     = inds[:, :, 0].astype(int)
    m     = inds[:, :, 1].astype(int)
    ct_i  = inds[:, :, 2].astype(int)
    phi_i = inds[:, :, 3].astype(int)
    rtp   = inds[:, :, 4].astype(int)

    p  = phi_l[phi_i]
    st = sintheta_l[ct_i]
    dst = div_sintheta_l[ct_i]

    d0   = (m == 0).astype(int)
    ratio = np.exp(gammaln(l-m+1) - gammaln(l+m+1))
    norm = (-1)**m * np.sqrt(ratio * (2-d0))

    sinmp  = np.sin(m*p)
    cosmp  = np.cos(m*p)
    Plm    = all_Plm[ct_i, m, l]
    dPlm   = all_dPlm[ct_i, m, l]
    Kl_val = Kl(l, Rs=Rs, Rc=Rc, a=a)

    result = np.zeros(np.shape(cosmp), dtype=np.float64)

    mask1 = (rtp == 1)
    mask2 = (rtp == 2)
    mask3 = (rtp == 3)

    result[mask1] = sinmp[mask1] * norm[mask1] * Plm[mask1]
    result[mask2] = -Kl_val[mask2] * sinmp[mask2] * norm[mask2] * dPlm[mask2] * (-st[mask2])
    result[mask3] = -m[mask3] * Kl_val[mask3] * cosmp[mask3] * norm[mask3] * Plm[mask3] *dst[mask3] #/ st[mask3]

    return result

def get_albe_AB(order, Nt, Np, 
                costheta_l, sintheta_l, phi_l, div_sintheta_l,
                Rs = 2.5, Rc = 1.7, a = 1):

    all_Plm, all_dPlm = get_allPdP(order, order, costheta_l)

    abind = get_albe_ind(order, Nt, Np)
    abmat = np.zeros([int((order+1)**2), Nt*Np*3])

    div_row = int((order+2)*(order+1)/2)
    abmat[:div_row] = alpha_byind(abind[:div_row, :], all_Plm, all_dPlm, 
                                  phi_l, sintheta_l, div_sintheta_l,
                                  Rs = Rs, Rc = Rc, a = a)
    abmat[div_row:] = beta_byind(abind[div_row:, :], all_Plm, all_dPlm, 
                                 phi_l, sintheta_l, div_sintheta_l, Rs = Rs, Rc = Rc, a = a)

    ABmat = np.matmul(abmat, abmat.T)
    
    ABinv = np.linalg.inv(ABmat)
    return ABinv, abmat

def get_gh_cusp(Br_cusp, Bt_cusp, Bp_cusp, 
                costheta_l, sintheta_l, phi_l, div_sintheta_l,
                Rs = 2.5, Rc = 1.7, a = 1,
                order=9):
    Nt, Np = Br_cusp.shape
    ABinv, abmat = get_albe_AB(order, Nt, Np,
                               costheta_l, sintheta_l, phi_l, div_sintheta_l,
                               Rs = Rs, Rc = Rc, a = a)

    # invert the field
    rev_Br = np.where(Br_cusp < 0, -Br_cusp, Br_cusp)
    rev_Bt = np.where(Br_cusp < 0, -Bt_cusp, Bt_cusp)
    rev_Bp = np.where(Br_cusp < 0, -Bp_cusp, Bp_cusp)
    Ba = np.concatenate([rev_Br, rev_Bt, rev_Bp], axis=1).ravel()

    gh_c = ABinv @ (abmat @ Ba.T)

    # Generate all (l,m) pairs correctly
    L = np.repeat(np.arange(order + 1), np.arange(1, order + 2))
    M = np.concatenate([np.arange(l + 1) for l in range(order + 1)])

    # g_ini and h_ini closed-form
    g_ini = L * (L + 1) // 2
    h_ini = L * (L - 1) // 2 + (order + 1) * (order + 2) // 2

    g_ind = g_ini + M
    glm = gh_c[g_ind]

    # hlm only for m != 0
    hlm = np.zeros_like(glm)
    mask = M != 0
    hlm[mask] = gh_c[h_ini[mask] + M[mask] - 1]

    # Stack into final array
    gaha_result = np.column_stack([L, M, glm, hlm]).astype(object)

    return gaha_result

def rec_Brtp_CSSS_high(gh_array, r_all, 
                       phi, costheta_l, sintheta_l, div_sintheta_l,
                       N = 85, Rs = 2.5, Rc = 1.7, a = 1, 
                       nproc = 14):
    r = r_all[np.where(r_all > Rc)]
    Nr = len(r)
    Nt, Np = len(costheta_l), len(phi)

    if N == 'all' or N > len(gh_array):
        print("Warning: N is set to 'all' or greater than the number of available coefficients. Using all coefficients.")
        N_rows = len(gh_array)
        order = int(gh_array[-1][0])
    else:
        N_rows = int((1+N+1)*(N+1)/2)
        order = N
    
    all_Plm, all_dPlm = get_allPdP(order, order, costheta_l)

    def compute_Rh(r, l, a, Rc, Rs):
        Rl_r = HCS_dR_H(r, l, a=a, Rc=Rc, Rs=Rs)
        Rl_tp = HCS_R_H(r, l, a=a, Rc=Rc, Rs=Rs)
        return Rl_r, Rl_tp
    
    def compute_term(entry):
        l, m, g, h = entry
 
        d0 = 1 if m == 0 else 0
        ratio = np.exp(gammaln(l-m+1) - gammaln(l+m+1))
        
        norm = (-1)**m * np.sqrt(ratio * (2-d0))

        gc, hs, gs, hc = g*np.cos(m*phi), h*np.sin(m*phi), g*np.sin(m*phi), h*np.cos(m*phi)

        lpmv = np.ones([Np, Nt]) * all_Plm[:, int(m), int(l)]
        dpmv = np.ones([Np, Nt]) * all_dPlm[:, int(m), int(l)]

        hm_r = np.array(norm * lpmv * (gc + hs), dtype=np.float32)
        hm_t = np.array(norm * dpmv * (gc + hs) * sintheta_l, dtype=np.float32)
        hm_p = np.array(m * norm * lpmv * div_sintheta_l * (gs - hc), dtype=np.float32)

        return l, hm_r, hm_t, hm_p
    
    # Parallel execution
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_term, gh_array[:N_rows]))

    with ThreadPoolExecutor() as executor:
        Rh_results = list(executor.map(lambda l: compute_Rh(r, l, a, Rc, Rs), range(order+1)))

    # Divide results into chunks
    chunksize = max(1, len(results)//nproc)
    chunks = [results[i:i+chunksize] for i in range(0, len(results), chunksize)]

    # Thread-based parallel accumulation
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        args = [(c, Rh_results, Nr, Np, Nt) for c in chunks]
        partials = list(executor.map(process_chunk, args))

    Br_h = sum(p[0] for p in partials)
    Bt_h = sum(p[1] for p in partials)
    Bp_h = sum(p[2] for p in partials)
    del all_Plm, all_dPlm, results, Rh_results

    return Br_h, Bt_h, Bp_h

def rec_Brtp_CSSS_full(gh_array, r_all, 
                       N = 85, Ncusp = 9,
                       Np = 360, Nt = 180, 
                       Rs = 2.5, Rc = 1.7, a = 1, 
                       eqtheta = False, gridcenter = True,
                       nproc = 14, step_size = 0.1):

    costheta, phi, costheta_l, sintheta_l, phi_l = get_ct_p_grids(Nt, Np, eqtheta=eqtheta, gridcenter=gridcenter)
    with np.errstate(divide='ignore', invalid='ignore'):
        div_sintheta_l = 1/sintheta_l
        div_sintheta_l = np.where(abs(sintheta_l) < 1e-15, 0, div_sintheta_l)

    # Calculate the field in the lower region, including the cusp surface for 2nd boundary condition
    Br_l, Bt_l, Bp_l = rec_Brtp_CSSS_low(gh_array, r_all, 
                                         phi, costheta_l, sintheta_l, div_sintheta_l,
                                         N=N, Rc=Rc, a=a, 
                                         nproc=nproc)
    Br_cusp, Bt_cusp, Bp_cusp = (Br_l[-1]).T, (Bt_l[-1]).T, (Bp_l[-1]).T

    # Spherical harmonics for the cusp surface
    gaha_result = get_gh_cusp(Br_cusp, Bt_cusp, Bp_cusp, 
                              costheta_l, sintheta_l, phi_l, div_sintheta_l,
                              Rs = Rs, Rc = Rc, a = a,
                              order = Ncusp)

    # Calculate the field in the upper region, using the cusp surface as the boundary condition
    Br_h, Bt_h, Bp_h = rec_Brtp_CSSS_high(gaha_result, r_all, 
                                          phi, costheta_l, sintheta_l, div_sintheta_l,
                                        N=Ncusp, 
                                        Rs=Rs, Rc=Rc, a=a, 
                                        nproc=nproc)

    # Temporary upper field grid including cusp surface for field line tracing
    rev_Br = np.where(Br_cusp < 0, -Br_cusp, Br_cusp).T
    rev_Bt = np.where(Br_cusp < 0, -Bt_cusp, Bt_cusp).T
    rev_Bp = np.where(Br_cusp < 0, -Bp_cusp, Bp_cusp).T

    Br_uptemp = np.append(rev_Br[np.newaxis, :, :], Br_h, axis = 0)
    Bt_uptemp = np.append(rev_Bt[np.newaxis, :, :], Bt_h, axis = 0)
    Bp_uptemp = np.append(rev_Bp[np.newaxis, :, :], Bp_h, axis = 0)

    r_up = np.append(np.array([Rc]), r_all[np.where(r_all > Rc)])

    # Trace all field lines in the upper region, to find their polarity
    nsteps = int((Rs - Rc)/step_size)
    tracer = StreamTracer(nsteps, step_size)
        
    #costheta, phi, costheta_l, sintheta_l, phi_l = get_ct_p_grids(Nt, Np, eqtheta = False, gridcenter = False)
    t_coord = np.arccos(costheta_l)
    p_coord = phi_l

    vector_grid_up = build_sph_vector_grid(Br_uptemp, Bt_uptemp, Bp_uptemp, 
                                        r_up, t_coord, p_coord,
                                        fliptheta = True)
    Br_const, Bt_const, Bp_const = Br_uptemp.copy(), Bt_uptemp.copy(), Bp_uptemp.copy()

    # Build seeds for all points in vector grid
    all_seeds_r, all_seeds_phi, all_seeds_theta = np.meshgrid(r_up, p_coord, t_coord, indexing='ij')
    all_seeds = np.array([all_seeds_r.flatten(),
                        all_seeds_phi.flatten(),
                        all_seeds_theta.flatten()]).T

    # Trace all seeds downward to determine the polarity of the field lines
    tracer.trace(all_seeds, vector_grid_up, -1)

    # Extract lowest positions
    line_inis = np.array([tracer.xs[i][0] for i in range(len(all_seeds))])
    line_inis = line_inis.reshape(len(r_up), Np, Nt, 3)
    
    icoords = np.stack([line_inis[..., 2], line_inis[..., 1]], axis=-1)  # shape (Nr, Np, Nt, 2)
    icoords_flat = icoords.reshape(-1, 2)

    # Find the values of line origins on the cusp surface using interpolation
    Binterp = interpn((t_coord, p_coord), Br_cusp, icoords_flat)
    Binterp = Binterp.reshape(len(r_up), Np, Nt)

    # Compute signs and apply in one broadcast
    signs = np.sign(Binterp)
    Br_const *= signs
    Bt_const *= signs
    Bp_const *= signs

    # Combine the lower and upper region fields, ensuring the cusp surface is not duplicated
    Br_full, Bt_full, Bp_full = np.append(Br_l[:-1], Br_const[1:], axis = 0), np.append(Bt_l[:-1], Bt_const[1:], axis = 0), np.append(Bp_l[:-1], Bp_const[1:], axis = 0)
    return Br_full, Bt_full, Bp_full
    
def build_sph_vector_grid(Br, Bt, Bp, 
                      r_coord, t_coord, p_coord,
                      fliptheta = True):
    # My ordering: (r, p, t) (Don't ask why)
    if fliptheta == True:
        # fliptheta: reverse theta ordering, making sure theta value increase north to south
        Br, Bt, Bp = Br[:, :, ::-1], Bt[:, :, ::-1], Bp[:, :, ::-1]
        t_coord = t_coord[::-1]
    
    field = np.ones((np.shape(Br)[0], np.shape(Br)[1], np.shape(Br)[2], 3))
    field[:, :, :, 0], field[:, :, :, 1], field[:, :, :, 2] = Br, Bp, Bt
    
    sgcorr, rcorr = np.abs(np.sin(t_coord))[np.newaxis, np.newaxis, :], r_coord[:, np.newaxis, np.newaxis]
    # phi corrections 
    with np.errstate(divide='ignore', invalid='ignore'):
        field[:, :, :, 1] /= sgcorr 
    field[:, :, :, 1] /= rcorr 
    field[:, :, 0, 1] = 0 # north and south pole singularity
    field[:, :, -1, 1] = 0
    
    # theta corrections 
    field[:, :, :, 2] /= rcorr 
    #field[:, :, :, 2] *= -sgcorr #'''
    
    # Force cyclic 
    field[:, -1, :, :] = field[:, 0, :, :]
    
    # build the vector grid
    cyclic = [False, True, False]
    grid_coords = [r_coord, p_coord, t_coord]
    vector_grid = VectorGrid(field, cyclic=cyclic, grid_coords=grid_coords)
    return vector_grid
