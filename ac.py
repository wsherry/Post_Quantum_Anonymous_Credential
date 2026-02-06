import numpy as np
from numpy.polynomial import polynomial as poly
from numpy import linalg as LA
import random
import copy
from sympy import *
import sage.all as sg
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
import math


def mod_2D(x, q):
    r = (x + np.floor(q/2)) % q - np.floor(q/2) 
    r[r <= -np.floor(q/2)] += q
    return r

def mod_3D(rows, x, q):
    for i in range(rows):
        x[i] = mod_2D(x[i],q)
    return x

def poly_matmul(a,b, mod, poly_mod, N=2048):
    matmul = []
    a = np.array(a)
    b = np.array(b)
    try:
        a_n = a.shape[0]
        a_m = a.shape[1]
        a_k = a.shape[2]
        b_n = b.shape[0]
        b_m = b.shape[1]
        b_k = b.shape[2]
    except IndexError:
        raise
    if a_m != b_n:
        raise ValueError("Cannot multiply matrices with different dimensions", a.shape, b.shape)
    elif a_k !=b_k:
        raise ValueError("Length of polynomial are different")

    for i in range(a_n):
        for j in range(b_m):
            element=np.array([0]*N)   
            for k in range(b_n):
                poly_val =  np.polydiv(np.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j])) ,poly_mod)[1]
                if np.array(poly_val).shape[0]>N:
                    print(np.array(poly_val).shape)
                    print("POLYVAL greater than N", poly_val)
                intermediate_element = np.polydiv(np.polyadd(np.array(element), np.array(poly_val)) , poly_mod)[1]
                element = mod_2D(intermediate_element.c, mod)
                if len(element) < N:
                    element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            if len(element) < N:
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            matmul.append(element)
    matmul = np.reshape(np.array(matmul), (a_n,b_m,N))        
    return matmul

def neg_poly_multiplication(neg_poly, neg_poly_exp, poly, poly_exp, poly_mod, d):

    max_degree = max(len(neg_poly), len(poly),d)
    res = np.array([0]*(int(max_degree*2+1)))
    for i in range(len(neg_poly)):
        temp = np.array([0]*(int(max_degree)*2+1))
        for j in range(len(poly)):
            exp = neg_poly_exp[i] + poly_exp[j] + max_degree
            temp[exp] = temp[exp] + neg_poly[i]*poly[j]
        res = res + temp
    res_neg = res[:max_degree]
    res_pos = np.flip(res[max_degree:])
    element_neg = np.polydiv(res_neg, poly_mod)[1]*-1
    element_neg[0] = element_neg[0]*-1
    element_pos = np.polydiv(res_pos,poly_mod)[1]
    if len(np.array(element_pos)) < d+1:
        element_pos = np.pad(element_pos, (d-len(np.array(element_pos)),0), 'constant', constant_values=0)  
    if len(np.array(element_neg)) < d+1:
        element_neg = np.pad(element_neg, (d-len(np.array(element_neg)),0), 'constant', constant_values=0)  
    element = np.array(element_pos) + np.array(element_neg)
    return element

def neg_poly_multiplication_right(poly, poly_exp,neg_poly, neg_poly_exp,poly_mod, d):
    res = np.array([0]*(d*2+1))
    for i in range(len(neg_poly)):
        temp = np.array([0]*(d*2+1))
        for j in range(len(poly)):
            exp = neg_poly_exp[i] + poly_exp[j] + d 
            temp[exp] = temp[exp] + poly[j]*neg_poly[i]
        res = res + temp
    res_neg = np.append(res[:d], [0])
    res_pos = np.flip(res[d:])

    element_neg = np.polydiv(res_neg, poly_mod)[1]*-1
    element_neg[0] = element_neg[0]*-1
    element_pos = np.polydiv(res_pos,poly_mod)[1]

    element = element_neg + element_pos
    if len(np.array(element)) < d:
        element = np.pad(element, (d-len(np.array(element)),0), 'constant', constant_values=0)  

    return element

def neg_poly_matmul(neg_a, b, mod, poly_mod, N=2048):
    matmul = []
    neg_a = np.array(neg_a)
    b = np.array(b)
    try:
        a_n = neg_a.shape[0]
        a_m = neg_a.shape[1]
        a_k = neg_a.shape[2]
        b_n = b.shape[0]
        b_m = b.shape[1]
        b_k = b.shape[2]
    except IndexError:
        raise
    if a_m != b_n:
        raise ValueError("Cannot multiply matrices with different dimensions")
    elif a_k !=b_k:
        raise ValueError("Length of polynomial are different")

    for i in range(a_n):
        for j in range(b_m):
            element=np.array([0]*N)   
            for k in range(b_n):
                poly_val =  neg_poly_multiplication(np.poly1d(neg_a[i][k]), np.array(list(range(-N,0))), np.poly1d(b[k][j]) , np.array(list(range(N))), poly_mod, N)
                intermediate_element = np.polydiv(np.polyadd(np.array(element), np.array(poly_val)) , poly_mod)[1]  
                element = mod_2D(intermediate_element.c, mod)   
                if len(element) < N:
                    element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            if len(element) < N:
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            matmul.append(element)
    matmul = np.reshape(np.array(matmul), (a_n,b_m,N))        
    return matmul

def neg_poly_matmul_right(a, neg_b, mod, poly_mod, N=2048):
    matmul = []
    a = np.array(a)
    b = np.array(neg_b)
    try:
        a_n = a.shape[0]
        a_m = a.shape[1]
        a_k = a.shape[2]
        b_n = b.shape[0]
        b_m = b.shape[1]
        b_k = b.shape[2]
    except IndexError:
        raise
    if a_m != b_n:
        raise ValueError("Cannot multiply matrices with different dimensions")
    elif a_k !=b_k:
        raise ValueError("Length of polynomial are different")

    for i in range(a_n):
        for j in range(b_m):
            element=np.array([0]*N)   
            for k in range(b_n):
                poly_val =  neg_poly_multiplication_right(np.poly1d(a[i][k]),np.array(list(range(N))), np.poly1d(b[k][j]) ,np.array(list(range(-N,1))), poly_mod, N)
                intermediate_element = np.polydiv(np.polyadd(np.array(element), np.array(poly_val)) , poly_mod)[1]         
                element = mod_2D(intermediate_element.c, mod)
                if len(element) < N:
                    element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            if len(element) < N:
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            matmul.append(element)
    matmul = np.reshape(np.array(matmul), (a_n,b_m,N))        
    return matmul

def poly_matmul_no_mod(a,b, poly_mod, N=2048):
    matmul = []
    a = np.array(a)
    b = np.array(b)
    try:
        a_n = a.shape[0]
        a_m = a.shape[1]
        a_k = a.shape[2]
        b_n = b.shape[0]
        b_m = b.shape[1]
        b_k = b.shape[2]
    except IndexError:
        raise
    if a_m != b_n:
        raise ValueError("Cannot multiply matrices with different dimensions", a.shape, b.shape)
    elif a_k !=b_k:
        raise ValueError("Length of polynomial are different")

    for i in range(a_n):
        for j in range(b_m):
            element=0   
            for k in range(b_n):
                poly_mul = np.polydiv(np.polymul(a[i][k], b[k][j]) , poly_mod)[1].coefficients
                element = np.polydiv(np.polyadd(element, poly_mul) , poly_mod)[1].coefficients 
            # left padding of element with 0s so that the size of the elements are consistent 
            if len(element) < N:
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            matmul.append(element)
    matmul = np.reshape(np.array(matmul), (a_n,b_m,N))            
    return matmul

# uniformly generate a matrix from polynomial ring Z_mod[X]/X^N + 1
def gen_matrix_ring(n, m, mod, N):
    matrix = [random.randint(-(mod-1)//2, (mod-1)//2) for _ in range(n*m*N)]
    matrix = np.reshape(np.array(matrix), (n,m,N))
    return matrix

def gen_message(n,m,N,w):
    random_coefficients = np.zeros(n*m*N)
    indices = np.random.choice(n*m*N, w, replace=False)
    random_coefficients[indices] = 1
    matrix = np.reshape(random_coefficients,(n,m,N))
    return matrix

def gen_matrix_ring_with_norm(n, m, N, norm):
    matrix = np.random.randint(-norm,norm, size = (n,m,N))
    return np.array(matrix)

def bdlop_ckeygen(n=1, k=4, l=1, q1=2**24-75, q2=2**79-67, N=2048):
    a1_prime = gen_matrix_ring(n,k-n,q1,N)
    a2_prime = gen_matrix_ring(l, k-n-l,q2,N)
    identity_ring_n = np.dstack([np.identity(n)]*N)
    a1 = np.concatenate((identity_ring_n, a1_prime), axis=1)
    identity_ring_l = np.dstack([np.identity(l)]*N)
    zeros = np.dstack([np.zeros((l,n))]*N)
    a2 = np.concatenate((zeros,identity_ring_l, a2_prime), axis=1)
    a0 = np.concatenate((a1,a2), axis=0)
    return a0, a1, a2

def generate_random(n, m, N, beta):
    random_coefficients = np.random.randint(-beta+1,beta, size=(n,m,N))
    return random_coefficients

def bdlop_commit(A0, message, rand, q1_mod, q2_mod, poly_mod, n1=1,n2=1,m=4, N= 2048):
    A0 = np.array(A0)
    message = np.array(message)
    rand = np.array(rand)
    comm1 = poly_matmul(A0[:n1,:], rand, q1_mod, poly_mod, N=N)
    comm2_intermediate = poly_matmul(A0[n1:,:], rand, q2_mod, poly_mod, N=N)
    comm2 = poly_add_3D(comm2_intermediate, message ,q2_mod, poly_mod, N)
    comm = np.concatenate(([comm1[0]], comm2), axis=0)
    return comm 

def poly_add_2D(a,b,mod, poly_mod, N=2048):
    sum = []
    a = np.array(a)
    b= np.array(b)
    if a.shape != b.shape:
        raise ValueError("a and b are not of the same dimensions")
    for i in range(len(a)):
        intermediate_element = poly.polydiv(np.fmod(np.polyadd(a[i],b[i]), mod), poly_mod)[1]
        element = mod_2D(intermediate_element.c, mod)
        if len(element) < N:
            element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
        sum.append(element)
    sum = np.reshape(sum, a.shape)
    return sum

def poly_add_2D_no_mod(a,b, poly_mod, N=2048):
    sum = []
    a = np.array(a)
    b= np.array(b)
    if a.shape != b.shape:
        raise ValueError("a and b are not of the same dimensions")
    for i in range(len(a)):
        element = poly.polydiv(np.polyadd(a[i],b[i]), poly_mod)[1] 
        if len(element) < N:
            element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
        sum.append(element)
    sum = np.reshape(sum, a.shape)
    return sum

def poly_add_3D_no_mod(a, b, poly_mod, N=2048):
    sum = []
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError("a and b are not of the same dimensions", a.shape,b.shape)
    for i in range(len(a)):
        sum_temp = []
        for j in range(len(a[i])):
            element = np.polydiv(np.polyadd(np.poly1d(a[i][j]),np.poly1d(b[i][j])), poly_mod)[1].coefficients
            if len(element) < N:
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            sum_temp.append(element)
        sum.append(sum_temp)
    sum = np.reshape(sum, a.shape)
    return sum 

def poly_add_3D(a, b, mod, poly_mod, N=2048):
    sum = []
    a= np.array(a)
    b=np.array(b)
    if a.shape != b.shape:
        raise ValueError("a and b are not of the same dimensions:", a.shape, b.shape)
    for i in range(len(a)):
        sum_temp = []
        for j in range(len(a[i])):
            intermediate_element = np.polydiv(np.polyadd(a[i][j],b[i][j]), poly_mod)[1]
            element = mod_2D(intermediate_element.c, mod)
            if len(element) < N:
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            sum_temp.append(element)
        sum.append(sum_temp)
    sum = np.reshape(sum, a.shape)
    return sum 

def bdlop_combine(a,b,mod, poly_mod, N=2048):
    sum = []
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        assert ValueError("a and b are not of the same dimensions")
    for i in range(len(a)):
        element = poly_add_2D(a[i],b[i],mod, poly_mod,N)
        sum.append(element)
    return np.array(sum)

def bdlop_randomize(comm, A0, rand, n1, n2, m, q1_mod, q2_mod, poly_mod, N):
    A0 = np.array(A0)
    rand = np.array(rand)
    comm = np.array(comm)
    temp1 = poly_matmul(A0[:n1,:], rand, q1_mod, poly_mod, N=N)
    temp2 = poly_matmul(A0[n1:,:], rand, q2_mod, poly_mod, N=N)
    t1 = poly_add_3D([comm[0]],[temp1[0]],q1_mod, poly_mod,N)
    t2 = poly_add_3D([comm[1]],[temp2[0]],q2_mod, poly_mod,N)
    new_comm = np.concatenate((t1, t2), axis=0)
    return new_comm

def convert_R_to_S_k(r,n,m, N, d, k):
    s = []
    for i in range(n):
        row = []
        for j in range(m):
            for l in range(1,k+1):
                s_i = r[i][j][(l-1)*d:l*d]
                row.append(s_i)
        s.append(row)
    return np.array(s)

def cts_setup(n=1, k=4 ,l=1, l_prime=2,q1=4294966997,q2=67108837,N=512):
    A, a1, a2 = bdlop_ckeygen(n=n, k=k, l=l, q1=q1, q2=q2, N=N)
    D = gen_matrix_ring(l, l+l_prime, q2,N)
    kappa = 14
    gamma = 2**(28.3) 
    gamma_prime = 2**51.07
    alpha = 2**20.14

    return A,D, q1, q2, N, kappa, gamma, gamma_prime, alpha

def cts_commit(A, message, rand, G, N,poly_mod, q1,q2, n1, l=1, tau=7):
    if type(G) == int:
        dott = message
    else:
        dott = np.array(poly_matmul(message, np.array(G), q2, poly_mod,N))
    comm = bdlop_commit(A, dott, rand, q1, q2, poly_mod, n1=n1,n2=l,m=l*tau, N=N)
    return comm

def cts_randomize(comm, A, rand, n1, n2, m, q1_mod, q2_mod, poly_mod, N):
    new_comm = bdlop_randomize(comm, A, rand, n1, n2, m, q1_mod, q2_mod, poly_mod, N)
    return new_comm

def cts_combine(rand, rand_prime):
    rand = np.array(rand)
    rand_prime = np.array(rand_prime)
    if rand.shape != rand_prime.shape:
        raise ValueError("a and b are not of the same dimensions")
    return np.add(rand,rand_prime)

def cts_keygen(l, l_hat, tau, D, G,N,q1,q2,beta=1):
    polymod = np.poly1d([1] + [0]*(N-1) + [1])
    T = gen_matrix_ring_with_norm(l+l_hat,l*tau,N, beta)
    a0_temp = poly_matmul(D,T,q2, polymod,N)
    a0 = poly_add_3D(a0_temp,np.array(G),q2, polymod, N)
    B= gen_matrix_ring(l, l*tau, q2, N)
    u = gen_matrix_ring(l,1,q2,N)
    return a0, B, u, T

q2= 67108837
# Zq = sg.Zmod(q2)
# R = sg.PolynomialRing(Zq, 'x')
# x = R.gen()
# R_q = R.quotient(x**N + 1, 'X')
# X = R_q.gen()

def find_particular_solution(A, t, q):
    n, m = A.nrows(), A.ncols()
    M = A.augment(sg.identity_matrix(R, n)).stack(sg.Matrix(R, ncols=n + ncols, nrows=m - n))
    M = M.echelon_form()
    for row in M.rows():
        if A * row[:m] % q == t:
            return sg.vector(ZZ, row[:m])
    raise ValueError("No solution found") 

def rot(a,n,m,degree,poly_mod):
    new_a_ZZ= np.zeros((n*degree, m*degree))
    for i in range(n):
        for j in range(m):
            row = []
            for k in range(degree):
                x = [0]*degree
                x[degree-k-1] = 1
                element =  np.polydiv(np.polymul(np.poly1d(x), np.poly1d(a[i][j])),poly_mod)[1]
                if len(element) < degree-1:
                    element = np.pad(element, (degree-len(element)-1,0), 'constant', constant_values=0)
                row.append(np.flip(element))
            new_a_ZZ[i*degree:(i+1)*degree,j*degree:(j+1)*degree] = np.array(row)
    return new_a_ZZ

def unrot(a, n, m, degree):
    new_a_ZZ= []
    for i in range(n):
        for j in range(m):
            poly = a[i*degree,j*degree:j*degree+degree]
            new_a_ZZ.append(np.flip(poly))
    return new_a_ZZ    

def get_F_perp_trapdoor(T, w, G_perp, w_prime, R, N=512):
    
    F_perp = np.zeros((11*N, 9*N))

    F_perp[:3*N, :3*N] = sg.identity_matrix(3*N) - sg.matrix(T)*sg.matrix(w)
    F_perp[:3*N, 3*N:4*N] = -sg.matrix(T)*sg.matrix(G_perp)
    F_perp[3*N:5*N, :3*N] = sg.matrix(w)
    F_perp[3*N:5*N, N*3:N*4] = sg.matrix(G_perp)

    F_perp[5*N:7*N, 4*N:5*N] =  sg.matrix(G_perp)
    F_perp[5*N:7*N, 5*N:] = sg.matrix(w_prime)
    F_perp[7*N:, 4*N:5*N] = sg.matrix(-R)*G_perp
    F_perp[7*N:, 5*N:] = -sg.matrix(R)*sg.matrix(w_prime)  + sg.identity_matrix(4*N)
    F_perp = sg.matrix(F_perp)

    return F_perp 

def sample_preimage(F_perp, F, u, N, alpha=2, q2=q2):
    u_q2 = sg.matrix(sg.Integers(q2), u)
    F_q2 = sg.matrix(sg.Integers(q2),F)
    x0 = F_q2.solve_right(u_q2)
    x0 = sg.matrix(x0.lift())
    v = np.random.randint(0, 10, size=(11*N, N))
    while np.all(np.rint(F*(sg.matrix(v+x0)))%q2 == np.rint(u)%q2) == False:
        for i in range(x0.ncols()):
            center_unchanged = sg.vector(-x0.column(i))
            x_col = DiscreteGaussianDistributionLatticeSampler(sg.transpose(F_perp), alpha, c=center_unchanged)
            tt = x_col()
            v[:,i] = np.array(tt)
        if np.all(np.rint(F*(sg.matrix(v+x0)))%q2 == np.rint(u)%q2) == False:
            print(np.rint(u)%q2)

    return sg.matrix(x0 + v)

def generate_f_comm(B,c2,D,A0,A2,q2, poly_mod,N):
    B_c2 = poly_add_3D(B,c2,q2,poly_mod,N)
    F_comm = np.concatenate((D,A0,B_c2,A2), axis=1)
    return F_comm

def cts_sign(D, A0, B, comm, A2, u, N, width, height, q2, bound, poly_mod):
    c2 = comm[1:,:]

    F_comm = generate_f_comm(B,c2,D,A0,A2,q2, poly_mod,N)

    sampler = DiscreteGaussianDistributionIntegerSampler(sigma=np.ceil(q2/2))
    sig_comm = []
    for i in range(width*N*height):
        sig_comm.append(int(sampler()))
    within_bound = False
    while not within_bound:
        calculated_bound = np.linalg.norm(sig_comm)
        if calculated_bound <= bound:
            within_bound = True
        else:
            for i in range(width*N*height):
                sig_comm = []
                sig_comm.append(int(sampler()))
    sig_comm = np.reshape(np.array(sig_comm), (width, height, N))
    u = poly_matmul(F_comm, sig_comm, q2, poly_mod, N)
    return sig_comm, u

# def cts_sign_(G_perp,F,T,u,alpha,N,poly_mod, q2, R, w, w_prime):
#     F_perp = get_F_perp_trapdoor(T, w, G_perp, w_prime, R,N=N)
#     sig_comm = sample_preimage(F_perp, F, u, N, alpha,q2=q2)

#     u_check = np.all(np.rint(F*(sig_comm))%q2 == np.rint(u)%q2)

#     if u_check != True:
#         exit()
#     return sig_comm, u_check

def pad_T(T, rand, n,m):
    full_matrix  = np.zeros((n,n))
    full_matrix  = np.zeros((9*512,9*512))

    full_matrix[:1536, :m] = T
    full_matrix[:1536, 1536:5*512] = rand*-1
    full_matrix[2048:2560, 1536:5*512] = np.identity(512)

    return full_matrix

def get_trap(T, R,N):
    full_matrix  = np.zeros((9*N,9*N))
    full_matrix[:N*3, :N*3] =  np.identity(N*3)
    full_matrix[:N*3, N*3:N*4] = T*-1

    full_matrix[N*3:N*4, N*3:N*4] = np.identity(N)
    full_matrix[N*4:N*5, N*4:N*5] = np.identity(N)
    full_matrix[N*5:, 4*N:N*5] = -1*R
    full_matrix[N*5:, N*5:] = np.identity(N*4)
    return full_matrix

def cts_transfer(D, A, A0, B,n, k,l, l_hat, tau, sig_comm, message, rand, rand_prime, N,poly_mod, q1,q2,n1,n2,m, u, gamma_prime,G):
    split_sig = np.split(sig_comm, [(1+tau)*l+l_hat,((1+tau)*l+l_hat) + l*tau])
    s1 = split_sig[0] 
    s2 = split_sig[1]
    s3 = split_sig[2]
    new_comm = cts_commit(A,message,rand,G,N, poly_mod, q1,q2, n1,l,tau=tau) #check params(A, message, rand,N,poly_mod, q1,q2, n1, l=1)
    randomized_comm = cts_randomize(new_comm,A,rand_prime,n1,n2, m, q1,q2,poly_mod,N)
    split_comm = np.split(randomized_comm, [n])
    c2 = split_comm[1]
    rand_tilde = cts_combine(rand,rand_prime)
    split_rand = np.split(rand_tilde,[n])
    last_row_temp = poly_matmul_no_mod(split_rand[1],s2,poly_mod,N)
    last_row = poly_add_3D_no_mod(s3,(-1)*last_row_temp,poly_mod,N)
    new_sig_comm = np.concatenate((s1, s2, last_row), axis=0)
    f_comm_prime = generate_f_comm(B,c2,D,A0,A[1:,l:],q2, poly_mod,N)
    return nizk_prove(f_comm_prime, u, new_sig_comm, gamma_prime, q2, poly_mod, N)

def cts_verify(u,A0,A, B,D,n,l, comm, sig, sig_type, q2, poly_mod,N):
    split_comm = np.split(comm, [n])
    c2 = split_comm[1]
    f_comm = generate_f_comm(B,c2,D,A0,A[1:,l:],q2, poly_mod,N)

    if sig_type == "proof":
        if nizk_prove(sig, f_comm, u) ==  True:
            return True
        return False
    elif sig_type == "vector":
        if np.all(poly_matmul(f_comm,sig,q2,poly_mod,N) == u):
            return True
        else: 
            return False
    return False

def AC_setup():
    A,D, q1, q2, N, kappa, gamma, gamma_prime, alpha  = cts_setup()
    # params = nizk_setup()
    l = 1
    l_hat = 2
    tau= 2
    g = []
    w = 22
    for i in range(tau):
        val = q2**(1/tau)**i
        g.append(val)
    G = np.zeros((1,tau,N))
    G[0][0][-1] = 1
    G[0][1][-1] = np.rint(np.sqrt(q2))

    beta = 2 
    #user setup
    d = 128
    N = 512
    usk = np.zeros((1,1,N))
    indices = np.random.choice(N, w, replace=False)
    for i in indices:
        usk[0][0][i] = 1
    #organizations setup
    opk_a0, opk_B, opk_u, osk = cts_keygen(l,l_hat,tau,D,G,N,q1,q2,beta)
    return A, D, q1, q2, N, kappa, gamma, gamma_prime, alpha, opk_a0, opk_B, opk_u, osk,usk

def AC_registration(n,m,N,A,poly_mod,q1,q2,n1,l,usk, G,tau,beta=2):
    rand = generate_random(n,m,N,beta)
    comm = cts_commit(A, usk, rand,G,N,poly_mod, q1,q2, n1, l=l, tau=tau)
    return comm,rand

def AC_issue(D,A2, A0,B,u,comm,N,poly_mod,q2,bound):
    sigma, u = cts_sign(D,A0,B,comm, A2,u,N,10,1,q2,bound,poly_mod)
    return sigma, u

def AC_prove(D,A,A0,B,n,k,l,l_hat,tau,message,rand,rand_prime,N,poly_mod,q1,q2,n1,n2,m,comm,credential,u,gamma_prime,G,beta=2):
    rand_prime = generate_random(n,m,N,beta)
    randomized_comm = cts_randomize(comm, A, rand_prime, n1, n2, m, q1, q2, poly_mod, N)
    transferred_sig = cts_transfer(D, A, A0, B,n, k,l, l_hat, tau, credential, message, rand, rand_prime, N,poly_mod, q1,q2,n1,n2,m, u, gamma_prime,G)
  
    #TODO: nizk for well_formedness of comm_prime
    pi_prime = 0
    return transferred_sig, (randomized_comm,pi_prime)

def AC_verify(u,A0,A,B,D,n,l, comm, sig, sig_type, q2, poly_mod,N, alpha):
    result = cts_verify(u,A0,A,B,D,n,l, comm, sig, sig_type, q2, poly_mod,N)
    split_comm = np.split(comm, [n])
    c2 = split_comm[1]
    f_comm = generate_f_comm(B,c2,D,A0,A[1:,l:],q2, poly_mod,N)

    gamma_prime = (math.sqrt(k-n)+1)*N*alpha*math.sqrt(2) + alpha*math.sqrt(8)*N 
    result2 = nizk_prove(f_comm, u, sig, gamma_prime, q1, poly_mod, N)
    if result == True and result2 == True:
        return True
    else: 
        return False

def ring_transpose(a):
    transpose = np.transpose(a, axes=(1,0,2))
    return transpose


q1 = 2**24-75
q2 = 2**79-67
d = 512
q = 1437757
n = 1
l = 1
k = 4
tau = 7
t = 4
w = 22
#M = 2**128
q_pke = 1437757
n_pke = 2
m_pke = 117
k_pke = 113
root_hermite = 1.003433
#Bit_sec_pke = 144.25/130.91
q_LNP = 2**103

gamma_1 = 17
gamma_2 = 1.2
gamma_e = 2.5
gamma_d = 12
n_LNP = 23
k_LNP = 1
n_LNP = 2
m1 = 581
l_LNP = 0
m2 = 18
v_LNP = 1
gamma_LNP = 2**35.69
D_LNP = 27

def enc_key_gen(poly_mod, d = 512, q = 1437757, n = 2, m = 117, k = 113):
    A = gen_matrix_ring(n,m,q,d)
    beta = 2
    S = np.random.randint(-beta,beta+1, size=(k,n,d))
    E = np.random.randint(-beta,beta+1,size=(k,m,d))
    B_temp = poly_matmul(S,A, q, poly_mod, N=d)
    B = mod_3D(len(B_temp), B_temp + 3*E,q)
    B = B_temp + 3*E
    B = mod_3D(len(B),B, q)
    return A,B,S

def enc_encrypt(A, B, miu, poly_mod, m = 117, d = 512, q = 1437757):
    beta = 2
    r_pke = np.random.randint(-beta,beta+1,size=(m,1,d))
    c0 = poly_matmul(A.astype(float), r_pke.astype(float), q, poly_mod, N=d)
    intermediate =poly_matmul(B, r_pke, q, poly_mod, N=d)+ ring_transpose(miu)
    c1 = mod_3D(len(intermediate), intermediate,q)
    return r_pke , c0, c1

def enc_decrypt(A,B,S,c0,c1, q,d=512):
    poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    miu_prime = np.fmod(np.array(c1) + np.array(-1*(poly_matmul(S, c0, q, poly_mod, d))), q)
    intermediate = np.array(c1) + np.array(-1*(poly_matmul(S, c0, q, poly_mod, d)))
    miu_prime = mod_3D(len(intermediate), intermediate, q)
    return miu_prime


def rejection_sampling_1(z,v,s,M=1.1):
    u = np.random.rand()
    a = -2* np.sum(np.inner(ring_transpose(z),ring_transpose(v)))
    b = LA.norm(v)**2
    d = (a+b)/(2*s**2)
    if u > ((1/M) * np.exp(d)):
        return 0 #reject
    else:
        return 1

def rejection_sampling_2(z,v,s,M=6):
    if np.inner(z,v).all() < 0:
        return 1
    return rejection_sampling_1(z,v,s,M)

def rejection_sampling_0_int(z,v,s,q,M=6):
    u = np.random.rand()
    z = np.array(z)
    norm = LA.norm(v)
    inner_prod = np.dot(z,v)
    if u > (1/(M*math.exp((-1*norm**2)/2*s**2)*np.cosh((inner_prod/s**2)))):
        return 1
    else:
        return 0   

def rejection_sampling_0(z,v,s,q,M=6):
    u = np.random.rand()
    z = np.array(z)
    norm = LA.norm(v)%q
    inner_prod = np.fmod(np.sum(z*v),q)
    if u > (1/(M*math.exp((-1*norm**2)/2*s**2)*np.cosh((inner_prod/s**2)))):
        return 1
    else:
        return 0  

def apply_automorphism_neg_1(sigma, v, dim, exp):
    if len(v) < dim:
        v = np.pad(v, (dim-len(v),0), 'constant', constant_values=0)
    if exp % 2 == 0:
        return v
    new_v = []
    for i in range(len(v)):
        new_v.append(np.float_power(v[i],sigma))
    new_v.append(v[-1])
    return new_v

def apply_automorphism_once(automorphism, v, d):
    if not isinstance(automorphism, int):
        return -1 
    new_v = []

    temp_v = copy.deepcopy(v)
    for j in range(len(temp_v)):
        for p in range(len(temp_v[j])):
            temp_v[j][p][:-1] = np.array([num*automorphism for num in temp_v[j][p][:-1]])
            element = temp_v
        new_v.append(element)
    new_v = np.reshape(np.array(new_v), (len(v),1,d))
    return new_v

def apply_automorphism(automorphism, v, k, d):
    if not isinstance(automorphism, int):
        return -1 
    new_v = []
    for i in range(k):
        if i % 2 == 0:
            element = v
        else:
            temp_v = copy.deepcopy(v)
            for j in range(len(temp_v)):
                for p in range(len(temp_v[j])):
                    temp_v[j][p][:-1] = np.array([num*automorphism for num in temp_v[j][p][:-1]])
                    element = temp_v
        new_v.append(element)
    new_v = np.reshape(np.array(new_v), (k*len(v),1,d))
    return new_v

def fig_6_prove_step_1(s1, s2, sigma1, sigma2, poly_mod, nu=23, k=4, l = 0, sigma=-1, d=512, m1=581, m2=18, gamma1= 17, n=2, gamma2=1.2, q=(2**24 - 75)*(2**79 - 67), v=1):

    s = apply_automorphism(sigma, s1, k, d)

    A1 = gen_matrix_ring(n,m1,q,d)
    A2 = gen_matrix_ring(n,m2,q,d)
    B_mat = gen_matrix_ring(l, m2, q,d)
    
    y1 = sample_discrete_gaussian_distribution(m1, q, d, sigma1)
    y2 = sample_discrete_gaussian_distribution(m2, q, d, sigma2)

    w1 = poly_matmul(A1, y1, q, poly_mod, d)  
    w2 = poly_matmul(A2, y2, q, poly_mod, d)
    w = np.fmod(w1 + w2,q)
    w = mod_3D(len(w1+w2),w1 + w2,q)

    y = apply_automorphism(-1, y1, k, d)

    R2 = gen_matrix_ring(k*(m1+l), k*(m1+l),q,d)

    sR2 = poly_matmul(ring_transpose(s),R2,q,poly_mod,d)

    R2s = poly_matmul(R2,s, q,poly_mod, d)
  
    sR2y = poly_matmul(sR2 ,y, q,poly_mod, d)    

    yR2s = poly_matmul(ring_transpose(y), R2s, q,poly_mod, d)

    r1 = gen_matrix_ring(k*(m1+l),1,q,d)
    r1y = poly_matmul(ring_transpose(r1),y,q,poly_mod,d)
    g1 = np.fmod(sR2y + yR2s + r1y,q)
    b_vec= gen_matrix_ring(m2,1,q, d)
    t = np.fmod(poly_matmul(ring_transpose(b_vec),s2, q, poly_mod, d) + g1, q)
    
    yR2 = poly_matmul(ring_transpose(y),R2, q,poly_mod, d)
    
    yR2y = poly_matmul(yR2, y, q,poly_mod, d) 

    by2 = poly_matmul(ring_transpose(b_vec),y2,q,poly_mod,d)
    v = np.fmod(yR2y+ by2 ,q)
    return b_vec, s,w,t,v,y1,y2,A1,A2, R2, r1

def sample_challenge(d, k,q, n=23):
    poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    c = [random.randrange(-k,k+1) for i in range(d)]
    exp = [i**k for i in c]
    tmp_val = LA.norm(neg_poly_matmul([[exp]], [[exp]] , q, poly_mod,d )[0][0],1)
    val = tmp_val**(1/(2*k))
    while val > n:
        c = [random.randrange(-k, k+1) for i in range(d)]
        exp = [i**k for i in c]
        tmp = apply_automorphism_neg_1(-1, exp,1,1)
        tmp_val = LA.norm(poly_matmul([[tmp]],[[exp]],q, poly_mod,d )[0][0],1)
        val = tmp_val**(1/(2*k))
    return c

def sample_challenge_v2(d, kappa, q, nu=23):
    poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    val = nu + 1
    c = []
    while val > nu:
        half_c = [random.randrange(-kappa,kappa+1) for i in range(int(d/2))]
        other_half_c = copy.deepcopy(half_c)
        c = [num*-1 for num in np.flip(other_half_c[:-1])] + [0] + half_c
        sigma_c = [num*-1 for num in c[:-1]]
        sigma_c.append(c[-1])
        c_k = [i**k for i in c]
        sigma_c_k = [i**k for i in sigma_c]
        product = poly_matmul([[sigma_c_k]],[[c_k]],q, poly_mod,d)
        tmp_val = np.sum(np.abs(product[0][0]))
        val = tmp_val**(1/(2*k))
    return [[c]]

    while val > n:
        c = [random.randrange(-k, k+1) for i in range(d)]
        exp = [i**k for i in c]
        tmp = apply_automorphism_neg_1(-1, exp,1,1)
        tmp_val = LA.norm(poly_matmul([[tmp]],[[exp]],q, poly_mod,d )[0][0],1)
        val = tmp_val**(1/(2*k))
    return c

def get_challenge(d,k,q,n):
    c_1 = sample_challenge(d,k,q,n)
    c_2 = sample_challenge(d,k,q,n)
    poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    for i in range(len(c_1)):
        if c_1[i] != c_2[i]:
            return poly_add_2D([c_1], [np.array(c_2)*-1], q, poly_mod, d)
    return False

def fig_6_prove_step_2(c,s1,s2,y1,y2, sigma1, sigma2, q, d, poly_mod):

    z1 = np.fmod(ring_transpose(np.array(poly_matmul(np.array(c),ring_transpose(np.array(s1)),q,poly_mod,d))) + np.array(y1), q)
    z2 = np.fmod(ring_transpose(np.array(poly_matmul(np.array(c),ring_transpose(np.array(s2)),q,poly_mod,d))) + np.array(y2),q)

    rej1_res = rejection_sampling_1(z1, ring_transpose(poly_matmul(np.array(c),ring_transpose(np.array(s1)),q,poly_mod,d)), sigma1)
    rej2_res = rejection_sampling_2(z2, ring_transpose(poly_matmul(np.array(c), ring_transpose(np.array(s2)),q,poly_mod,d)),sigma2)
    if rej1_res == 1 or rej2_res ==1:
        return -1, -1 
    else:
        return z1, z2

def check_zkp_1(A1,A2,w,t,v,c,z1,z2,std1,std2,m1,m2,l,d,t1,t2,b,r0,r1, R2,m,s2,q,poly_mod,k=4):

    z = apply_automorphism(-1, z1, k, d)
    f = np.fmod(poly_matmul(c,t, q, poly_mod, d) + poly_matmul(ring_transpose(b), z2, q, poly_mod,d)*-1, q)

    new_list = [LA.norm(poly[0])**2 for poly in z1]
    norm_z1 = math.sqrt(np.sum(new_list))

    new_list = [LA.norm(poly[0])**2 for poly in z2]
    norm_z2 = math.sqrt(np.sum(new_list))

    if norm_z1 > std1*math.sqrt(2*m1*d):
        return False
    if norm_z2  > std2*math.sqrt(2*m2*d):
        return False
    lhs = np.fmod(poly_matmul(A1,z1, q,poly_mod, d)+ poly_matmul(A2,z2, q, poly_mod,d), q)
    rhs = np.fmod(w + ring_transpose(poly_matmul(np.array(c),ring_transpose(np.array(t1)),q,poly_mod,d)), q)

    for i in range(len(lhs)):
        for j in range(len(lhs[0])):
            for k in range(len(lhs[0][0])):
                if lhs[i][j][k]%q != rhs[i][j][k] %q:
                    return False
    zR2 = poly_matmul(ring_transpose(z),R2, q, poly_mod,d)
    zR2z = poly_matmul(zR2, z,q, poly_mod,d)
    cR1 =poly_matmul(c,ring_transpose(r1), q, poly_mod,d)
    cR1z =poly_matmul(cR1,z,q, poly_mod,d) 
    step5=np.fmod(zR2z+ cR1z, q)
    cc=poly_matmul(c,c,q,poly_mod,d)
    ccr0=poly_matmul(cc,r0, q, poly_mod,d)
    step7 =np.fmod(ccr0 + -1*f, q)
    lhs2= np.fmod(step5+step7, q)  
    for i in range(len(lhs2)):
        for j in range(len(lhs2[0])):
            for k in range(len(lhs2[0][0])):
                if lhs2[i][j][k]%q != v[i][j][k]%q:
                    return False
    return True
    
def fig_7_verifier(q,d,N):
    mus = [np.random.randint(-q+1,q, size = (1,d)) for i in range(N)]
    return mus

def fig_7_prover(N, mus,fs,q,poly_mod,d):
    f = poly_matmul([[mus[0][0]]],[[fs[0][0]]],q,poly_mod,d)
    for i in range(1,N):
        f = np.fmod(f+ poly_matmul([[mus[i][0]]],[[fs[0][i]]],q,poly_mod,d),q)
    return f

def convert_R_to_S_k(r,n,m, N, d, k):
    s = []
    for i in range(n):
        row = []
        for j in range(m):
            for l in range(0,k):
                poly = []
                for o in range(0,N,k):
                    s_i = r[i][j][l+o]
                    poly.append(s_i)
                row.append(poly)
        s.append(row)
    return np.array(s)

def fig_8_prover_1(sigma, s1, lmda, q,d, s2, poly_mod, k = 4,m1 = 581,m2 = 18, l = 1):
    s = apply_automorphism(sigma, s1, k, d)
    g = gen_matrix_ring(lmda,1, q,d)
    for i in range(lmda):
        g[i][0][-1] = 0
    Bg = gen_matrix_ring(lmda, m2, q, d)
    tg = poly_add_3D(poly_matmul(Bg, s2, q, poly_mod, d), g, q, poly_mod, d)
    return s, g, tg

def fig_8_verifier_1(q, lmda, M=1):
    gamma = [random.randrange(-q+1, q) for i in range(lmda*M)]
    gamma = np.reshape(gamma,(lmda,M))
    return gamma

def fig_8_prover_2(g, F1, gamma, lmda, d,q, M=1):

    h = []

    for i in range(lmda):
        sum = 0
        sum = np.sum([gamma[i][j]*np.array(F1) for j in range(M)])

        add_tmp = np.fmod([g[i]]+ sum, q)
        h.append(add_tmp[0])
    return h

def fig_8_verifier_2(h):
    h = np.array(h)
    for i in range(len(h)):
        if h[i][0][-1] != 0:
            return False
    return True

def sample_discrete_gaussian_distribution(m, q, d, sigma):
    R = PolynomialRing(Integers(q), "X")
    X = R.gen()
    S = R.quotient(X**d + 1)
    y_sampler = DiscreteGaussianDistributionPolynomialSampler(S,d,sigma)
    y = []
    for i in range(m):
        y.append([list(map(int, y_sampler().list()))]) 
    return np.array(y)

def fig_10_prove_1(s2, std_d, std_e, poly_mod, m2 = 18,N = 2048,k = 4,d = 512,m_pke = 117, n_pke = 2, p_i = 117, v_e = 8, k_bin = 1 , y_dim = int(256/d), q=2**103 + 125):
    bin_b_d = np.zeros((1,1,d))
    bin_b_d[0][0][-1] = np.random.choice([-1,1], 1)
    bin_b_e = np.zeros((1,1,d))
    bin_b_e[0][0][-1] = np.random.choice([-1,1], 1)

    y_d = sample_discrete_gaussian_distribution(y_dim, q, d, std_d)
    y_e = sample_discrete_gaussian_distribution(y_dim, q, d, std_e)

    mat_B_d = gen_matrix_ring(y_dim, m2, q, d)
    mat_B_e = gen_matrix_ring(y_dim, m2, q, d)

    t_d = poly_add_3D(poly_matmul(mat_B_d, s2, q, poly_mod, d), np.array(y_d), q, poly_mod, d)

    t_e = poly_add_3D(poly_matmul(mat_B_e, s2, q, poly_mod, d), np.array(y_e), q, poly_mod, d)

    vec_b_d = gen_matrix_ring(m2, 1,q, d)
    vec_b_e = gen_matrix_ring(m2, 1,q, d)

    t_d_prime = poly_matmul(ring_transpose(vec_b_d), s2, q, poly_mod, d) + bin_b_d 
    t_e_prime = poly_matmul(ring_transpose(vec_b_e), s2, q, poly_mod, d) + bin_b_e
    return t_d, t_d_prime, t_e, t_e_prime, bin_b_d, bin_b_e, y_d, y_e

def fig_10_verifier_1(cd, ce, dim=512):
    R_d = np.random.choice([-1,0,1], p=[0.25,0.5,0.25] , size=(dim,cd))
    R_e = np.random.choice([-1,0,1], p=[0.25,0.5,0.25], size=(dim,ce))
    return R_d, R_e

def T_map_computation(a,b, k,d, poly_mod):
    res = 0 
    a_prime =[]
    b_prime =[] 
    a_exp = []
    b_exp = []
    sum = 0
    for i in range(k-1):
        for j in range(d-1):
            a_prime.append(a[i*d+j])
            a_exp.append(-j)
            b_prime.append(b[i*d+j])
            b_exp.append(j)
        res = neg_poly_multiplication(a_prime, a_exp, b_prime, b_exp, poly_mod,d)
        sum = sum + res
    return sum

def map_ring_n_to_d(element_N, N, k):
    s = []
    if len(element_N) < N:
        element_N = np.pad(element_N, (N-len(element_N),0), 'constant', constant_values=0)

    for i in range(k):
        s.append([element_N[j] for j in range(i, len(element_N), k)])
    return s

def fig_10_prover_2(e_d, e_e, b_d,R_d,y_e,y_d,b_e,R_e,s_d, s_e,d,q, y_dim=1):
    bdRded = np.matmul(b_d[0][0][-1]*R_d, e_d) 
    beReee = np.matmul(b_e[0][0][-1]*R_e, e_e)

    new_ze = beReee + y_e 
    new_zd = bdRded + y_d 

    if rejection_sampling_0_int(new_zd, bdRded, s_d,q) !=0:
        return False, None, None #,None, None, None, None, None
    
    if rejection_sampling_0_int(new_ze, beReee, s_e,q) != 0: 
        return False, None, None #None, None, None, None, None

    return True, np.array(new_zd), np.array(new_ze) #, s_star, sigma, phi, psi

def accept_fig_10(z_d, z_e, s_d, t_bound, s_e):

    new_d = [np.max(z_d[i]) for i in range(z_d.shape[0])]
    new_e = [np.max(z_e[i]) for i in range(z_e.shape[0])]

    if LA.norm(new_d, np.inf) <= 14*s_d and LA.norm(new_e) <=t_bound*math.sqrt(256)*s_e:
        return True
    else:
        return False

def nizk_prove(f_comm, u, sig, gamma_prime, mod, poly_mod, N):
    if np.all(poly_matmul(f_comm, sig, mod, poly_mod, N) == np.array(u)):
        if LA.norm(sig,2) <= gamma_prime:
            return True
    return False
