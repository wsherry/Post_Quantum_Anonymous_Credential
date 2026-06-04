import numpy as np
from numpy.polynomial import polynomial as poly
from numpy import linalg as LA
import random
import copy
from sympy import *
import sage.all as sg
from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionLatticeSampler
from sage.stats.distributions.discrete_gaussian_polynomial import DiscreteGaussianDistributionPolynomialSampler
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
# import oqs
# from operator import add
# print("NON INTEGER ISSUE")
# print(np.fmod(np.array([1, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 2, 2, 1, 1]), 604462909807314587353021))

def custom_mod(a,b):
    func_vec = np.vectorize(lambda x: x % b)
    res = func_vec(a)
    # return res
    return res if not res else res-b if a<0 else res

def mod_2D(x, q):
    r = (x + np.floor(q/2)) % q - np.floor(q/2) 
    r[r <= -np.floor(q/2)] += q
    return r

def mod_3D(rows, x, q):
    for i in range(rows):
        x[i] = mod_2D(x[i],q)
    return x

# def custom_fmod_3D(a, n, m, d, q):
#     print(a,n,m,d,q)
#     for i in range(n):
#         for j in range(m):
#             for k in range(d):
#                 a[i][j][k] = custom_mod(a[i][j][k],q)
#     return a


# def custom_fmod_2D(a, n,m, q):
#     # print(a,n,m,d,q)
#     for i in range(n):
#         for j in range(m):
#             a[i][j]= custom_mod(a[i][j],q)
#     return a

# mod = 100
# mat = [[[5,4,50]],[[-101, -205,-99]],[[0,5,-1]]]
# mod2 = [[[ 100, 100, 100]],[[ 100,  100, 100]],[[ 100, 100, 100]]]
# print(mod_3D(3, mat, mod))
# print(np.fmod(mat, mod2))
# print(mod_2D(np.array([range(-102,102)]), mod))
# print(np.fmod(np.array([range(-102,102)]), mod))
# exit()
# print(symmetric_mod_2D(np.array([-52, 52, 50, -50,51,3,-902, 101,-101]), mod))
# mod = 100
# print(symmetric_mod_2D(np.array([-52, 52, 50, -50,-51,51,3,-902, 101,-101]), mod))

# x = -5 % 100
# y = 95 % 100
# print(x,y)
# exit()

# multiply a 2D matrix/vector where its elements are polynomials

# def matmul(a,b, mod ,N):
#     matmul = []
#     a = np.array(a)
#     b = np.array(b)
#     print("MATMUL:",a.shape, b.shape)
#     try:
#         a_n = a.shape[0]
#         a_m = a.shape[1]
#         a_k = a.shape[2]
#         b_n = b.shape[0]
#         b_m = b.shape[1]
#         b_k = b.shape[2]
#     except IndexError:
#         raise
#     if a_m != b_n:
#         raise ValueError("Cannot multiply matrices with different dimensions")
#     elif a_k !=b_k:
#         raise ValueError("Length of polynomial are different")

#     for i in range(a_n):
#         for j in range(b_m):
#             element=np.array([0]*N)   
#             for k in range(b_n):
#                 val =  a[i][k]*b[k][j]
#                 # print(p1)
#                 # poly_mul = [custom_mod(num, mod) for num in p1]
#                 # print("POLYVAL",poly_val.c)
#                 # element = np.poly1d(np.polydiv(np.polyadd(np.array(element), np.array(poly_val)) , poly_mod)[1]).coefficients
#                 intermediate_element = element + val
#                 # print(intermediate_element)
#                 # element = [custom_mod(num, mod) for num in intermediate_element]
#                 element = np.fmod(intermediate_element,mod)
#                 # print("SUM", element)
#                 # print(poly_mul)
#                 # poly_mul_2 = poly.polydiv(poly.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j]))%mod , poly_mod)[1] % mod
#                 # print(np.array(poly_mul).shape)
#                 # if len(poly_mul) < N:
#                 #     poly_mul = np.pad(poly_mul, (N-len(poly_mul),0), 'constant', constant_values=0)  
#                 if len(element) < N:
#                     element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
#             if len(element) < N:
#                 element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
#             # print("ELEMENT",element)
#             matmul.append(element)

#     print("matmul", np.array(matmul).shape, a_n,b_m,N)
#     matmul = np.reshape(np.array(matmul), (a_n,b_m,N))        
#     return matmul


def alt_poly_matmul(A,B,mod, poly_mod, N=2048):
    rows_A = A.shape[0]
    cols_A = A.shape[1]
    rows_B = B.shape[0]
    cols_B = B.shape[1]
    # print(rows_A, cols_B, cols_A, rows_B)
    result = np.empty((rows_A, cols_B), dtype=object)
    # print(result.shape, cols_A)
    
    for i in range(rows_A):
        for j in range(cols_B):
            # Multiply and sum polynomials like in normal matrix multiplication
            # print(np.polymul(A[i][k], B[k][j]))
            element = [np.pad(np.polymul(A[i][k], B[k][j]), (N-np.polymul(A[i][k], B[k][j]).shape[0],0), 'constant', constant_values=0) if N-np.polymul(A[i][k], B[k][j]).shape[0] >0 else  np.polymul(A[i][k], B[k][j]) for k in range(cols_A)]
            # print(element)
            # print(element.shape)
            # element = np.sum(elements)
            result[i][j] = np.polydiv(element ,poly_mod)[1]
    res = mod_2D(result, mod)
    # print(res)
    # print()
    # return np.fmod(result, mod)
    return res


# import numpy as np
def poly_matmul(a,b, mod, poly_mod, N=2048):
    matmul = []
    a = np.array(a)
    b = np.array(b)
    # print("POLY_MATMUL:",a.shape, b.shape)
    try:
        a_n = a.shape[0]
        a_m = a.shape[1]
        a_k = a.shape[2]
        b_n = b.shape[0]
        b_m = b.shape[1]
        b_k = b.shape[2]
        # n1 = len(a) 
        # m = len(a[0])
        # n2 = len(b[0])
        # print(n1,m,n2)
        # print(a_n,b_m,b_n)
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
                # print("----------A and B ",a[i][k],b[k][j])
                # hhh = poly.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j]))
                # print(poly.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j])))
                # print(poly.polydiv(np.fmod(np.array(poly.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j])),mod) , poly_mod)[1])
                # part_1 = poly.polymul(a[i][k], b[k][j])                 
                # print(np.poly1d(a[i][k]).c, np.poly1d(b[k][j]).c)
                # print(poly.polydiv(part_1,np.poly1d(poly_mod)))
                # part_1 = poly.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j]))
                # p1 =  poly.polydiv(np.poly1d(poly.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j]))) , np.poly1d(poly_mod))
                # print(np.poly1d(poly_mod))
                # print(p1)
                poly_val =  np.polydiv(np.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j])) ,poly_mod)[1]
                if np.array(poly_val).shape[0]>N:
                    print(np.array(poly_val).shape)
                    print("POLYVAL greater than N", poly_val)
                # print(p1)
                # poly_mul = [custom_mod(num, mod) for num in p1]
                # print("POLYVAL",poly_val.c)
                # element = np.poly1d(np.polydiv(np.polyadd(np.array(element), np.array(poly_val)) , poly_mod)[1]).coefficients
                intermediate_element = np.polydiv(np.polyadd(np.array(element), np.array(poly_val)) , poly_mod)[1]
                # print(type(intermediate_element.coeffs))
                # exit()
                # element = [custom_mod(num, mod) for num in intermediate_element] 
                element = mod_2D(intermediate_element.c, mod)
                # print("ELEMENT VS ELEMENT2")
                # print(intermediate_element.c)
                # print(element)
                # print()
                # print(element2)
                # exit()
                # print(intermediate_element)
                # print(type(intermediate_element.coeffs.astype(int)), intermediate_element.coeffs.astype(int))
                # print([dtype(num) for num in intermediate_element], type(mod))
                # print(intermediate_element.coeffs.dtype)
                # element = [np.fmod(num, mod) for num in intermediate_element]
                # element = np.fmod(np.array(intermediate_element.coeffs), mod)
                # np.round(org).astype(int)
                # print(type(element))
                # print("SUM", element)
                # print(poly_mul)
                # poly_mul_2 = poly.polydiv(poly.polymul(np.poly1d(a[i][k]), np.poly1d(b[k][j]))%mod , poly_mod)[1] % mod
                # print(np.array(poly_mul).shape)
                # if len(poly_mul) < N:
                #     poly_mul = np.pad(poly_mul, (N-len(poly_mul),0), 'constant', constant_values=0)  
                if len(element) < N:
                    element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
                # print("ELEMENT", element)
                # print("ISSUE")
                # print(element, poly_mul, poly_mod)
                # print(np.array(element).shape, np.array(poly_mul).shape)
                # print("-------------")

                # print(element+poly_mul)
                # poly_mul = poly.polydiv(poly.polymul(a[i][k], b[k][j]) , poly_mod)[1] 
                # element = poly.polydiv(poly.polyadd(element, poly_mul) , poly_mod)[1] 
                # element = [custom_mod(num, mod) for num in poly.polydiv([custom_mod(num, mod) for num in np.add(element,poly_mul)], poly_mod)[1]]

                # element = np.fmod(poly.polydiv(np.fmod(element+poly_mul, mod), poly_mod)[1], mod)
                # print("sum", element)
            # print(len(element))              

            # left padding of element with 0s so that the size of the elements are consistent 
            if len(element) < N:
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            # print("ELEMENT",element)
            matmul.append(element)
    # print(np.array(matmul).shape, matmul)
    # print(matmul)
    # print("matmul", np.array(matmul).shape, a_n,b_m,N)
    matmul = np.reshape(np.array(matmul), (a_n,b_m,N))        
    return matmul

def neg_poly_multiplication(neg_poly, neg_poly_exp, poly, poly_exp, poly_mod, d):
    # print(np.array(neg_poly).shape, np.array(neg_poly_exp).shape, np.array(poly).shape, np.array(poly_exp).shape)
    # print("VALUES______________________-")
    # print(neg_poly, neg_poly_exp, poly, poly_exp)
    max_degree = max(len(neg_poly), len(poly),d)
    # max_degree = max(max_degree, d)
    res = np.array([0]*(int(max_degree*2+1)))
    # print("LENGTHS", len(neg_poly), len(poly), d*d-1)
    for i in range(len(neg_poly)):
        # temp = np.array([0]*(len(poly)*len(neg_poly)))
        temp = np.array([0]*(int(max_degree)*2+1))
        # print("temp", temp)
        for j in range(len(poly)):
            # print(poly_exp)
            # print("---------",i,j,neg_poly_exp[i], poly_exp[j], max_degree)
            exp = neg_poly_exp[i] + poly_exp[j] + max_degree
            # print("indices",exp)
            # print(temp[exp])
            temp[exp] = temp[exp] + neg_poly[i]*poly[j]
        # print("temp all",temp)
        # temp_neg_poly = np.poly1d(temp[:len(poly)])
        # temp_poly = np.poly1d(temp)
        # print("temp poly", temp_poly)
        res = res + temp
    # print("RES",res)
    # res_neg = np.append(res[:d], [0])
    # res_pos = np.flip(res[d:])
    # res_neg = np.append(res[:d-1])
    res_neg = res[:max_degree]
    res_pos = np.flip(res[max_degree:])
    # print("RES_NEG, RES_POS")
    # print(res_neg, res_pos)
    element_neg = np.polydiv(res_neg, poly_mod)[1]*-1
    element_neg[0] = element_neg[0]*-1
    element_pos = np.polydiv(res_pos,poly_mod)[1]
    # print("element_pos", element_pos)
    # print("ELEMENT", element)
    if len(np.array(element_pos)) < d+1:
        element_pos = np.pad(element_pos, (d-len(np.array(element_pos)),0), 'constant', constant_values=0)  
    if len(np.array(element_neg)) < d+1:
        element_neg = np.pad(element_neg, (d-len(np.array(element_neg)),0), 'constant', constant_values=0)  
    # print(np.array(element_neg), np.array(element_pos))
    element = np.array(element_pos) + np.array(element_neg)
    # print(element)
    # print(np.poly1d(element))
    return element

def neg_poly_multiplication_right(poly, poly_exp,neg_poly, neg_poly_exp,poly_mod, d):
    # print(np.array(neg_poly).shape, np.array(neg_poly_exp).shape, np.array(poly).shape, np.array(poly_exp).shape)
    res = np.array([0]*(d*2+1))
    # print("LENGTHS", len(neg_poly), len(poly), d*d-1)
    for i in range(len(neg_poly)):
        # temp = np.array([0]*(len(poly)*len(neg_poly)))
        temp = np.array([0]*(d*2+1))
        # print("temp", temp)
        for j in range(len(poly)):
            # print(poly_exp)
            # print("---------",i,j,neg_poly_exp[i], poly_exp[j])
            exp = neg_poly_exp[i] + poly_exp[j] + d 
            # print("indices",exp)
            # print(temp[exp])
            temp[exp] = temp[exp] + poly[j]*neg_poly[i]
        # print("temp all",temp)
        # temp_neg_poly = np.poly1d(temp[:len(poly)])
        # temp_poly = np.poly1d(temp)
        # print("temp poly", temp_poly)
        res = res + temp
    res_neg = np.append(res[:d], [0])
    res_pos = np.flip(res[d:])

    element_neg = np.polydiv(res_neg, poly_mod)[1]*-1
    element_neg[0] = element_neg[0]*-1
    element_pos = np.polydiv(res_pos,poly_mod)[1]

    element = element_neg + element_pos
    # print("ELEMENT", element)
    if len(np.array(element)) < d:
        element = np.pad(element, (d-len(np.array(element)),0), 'constant', constant_values=0)  

    # print(np.poly1d(element))
    return element

def neg_poly_matmul(neg_a, b, mod, poly_mod, N=2048):
    matmul = []
    neg_a = np.array(neg_a)
    b = np.array(b)
    print("NEG_MATMUL", np.shape(neg_a), np.shape(b))
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
                # element = [custom_mod(num, mod) for num in intermediate_element]      
                # element = np.fmod(intermediate_element, mod)     
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
    print("NEG_POLY_MATMUL", a.shape, b.shape)
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
                # element = [custom_mod(num, mod) for num in intermediate_element]  
                # element = np.fmod(intermediate_element, mod)            
                element = mod_2D(intermediate_element.c, mod)
                if len(element) < N:
                    element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            if len(element) < N:
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            matmul.append(element)
    matmul = np.reshape(np.array(matmul), (a_n,b_m,N))        
    return matmul
# poly_mod = np.poly1d([1] + [0]*(2-1) + [1])
# A0 =[[[0,0],[1,1]]] 
# # A0 =[[[1,1],[7,4]]] 
# rand = [[[0,4],[8,8],[1,8],[15,16],[2,0],[9,12],[14,10]],[[5,5],[2,6],[4,8],[2,1],[9,7],[13,3],[15,16]]]
# x = poly_matmul(A0, rand, 17 ,poly_mod, 2)
# print("$$$$$$$$$$$$$$$$$$$",x)
# tau = 7
# g = []
# for i in range(tau):
#     val = 17**(i/tau)
#     g.append(val)
# id = np.identity(1)
# G = [[i]*2 for i in g]
# print("$$$$$$$$$$$$$$$$$$$",G)

# a=[[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]],[[7,7],[8,8]]]
# b1=[[[1,2,3]],[[4,5,6]]]
# mod = 5
# N=2
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# poly_matmul(a,b1,mod,poly_mod,N=N)
# print(STA)

# multiply a 2D matrix/vector where its elements are polynomials
def poly_matmul_no_mod(a,b, poly_mod, N=2048):
    matmul = []
    a = np.array(a)
    b = np.array(b)
    # print(a.shape, b.shape)
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
                # print("a", i, k)
                # print("b", k, j)
                # p1 =  np.polydiv(np.polymul(a[i][k], b[k][j]) , np.poly1d(poly_mod))[1]
                # print(p1)
                # poly_mul = [custom_mod(num, mod) for num in p1]
                # print(poly_mul)
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
    # random_coefficients = []
    # matrix_size = n*m
    # for i in range(matrix_size):
    #     coefficients = [random.randrange(-mod, mod) for i in range(N)]
    #     random_coefficients.append(coefficients)
    #     # print(coefficients)
    # matrix = np.reshape(random_coefficients,(n,m,N))

    matrix = [random.randint(-(mod-1)//2, (mod-1)//2) for _ in range(n*m*N)]
    matrix = np.reshape(np.array(matrix), (n,m,N))
    # matrix.astype("float")

    # matrix = np.random.randint(-mod+1,mod, size = (n,m,N))
    return matrix

def gen_message(n,m,N,w):
    random_coefficients = np.zeros(n*m*N)
    indices = np.random.choice(n*m*N, w, replace=False)
    random_coefficients[indices] = 1
    matrix = np.reshape(random_coefficients,(n,m,N))
    return matrix

# print(gen_message(1,5,3,10))
# exit()

def gen_matrix_ring_with_norm(n, m, N, norm):
    # random_coefficients = []
    # matrix_size = n*m
    # for i in range(matrix_size):
    #     coefficients = [random.randrange(-norm, norm) for i in range(N)]
    #     # coefficients = [random.randrange(0, norm+1) for i in range(N)]

    #     random_coefficients.append(coefficients)
    #     # print(coefficients)
    # # matrix = np.reshape(random_coefficients,(n,m,N))
    matrix = np.random.randint(-norm,norm, size = (n,m,N))
    # print(type(matrix), matrix.dtype)
    # exit()
    return np.array(matrix)
# mod =10
# N= 3
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# a = gen_matrix_ring(2,3,mod,poly_mod,N)

# b = gen_matrix_ring(3,5,mod,poly_mod,N)
# mat = poly_matmul(a,b,mod, poly_mod,N=N)
# print(a)
# print(b)
# print(mat)

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
    # polymod = np.poly1d([1] + [0]*(N-1) + [1])
    random_coefficients = np.random.randint(-beta+1,beta, size=(n,m,N))
    # for i in range(n):
    #     row = []
    #     for j in range(m):
    #         # x = [random.randrange(-beta, beta+1) for i in range(N)]
    #         x = [random.randrange(0, beta+1) for k in range(N)]

    #         # S = poly.polydiv(x, polymod)[1]
    #         # rand = LA.norm(S, np.inf)
    #         # print(rand)
    #         # while rand > beta:
    #             # S = poly.polydiv(x, polymod)[1]
    #             # rand = LA.norm(S, np.inf)
    #         row.append(x)
    #     random_coefficients.append(row)
    # random_coefficients = np.reshape(np.array(random_coefficients), (n,m,N))

    return random_coefficients

# def generate_random_norm(n, m, N, beta):
#     polymod = np.poly1d([1] + [0]*(N-1) + [1])
#     random_coefficients = []
#     for i in range(n):
#         row = []
#         for j in range(m):
#             x = [random.randrange(-beta, beta+1) for i in range(N)]
#             # x = [random.randrange(0, beta+1) for i in range(N)]

#             # S = poly.polydiv(x, polymod)[1]
#             # rand = LA.norm(S, np.inf)
#             # print(rand)
#             # while rand > beta:
#                 # S = poly.polydiv(x, polymod)[1]
#                 # rand = LA.norm(S, np.inf)
#             row.append(x)
#         random_coefficients.append(row)
#     random_coefficients = np.reshape(np.array(random_coefficients), (n,m,N))
#     return random_coefficients
# import
def bdlop_commit(A0, message, rand, q1_mod, q2_mod, poly_mod, n1=1,n2=1,m=4, N= 2048):
    A0 = np.array(A0)
    message = np.array(message)
    rand = np.array(rand)
    # print("rand", rand.shape)
    # print("BDLOP COMMIT", n1, m, n1+n2, A0.shape, poly_mod)
    comm1 = poly_matmul(A0[:n1,:], rand, q1_mod, poly_mod, N=N)
    comm2_intermediate = poly_matmul(A0[n1:,:], rand, q2_mod, poly_mod, N=N)
    comm2 = poly_add_3D(comm2_intermediate, message ,q2_mod, poly_mod, N)
    # print("________________________COMM2-")
    # print(comm2_intermediate[0])
    # print("________________________MSG")
    # print(message)
    # exit()
    # print(n1,n2,m,N)
    # for i in range(len(comm2_intermediate[0])):

    #             #         intermediate_element = np.polydiv(np.polyadd(np.array(element), np.array(poly_val)) , poly_mod)[1]
    #             # # print(intermediate_element)
    #             # element = [custom_mod(num, mod) for num in intermediate_element]
    #     # print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE", np.array(comm2_intermediate[0]),np.array(message[0]))
    #     # print(len(comm2_intermediate[0]),i)
    #     # exit()
    #     element = np.polydiv(np.polyadd(np.array(comm2_intermediate[0][i]),np.array(message[0][0])), poly_mod)[1] 
    #     # element = [custom_mod(num, q2_mod) for num in element]
    #     # element = np.fmod(element, mod) 
    #     element = mod_2D(element.c, q2_mod)
    #     # print(element)
    #     # element = mod_2D(element.c, q2_mod)
    #     # print(element)
    #     # print(element, q2_mod)
    #     # element = np.fmod(element, q2_mod)            

    #     if len(element) < N:
    #             # element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
    #         element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
    #     # print("AFTERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR", element)
    #     comm2.append(element)
    # # print("__________________________BDLOP COMMIT______________________________")
    # # print(np.array(comm2).shape, np.array(comm2_intermediate).shape)
    # # print(comm2, comm2_intermediate)

    # comm2 = np.array(np.reshape(comm2, comm2_intermediate.shape))
    comm = np.concatenate(([comm1[0]], comm2), axis=0)
    # print(comm.shape)
    # print(comm)
    return comm 

# mod = 10
# N= 3
# n= 1
# k= 3
# l= 1
# q1_mod = 10
# q2_mod = 20
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# a0 = [[[1,1,1],[1,5,7],[3,6,6]],[[0,0,0],[1,1,1],[17,9,14]]]
# # message = [[[1,2,6],[8,7,4],[2,5,4],[4,8,9],[5,0,2]]]
# message = [[[1,0,0],[0,1,0],[0,1,0],[0,0,1],[1,1,1]]]
# random = [[[9,0,9],[9,1,4],[2,1,9],[5,6,9],[9,6,6]],[[2,3,3],[3,7,6],[9,6,1],[5,1,4],[8,4,4]],[[0,1,5],[2,8,4],[7,3,5],[5,6,9],[3,7,5]]]
# comm = bdlop_commit(a0, message, random, q1_mod, q2_mod, poly_mod,n,l,k,N)
# print(comm)
# print("__________________________")

# mod =10
# N= 3
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# a0,a1,a2 = bdlop_ckeygen(n=1, k=3, l=1, q1=10, q2=20, N=3)
# print("!!!!!!!!!!!!!!")
# # print(a0)
# message = gen_matrix_ring(1,5,mod,N)
# print(message)
# rand = gen_matrix_ring(3,5,mod,N)
# print(rand)
# q1_mod = 10
# q2_mod = 20
# comm = bdlop_commit(a0, message, rand, q1_mod, q2_mod, poly_mod,n1=1,n2=1,m=3,N=3)
# print(lkk)

# def bdlop_commit(A0, message,rand, q1_mod, q2_mod, n1=1,n2=1,m=4, N= 2048):
#     new_matrix= []
#     for i in range(n1+n2):
#         if i >= n1:
#             mod = q2_mod
#         else:
#             mod = q1_mod
#         for j in range(m):
#             # print(A0[i][j], rand[0][j])
#             new_poly = np.multiply(A0[i][j],rand[0][j])
#             element = np.mod(new_poly, mod)
#             # print(element)
#             new_matrix.append(element)
#     new_matrix = np.reshape(new_matrix, (n1+n2,m,N))
#     comm1 = new_matrix[0:n1,0:m] #+ np.dstack([np.zeros(n1)]*N) 
#     comm2 = np.mod(np.add(new_matrix[n1:n1+n2,0:m], message), q2_mod) 
#     comm = np.concatenate((comm1[0], comm2[0]), axis=0)
#     return comm 

# def bdlop_combine(r, r_prime, mod, poly_mod, N):
#     result = poly_matadd(r, r_prime, mod, poly_mod, N)
#     return result 



# rand_0= [[[9,0,9],[9,1,4],[2,1,9],[5,6,9],[9,6,6]],[[2,3,3],[3,7,6],[9,6,1],[5,1,4],[8,4,4]]]
# rand_1= [[[6,3,1],[3,6,9],[1,1,4],[7,1,5],[1,9,3]],[[1,6,3],[3,9,1],[2,2,8],[3,0,8],[2,6,6]]]    
# mod = 12
# N = 3
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# result_0 = bdlop_combine(rand_0,rand_1,mod, poly_mod,N)
# print(result_0)

#TODO: consider what dim are a,b are they 2D or 3D and make consistent
def poly_add_2D(a,b,mod, poly_mod, N=2048):
    sum = []
    # print("+++++++++++++++++++++++\n", a)
    # print("+++++++++++++++++++++++\n", b)
    a = np.array(a)
    b= np.array(b)
    print("POLYADD 2D", a.shape, b.shape)
    if a.shape != b.shape:
        raise ValueError("a and b are not of the same dimensions")
    for i in range(len(a)):
        # element = [custom_mod(num, mod) for num in poly.polydiv([custom_mod(num, mod) for num in np.polyadd(a[i],b[i])], poly_mod)[1]]         
        intermediate_element = poly.polydiv(np.fmod(np.polyadd(a[i],b[i]), mod), poly_mod)[1]
        # element = np.fmod(intermediate_element, mod)
        element = mod_2D(intermediate_element.c, mod)
        # element = np.fmod(poly.polydiv(np.fmod(a[i] + b[i], mod), poly_mod)[1], mod) 
        if len(element) < N:
            element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
        sum.append(element)
    sum = np.reshape(sum, a.shape)
    return sum

# B = [[11,3,13],[17,10,13],[8,10,2],[17,11,7],[15,15,3],[16,3,11],[7,9,8]]
# c2 = [[11,1,17],[0,0,13.06825481],[10,8,8.70709379],[10,18,19.22128158],[17,15,6.07836596],[18,8,8.99562482],[10,4,0.07345379]]
# N=3
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])    

# x= poly_add_2D(B,c2,20, poly_mod, N)
# print(x)

def poly_add_2D_no_mod(a,b, poly_mod, N=2048):
    sum = []
    a = np.array(a)
    b= np.array(b)
    # print("+++++++++++++++++++++++\n", a)
    # print("+++++++++++++++++++++++\n", b)
    if a.shape != b.shape:
        raise ValueError("a and b are not of the same dimensions", a.shape, b.shape)
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
    # print(a.shape, b.shape)
    if a.shape != b.shape:
        raise ValueError("a and b are not of the same dimensions", a.shape,b.shape)
    for i in range(len(a)):
        sum_temp = []
        for j in range(len(a[i])):
            # print(np.array(a[i][j]).shape, np.array(b[i][j]).shape)
            element = np.polydiv(np.polyadd(np.poly1d(a[i][j]),np.poly1d(b[i][j])), poly_mod)[1].coefficients
            if len(element) < N:
                # element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            sum_temp.append(element)
        sum.append(sum_temp)
    sum = np.reshape(sum, a.shape)
    return sum 

def poly_add_3D(a, b, mod, poly_mod, N=2048):
    sum = []
    a= np.array(a)
    b=np.array(b)
    # print("ADD 3D", a.shape, b.shape)
    if a.shape != b.shape:
        raise ValueError("a and b are not of the same dimensions:", a.shape, b.shape)
    for i in range(len(a)):
        sum_temp = []
        for j in range(len(a[i])):
            # print(np.array(a[i][j]).shape, np.array(b[i][j]).shape)
            # element = np.array([custom_mod(num, mod) for num in np.polydiv(np.polyadd(a[i][j],b[i][j]), poly_mod)[1]]) 
            intermediate_element = np.polydiv(np.polyadd(a[i][j],b[i][j]), poly_mod)[1]
            # element = [custom_mod(num, mod) for num in intermediate_element]
            # element = np.fmod(intermediate_element, mod) 
            element = mod_2D(intermediate_element.c, mod)
            # element = np.array([custom_mod(num, mod) for num in np.polydiv([custom_mod(num, mod) for num in np.polyadd(a[i][j],b[i][j])], poly_mod)[1]]) 
            # element = np.fmod(poly.polydiv(np.fmod(a[i][j] + b[i][j],mod), poly_mod)[1], mod) 
            # print(element)
            if len(element) < N:
                # element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
                element = np.pad(element, (N-len(element),0), 'constant', constant_values=0)
            sum_temp.append(element)
        sum.append(sum_temp)
    # print("SUM", sum)
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
    # print(sum)
    # sum = poly_add_2D(a,b,mod, poly_mod,N)
    return np.array(sum)

def bdlop_randomize(comm, A0, rand, n1, q1_mod, q2_mod, poly_mod, N):
    # print("A0 submatrix\n",A0[0:n1,0:m])
    # print("rand\n",rand.shape)
    A0 = np.array(A0)
    rand = np.array(rand)
    comm = np.array(comm)
    temp1 = poly_matmul(A0[:n1,:], rand, q1_mod, poly_mod, N=N)
    temp2 = poly_matmul(A0[n1:,:], rand, q2_mod, poly_mod, N=N)
    # print("TEMP1\n", temp1)
    # print("TEMP2\n", temp2)
    # print("comm t1 \n",comm[0] )
    # print("comm t2 \n", comm[1])
    # poly_add_3D(comm,)
    t1 = poly_add_3D([comm[0]],[temp1[0]],q1_mod, poly_mod,N)
    t2 = poly_add_3D([comm[1]],[temp2[0]],q2_mod, poly_mod,N)
    new_comm = np.concatenate((t1, t2), axis=0)
    return new_comm

# # mod =9
# N= 2
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# # # rand_prime = gen_matrix_ring(2,3,mod,N)
# q1_mod = 10
# q2_mod = 20
# # a0,a1,a2 = bdlop_ckeygen(n=1, k=2, l=1, q1=q1_mod, q2=q2_mod, N=N)
# a0 = np.array([[[1,1],[7,4]],[[0,0],[1,1]]])
# rand = [[[6,5],[8,3],[7,5]],[[0,3],[7,4],[1,4]]]
# rand_prime = [[[0,4],[8,8],[1,8]],[[5,5],[2,6],[4,8]]]

# # message = gen_matrix_ring(1,3,mod,N)
# message = [[[5,7],[3,3],[8,2]]]

# # print("message")
# # print(message)
# # rand = gen_matrix_ring(2,3,mod,N)
# comm = bdlop_commit(a0, message, rand, q1_mod, q2_mod, poly_mod,n1=1,n2=1,m=3,N=N)
# print("CCCCCCCCCCCCCCCCCCCCCCCCc",comm)
# res = bdlop_randomize(comm,a0,rand_prime,n1=1, n2=1,m=2,q1_mod=q1_mod, q2_mod=q2_mod, poly_mod=poly_mod,N=N)
# # print("A0")
# # print(a0)
# # print("rand")
# # print(rand)
# print("RESULT OF BDLOP RANDOMIZE")
# print(res)
# # print("COMM")
# # print(comm)
# # print("RAND_PRIME")
# # print(rand_prime)
# N= 2
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# A0 =[[[1,1],[7,4]],[[0,0],[1,1]]] 
# rand = [[[0,4],[8,8],[1,8]],[[5,5],[2,6],[4,8]]]
# res1 = poly_matmul([A0[0]], rand, 9, poly_mod,N)
# res2= poly_matmul([A0[1]],rand,17,poly_mod,N)
# print("RESS", res1, res2)
#TODO: NIZK part

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
    # print("INSIDE CTS_SETUP", np.array(A).shape)
    D = gen_matrix_ring(l, l+l_prime, q2,N)
    kappa = 14
    gamma = 2**(28.3) 
    gamma_prime = 2**51.07
    alpha = 2**20.14

    return A,D, q1, q2, N, kappa, gamma, gamma_prime, alpha

def cts_commit(A, message, rand, G, N,poly_mod, q1,q2, n1, l=1, tau=7):
    # g = []
    # for i in range(tau):
    #     val = int(q2**(i/tau))
    #     g.append(val)

    # g = [int(q2**(i/tau)) for i in range(tau)]
    # print(g, g11)
    # print(np.array(g).shape, np.array(g11).shape)
    # exit()
    # iterable_g = (q2**(1/tau)**y for y in range(tau))
    # print(iterable_g)
    # g = np.fromiter(iterable_g, dtype=int)
    # print("g!!!!!!!!!!!\n",g)
    # id = np.dstack([np.identity(l)]*N)
    # print(id)
    # print("HERE")
    # print(id.shape, np.array(g).shape)
    # G1= np.tensordot(id, g, axes=0)
    # id = [[i[0]]*N for i in np.identity(l)]
    # G = [[i]*N for i in g]

    # print(G1.tolist())
    # print(G)
    # print(G1.shape, np.array(G).shape)
    # exit()
    # print(G)
    # print("Identity", np.identity(l))
    # print("Identity",id)
    # G = np.tensordot([id], g, axes=0)
    # G=G[0]
    # print("GGGGGGGGG!!!!!!!!!!!\n",G)
    # reformating G_new
    # G_new = np.array([[x] for x in G[0][0]])
    # print("G_new\n",G[0])
    # print("MESSAGE\n", message)
    # print("SHAPE",G.shape, message.shape)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print(poly_mod)
    if type(G) == int:
        # print("int")
        dott = message
    else:
        # print("ok")
        # print(message.shape)
        dott = np.array(poly_matmul(message, np.array(G), q2, poly_mod,N))
    # print(dott.tolist())
    # print(message)
    # exit()
    # print("dott of message and G\n",dott, dott.shape)
    # exit()
    # print("BDLOP COMMIT", n1, l*tau, n1+l, A.shape)
    # dott = message
    # print(np.array(rand).shape)
    comm = bdlop_commit(A, dott, rand, q1, q2, poly_mod, n1=n1,n2=l,m=l*tau, N=N)
    # print("COMM SHAPE",comm.shape)
    # print("COMM",comm)
    return comm

# A = [[[1,1,1],[19,5,19]],[[0,0,0],[1,1,1]]]
# rand = [[[0,5,1],[3,7,6],[5,7,8],[7,9,7],[3,4,5],[4,7,8],[1,9,5]],[[3,5,7],[5,0,5],[7,9,8],[1,4,7],[6,9,0],[5,5,2],[2,7,5]]]
# message = [[[1,0,1]]]
# N=3
# q1= 24
# q2 =20
# comm = cts_commit(A, message, rand,N,poly_mod, q1,q2, 1, l=1)
# print("$$$$$$$$$$$$$")
# print(comm)
# print("$$$$$$$$$$$$$")
# N= 2
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# q1_mod = 9
# q2_mod = 17
# A0 =[[[1,1],[7,4]],[[0,0],[1,1]]] 
# rand = [[[0,4],[8,8],[1,8],[15,16],[2,0],[9,12],[14,10]],[[5,5],[2,6],[4,8],[2,1],[9,7],[13,3],[15,16]]]
# message =np.array([[[1,1]]])
# result = cts_commit(A0, message, rand, N, poly_mod, q1_mod, q2_mod,n1=1,l=1)
# print("_______________________")
# print(result)
# mod =9
# N= 3
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# q1_mod = 10
# q2_mod = 20
# A,D, q1, q2, N, k, gamma, gamma_prime, alpha = cts_setup(n=1, k=4 ,l=1, l_prime=1,q2=20,N=3,q1=24)
# print(A)
# rand = gen_matrix_ring(4,7,mod,3)
# print(rand)
# message =np.array([[[0,0,1]]])
# comm = cts_commit(A, message, rand, N, poly_mod, q1_mod, q2_mod,n1=1,l=1)
# print("A")
# print(A)
# print("rand")
# print(rand)
# print("CTS COMMITMENT")
# print(comm)
# print(ksdfs)

def cts_randomize(comm, A, rand, n1, q1_mod, q2_mod, poly_mod, N):
    new_comm = bdlop_randomize(comm, A, rand, n1, q1_mod, q2_mod, poly_mod, N)
    # print("cts rando", new_comm.shape)
    return new_comm

#TODO:##################################TEST RANDOMIZE############################################
# print("TEST Randomize")
# mod =9
# N= 3
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# q1_mod = 10
# q2_mod = 20
# A,D, q1, q2, N, k, gamma, gamma_prime, alpha = cts_setup(n=1, k=3 ,l=1, l_prime=1,q2=q2_mod,N=N,q1=q1_mod)
# rand = gen_matrix_ring(3,5,mod,N)
# message =np.array([[[0,0,1]]])
# comm = cts_commit(A, message, rand, N, poly_mod, q1_mod, q2_mod,n1=1,l=1)
# rand_prime = gen_matrix_ring(3,5,mod,N)
# comm_prime = cts_randomize(comm,A,rand_prime,1, 1,5, q1_mod, q2_mod, poly_mod, N)
# print("A0")
# print(A)
# print("rand")
# print(rand)
# print("CTS COMMITMENT")
# print(comm)
# print("CTS COMMITMENT RANDOMIZE")
# print(comm_prime)
# print(ksdfs)
######################
# k = 1
# l = 1
# tau = 7
# beta=10000000
# q2_mod=2**79-67
# N=2048
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# A,D,q1,q2,N,param_k,param_gamma, param_gamma_prime, alpha = cts_setup()
# rand = generate_random(k,l*tau,N,beta)
# rand_prime=generate_random(k,l*tau,N,beta)
# message = gen_matrix_ring(l,k,q2_mod, poly_mod,N)
# comm = cts_commit(A,message,rand)
# comm_prime = cts_randomize(comm,A,rand_prime)
# print(comm_prime)

def cts_combine(rand, rand_prime):
    rand = np.array(rand)
    rand_prime = np.array(rand_prime)
    if rand.shape != rand_prime.shape:
        raise ValueError("a and b are not of the same dimensions")
    return np.add(rand,rand_prime)

##################################TEST COMBINE##############################################
# print("TEST Combine")
# k = 4
# l = 1
# tau = 2
# beta=1000
# N=3
# # A,D,q1,q2,N,param_k,param_gamma, param_gamma_prime, alpha = cts_setup()
# rand = generate_random(k,l*tau,N,beta)
# rand_prime=generate_random(k,l*tau,N,beta)
# combined = cts_combine(rand, rand_prime)
# print("RAND\n",rand)
# print("RAND_PRIME\n",rand_prime)
# print("COMBINED RANDOM\n",combined)
# # print(np.add([4,5,6], [8,9,0]))
# print(SDFDS)

def cts_keygen(l, l_hat, tau, D, G,N,q1,q2,beta=1):
    polymod = np.poly1d([1] + [0]*(N-1) + [1])
    T = gen_matrix_ring_with_norm(l+l_hat,l*tau,N, beta)
    # print(T)
    # print(D.shape, np.array(T).shape)
    a0_temp = poly_matmul(D,T,q2, polymod,N)
    # print("A0 calc")
    # print(a0_temp,"\n\n", G.shape)
    # print(a0)
    # print(G)
    a0 = poly_add_3D(a0_temp,np.array(G),q2, polymod, N)
    B= gen_matrix_ring(l, l*tau, q2, N)
    u = gen_matrix_ring(l,1,q2,N)
    # print(np.array(a0).shape)
    # print("a0")
    return a0, B, u, T

#################TEST KEYGEN#########################
# print("test key gen cts")

# l = 1
# l_hat = 2
# tau = 7
# D = [[[0,7,15],[12,1,10],[8,16,2]]]
# G = [[[20,20,20],[1.5341274,1.5341274,1.5341274],[1.06304496,1.06304496,1.06304496],[1.00877217,1.00877217,1.00877217],[1.0012484806087052,1.0012484806087052,1.0012484806087052],[1.0001782590155153,1.0001782590155153,1.0001782590155153],[1.0000254636283734,1.0000254636283734,1.0000254636283734]]]
# N = 3
# q1_mod = 10
# q2_mod = 20
# beta = 10000000 
# A0,B,u,T = cts_keygen(l, l_hat, tau, D, G,N,q1_mod, q2_mod,beta)
# print("---------------------")
# print(A0,B,u,T)
# mod =9
# N= 3
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# q1_mod = 10
# q2_mod = 20
# A,D, q1, q2, N, k, gamma, gamma_prime, alpha = cts_setup(n=1, k=3 ,l=1, l_prime=2,q2=q2_mod,N=N,q1=q1_mod)
# tau=7
# print(D)

# g = []
# for i in range(tau):
#     val = q2**(1/tau)**i
#     g.append(val)
# # iterable_g = (q2**(1/tau)**y for y in range(tau))
# # print(iterable_g)
# # g = np.fromiter(iterable_g, dtype=int)
# print("g!!!!!!!!!!!\n",g)
# l=1
# id = np.identity(1)
# id = [[i]*N for i in np.identity(l)]
# print("Identity", np.identity(l))
# print("Identity",id)
# G = np.tensordot(id, g, axes=0)
# # G=G[0]
# print("GGGGGGGGG!!!!!!!!!!!\n",G)

# l=1
# g = []
# for i in range(tau):
#     val = q2**(1/tau)**i
#     g.append(val)
# G = np.tensordot(np.identity(l), g, axes=0)
# G_new = np.array([[x]*N for x in G[0][0]])
# a0,b,u,t =cts_keygen(1,2,7,D,[G_new],3,q2,beta=10000000000)
# print("RESULTS")
# print(a0,'\nBBBBB\n',b,'\nUUUUUUU\n',u,'\nTTTTTTTT\n',t)
# print(STOP)
N = 512
# IR = sg.IntegerRing()
# P = sg.PolynomialRing(IR, 't')
# t = P.gen()
# R = sg.QuotientRing(P, t**N + 1, 'X')
# X = R.gen()
q2 = 67108837
# IR_q = sg.IntegerModRing(q2)
# P_q = sg.PolynomialRing(IR_q, 't')
# t_q = P_q.gen()
# R_q = sg.QuotientRing(P_q, t_q**N + 1, 'X')
# X_q = R_q.gen()

# IR_q = sg.IntegerModRing(q2)
# P_q = sg.PolynomialRing(IR_q, 't')
# # t_q = P_q.gen()
# R_q = P_q.quotient( t_q**N + 1, 'X')
# X_q = R_q.gen()

# R = sg.Zmod(q2)['x']
# f = x**N + 1
# S = R.quotient(f)

Zq = sg.Zmod(q2)
R = sg.PolynomialRing(Zq, 'x')
x = R.gen()
R_q = R.quotient(x**N + 1, 'X')
X = R_q.gen()
def find_particular_solution(A, t, q):
    n, m = A.nrows(), A.ncols()
    M = A.augment(sg.identity_matrix(R, n)).stack(sg.Matrix(R, ncols=n + ncols, nrows=m - n))
    M = M.echelon_form()
    for row in M.rows():
        if A * row[:m] % q == t:
            return vector(ZZ, row[:m])
    raise ValueError("No solution found") 

# def rot(a,n,m,degree,poly_mod):
#     # print(a.shape)
#     new_a_ZZ= np.zeros((n*degree, m*degree))
#     # print(new_a_ZZ.shape)
#     for i in range(n):
#         for j in range(m):
#             row = []
#             for k in range(degree):
#                 x = [0]*degree
#                 # x[degree-k-1] = 1
#                 x[k] = 1

#                 # if (k == 0):
#                 #     element =  np.polydiv(np.polymul(np.poly1d(x), np.poly1d(a[i][j])),poly_mod)[1]
#                     # print(element)
#                 # print(degree)
#                 # x[degree-k-1] = 1
#                 # print(x,np.poly1d(x))
#                 element =  np.polydiv(np.polymul(np.poly1d(x), np.poly1d(a[i][j])),poly_mod)[1]
#                 # print("eeeee",element, len(element))
#                 if len(element) < degree-1:
#                     element = np.pad(element, (degree-len(element)-1,0), 'constant', constant_values=0)
#                 # print(np.array(element).shape)
#                 row.append(element)
#             # print(np.array(row).shape)
#             new_a_ZZ[i*degree:(i+1)*degree,j*degree:(j+1)*degree] = np.array(row)

#             # new_a_ZZ[i*degree:(i+1)*degree,j*degree:(j+1)*degree] = np.array(row)
#     # print(np.array(new_a_ZZ).shape)
#     return new_a_ZZ

# def rot_ZZ(A,n):

#     return A

def rot(a,n,m,degree,poly_mod):
    # print(a.shape)
    new_a_ZZ= np.zeros((n*degree, m*degree))
    # print(new_a_ZZ.shape)
    for i in range(n):
        for j in range(m):
            row = []
            for k in range(degree):
                x = [0]*degree
                # x[k] = 1
                # if (k == 0):
                #     element =  np.polydiv(np.polymul(np.poly1d(x), np.poly1d(a[i][j])),poly_mod)[1]
                    # print(element)
                # print(degree)
                x[degree-k-1] = 1
                # print(x,np.poly1d(x))
                # print("before",  np.poly1d(a[i][j]))
                element =  np.polydiv(np.polymul(np.poly1d(x), np.poly1d(a[i][j])),poly_mod)[1]
                print("eeeee",element, len(element))
                if len(element) < degree-1:
                    element = np.pad(element, (degree-len(element)-1,0), 'constant', constant_values=0)
                # print(np.array(element).shape)
                print("flipped",np.flip(element))
                row.append(np.flip(element))
            # print(np.array(row).shape)
            new_a_ZZ[i*degree:(i+1)*degree,j*degree:(j+1)*degree] = np.array(row)

            # new_a_ZZ[i*degree:(i+1)*degree,j*degree:(j+1)*degree] = np.array(row)
    # print(np.array(new_a_ZZ).shape)
    return new_a_ZZ

def unrot(a, n, m, degree):
    new_a_ZZ= []
    # print(a.shape)
    for i in range(n):
        for j in range(m):
            # print("++++++++++++++++++++++++++")
            # print(i*degree," to" , i*degree+degree," --- " ,j*degree)
            # print(np.array(a[i*degree: i*degree + degree,j*degree]).shape)
            # new_a_ZZ.append(np.reshape(a[i*degree: i*degree + degree,j*degree+1], (1,degree)))
            poly = a[i*degree,j*degree:j*degree+degree]
            # print(poly)
            # print(np.rint(poly))
            # exit()
            new_a_ZZ.append(np.flip(poly))
    # print(np.array(new_a_ZZ).shape)
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
    # x0 = F.solve_right(u)
    v = np.random.randint(0, 10, size=(11*N, N))
    # count = 0
    print("sample pre" ,q2)
    # print(np.all(np.rint(F*(v+x0))%q2 == np.rint(u)%q2) == True)
    # print(count != 0, count != 0 and np.all(np.rint(F*(v+x0))%q2 == np.rint(u)%q2) == False)
    while np.all(np.rint(F*(sg.matrix(v+x0)))%q2 == np.rint(u)%q2) == False:
        print("in")
        for i in range(x0.ncols()):
            # center_ = sg.vector(mod_2D(np.array(-x0.column(i)),q2))
            center_unchanged = sg.vector(-x0.column(i))
            x_col = DiscreteGaussianDistributionLatticeSampler(sg.transpose(F_perp), alpha, c=center_unchanged)
            tt = x_col()
            v[:,i] = np.array(tt)
        # v = sg.matrix(v)
        if np.all(np.rint(F*(sg.matrix(v+x0)))%q2 == np.rint(u)%q2) == False:
            print(np.rint(F*(sg.matrix(v+x0)))%q2)
            print()
            print(np.rint(u)%q2)

        # count += 1

    return sg.matrix(x0 + v)


def generate_f_comm(B,c2,D,A0,A2,q2, poly_mod,N):
    B_c2 = poly_add_3D(B,c2,q2,poly_mod,N)
    F_comm = np.concatenate((D,A0,B_c2,A2), axis=1)
    return F_comm
def cts_sign___(D, A0, B, comm, A2, u, N, width, height, q2, bound, poly_mod):
    c2 = comm[1:,:]
    # f_temp = poly_add_3D(B,c2,q2,poly_mod,N)
    # F_comm = np.concatenate((D,A0,f_temp,A2), axis=1)
    F_comm = generate_f_comm(B,c2,D,A0,A2,q2, poly_mod,N)
    # print("HELLO_____________________", F_comm.shape, D.shape, A0.shape, f_temp.shape, A2.shape)
    # exit()
    sampler = DiscreteGaussianDistributionIntegerSampler(sigma=np.ceil(q2/2))
    sig_comm = []
    for i in range(width*N*height):
        sig_comm.append(int(sampler()))
    # print("length", len(sig_comm))
    within_bound = False
    while not within_bound:
        # print("within_bound")
        calculated_bound = np.linalg.norm(sig_comm)
        print("within_bound",calculated_bound, bound)
        if calculated_bound <= bound:
            within_bound = True
        else:
            for i in range(width*N*height):
                sig_comm = []
                sig_comm.append(int(sampler()))
    sig_comm = np.reshape(np.array(sig_comm), (width, height, N))
    print(sig_comm.shape, F_comm.shape)
    u = poly_matmul(F_comm, sig_comm, q2, poly_mod, N)
    print("RESULT_CTS_SIGN",np.all(poly_matmul(F_comm, sig_comm, q2, poly_mod, N) == np.array(u)) )
    # F_perp = get_F_perp_trapdoor(T, w, G_perp, w_prime, R,N=N)
    # sig_comm = sample_preimage(F_perp, F_comm, u, N, alpha,q2=q2)
    return F_comm, sig_comm, u

def cts_sign(G_perp,F,T,u,alpha,N,poly_mod, q2, R, w, w_prime):
    # split_comm = np.split(comm, [n])
    # c1 = split_comm[0]
    # c2 = split_comm[1]
    # c1 = comm[:1,:]
    # c2 = comm[1:,:]
    # split_A = np.split(A, [n])
    # a1 = split_A[0]
    # a2 = split_A[1]
    # print(T)
    # exit()
    # print("======")
    # print(A.shape)
    # a1 =  A[:n,:]
    # a2 =  A[n:,:]
    # f_temp1 = np.concatenate((D,A0), axis=1)
    # f_temp2 = poly_add_3D(B,c2,q2,poly_mod,N)
    # f_comm = np.concatenate((f_temp1, f_temp2,a2), axis=1)
    # f_temp = poly_add_3D(B,c2,q2,poly_mod,N)
    # F = np.concatenate((D,A0,f_temp,a2), axis=1)  
    # print(q2)
    # G_perp = np.zeros((2,1,N))
    # G_perp[0][0][-1] = np.rint(np.sqrt(q2))
    # G_perp[1][0][-1] = -1
    
    # G_perp_ZZ = rot(G_perp, 2,1,N,poly_mod)

    # G_perp = sg.matrix(G_perp_ZZ)
    # print(f_temp1.shape, f_temp2.shape,a2.shape)
    # exit()


    # f_comm_rq = sg.Matrix(R_q, [[R_q(f_comm.tolist()[row][col]) for col in range(len(f_comm.tolist()[0]))] for row in range(len(f_comm.tolist()))])
    # u_rq = sg.Matrix(R_q,[[R_q(u.tolist()[row][col]) for col in range(len(u.tolist()[0]))] for row in range(len(u.tolist()))])
    # print(f_comm.shape)
    # print(f_comm_rq)
    # f_comm_rq = []
    # for i in range(len(f_comm.tolist())):
    #     for j in range(len(f_comm.tolist()[0])):
    #         f_comm_rq.append(R_q(sum(R_q(c) * X**i for i, c in enumerate(f_comm.tolist()[i][j]))))

            # f_comm_rq.append([R_q(a) for a in f_comm.tolist()[i][j]])
    # u_rq = []
    # for i in range(len(u.tolist())):
    #     for j in range(len(u.tolist()[0])):
    #         u_rq.append(R_q(sum(R_q(c) * X**i for i, c in enumerate(u.tolist()[i][j]))))
            # u_rq.append([R_q(a) for a in u.tolist()[i][j]])

    # print(T.shape, "=============================")
    # T_zz = rot(T, 3,1, N, poly_mod)
    # exit()
    # f_comm_ZZ = rot(F, 1,9,N,poly_mod)
    # w =sg.matrix(sg.Integers(q2),G_q2.solve_right(_D_q2)).lift()
    # print(np.all(f_comm_ZZ[0][:N] == f_comm[0][0][:]))
    # exit()
    # print(int(f_comm[0][0][0]))
    # print(f_comm)
    # exit()
    # u_ZZ = rot(u, 1, 1, N, poly_mod)

    # print(u_ZZ.shape, f_comm_ZZ.shape)
    # F = sg.Matrix(f_comm_ZZ)
    # u = sg.Matrix(u_ZZ)
    # T = sg.Matrix(T_zz)
    # print(type(f_comm), type(u))
    # print(f_comm.is_integral_domain(), u.is_integral_domain())

    # print(f_comm.nrows(), f_comm.ncols(),u.nrows(), u.ncols())
    # print(f_comm)
    # print(np.array(T).shape)
    # T_block = [sg.Matrix(T[0]), sg.Matrix(T[1]), sg.Matrix(T[2])]
    # T = ring_transpose(T).tolist()
    # print(np.array(T).shape)
    # T = T.LLL()
    # print(f_comm.ncols(), f_comm.nrows())
    # print(u.ncols(), u.nrows())
    # print(T.ncols(), T.nrows())
    # sig_comm = SamplePre_correct(f_comm_ZZ, T, u_ZZ, rand, alpha)
    # G_q2 = sg.matrix(Integers(q2), np.rint(G))
    # _B =-sg.matrix(Integers(q2), (m_ZZ + m_s_ZZ)).inverse()*sg.matrix(Integers(q2),a2).lift()

    # _B_q2 = sg.matrix(Integers(q2), _B)

    # w_prime = sg.matrix(sg.Integers(q2),G_q2.solve_right(_B_q2)).lift()
    # Fcomm = 
    F_perp = get_F_perp_trapdoor(T, w, G_perp, w_prime, R,N=N)
    sig_comm = sample_preimage(F_perp, F, u, N, alpha,q2=q2)
    print("post sample D")
    # print(sig_comm)
    # exit()
    u_check = np.all(np.rint(F*(sig_comm))%q2 == np.rint(u)%q2)
    # print(u_check)
    # u_check = np.rint(u_check).astype(int)
    # check = np.all(u_check %q2 == u %q2)
    if u_check != True:
        print("ERRRORRRRRRRRRRRRRRRRRRRRRRRRRR")
        # print(np.all(u_check %q2 == u %q2))
        print(u_check)
        print(u)
        exit()
    return sig_comm, u_check

# B = [[[11,3,13],[17,10,13],[8,10,2],[17,11,7],[15,15,3],[16,3,11],[7,9,8]]]
# comm = [[[3,7,5],[12,6,14],[8,10,14],[19,5,3],[9,11,3],[3,13,9],[11,5,15]],[[11,1,17],[0,0,13.06825481],[10,8,8.70709379],[10,18,19.22128158],[17,15,6.07836596],[18,8,8.99562482],[10,4,0.07345379]]]
# f_temp2 = poly_add_2D(B,c2,q2,poly_mod,N)

# def SamplePre(comm, T, u, alpha):
#     D = DiscreteGaussianDistributionLatticeSampler(T, alpha)
#     return D

def pad_T(T, rand, n,m):
    full_matrix  = np.zeros((n,n))
    full_matrix  = np.zeros((9*512,9*512))

    full_matrix[:1536, :m] = T
    full_matrix[:1536, 1536:5*512] = rand*-1
    full_matrix[2048:2560, 1536:5*512] = np.identity(512)

    return full_matrix

# def get_trap(T, R,N):
#     full_matrix  = np.zeros((9*N,9*N))
#     full_matrix[:N*3, :N] = np.identity(N*3)
#     full_matrix[N*3:N*4, N*3:N*4] = T*-1

#     # full_matrix[:N, N*3:N*6] = np.transpose(-1*T)
#     full_matrix[N*3:N*4, N*3:N*4] = np.identity(N)
#     full_matrix[N*4:N*5, N*4:N*5] = np.identity(N)
#     full_matrix[N*5:, N*4:N*5] = -1*R
#     full_matrix[N*5:, N*5:] = np.identity(N*4)
#     return full_matrix

def get_trap(T, R,N):
    full_matrix  = np.zeros((9*N,9*N))
    full_matrix[:N*3, :N*3] =  np.identity(N*3)
    full_matrix[:N*3, N*3:N*4] = T*-1

    # full_matrix[:N, N*3:N*6] = np.transpose(-1*T)
    full_matrix[N*3:N*4, N*3:N*4] = np.identity(N)
    full_matrix[N*4:N*5, N*4:N*5] = np.identity(N)
    full_matrix[N*5:, 4*N:N*5] = -1*R
    full_matrix[N*5:, N*5:] = np.identity(N*4)
    return full_matrix

# def SampleD(B, n, s, c):
#     v_n = 0
#     c_n = c
#     print(np.array(c).shape)
#     for i in range(n-1, 0,-1):
#         a, g = sg.matrix(B).gram_schmidt()
#         # print(type(g), len(g), type(c_n[i]))
#         # print(np.array(a)[i*512: i+1*(512+1)].shape,np.array(g).shape, np.array(c_n).shape)
#         # print(np.transpose(a[(n-i-1)*512: (n-i)*(512)]).shape,np.array(a)[(n-i-1)*512: (n-i)*(512)].shape)
#         x = np.transpose(np.array(c_n))@np.transpose(np.array(a)[(n-i-1)*512: (n-i)*(512)])/(np.array(a)[(n-i-1)*512: (n-i)*(512)]@np.transpose(np.array(a)[(n-i-1)*512: (n-i)*(512)]))
#         # print(x)
#         s_prime = s/LA.norm(a,2)
#         D = DiscreteGaussianDistributionIntegerSampler(s_prime, c_n)
#         z_i = D()
#         c_n = c_n - z_i*B[i]
#         v_n = v_n + z_i*B[i]
#     # c[i-1] = c_n[i] - z_i * B[i]
#     # v_n[i-1] = v_n[i] + z_i * B[i]
#     return v_n


# def sample_preimage(comm, T, u, rand, alpha, q2, poly_mod, N):
    T  =  rot(T, 3,1, N, poly_mod)
    rand = rot(rand,4,1, N, poly_mod)
    trap = get_trap(T,rand, N)
    short_basis = comm @ trap
    short_basis = sg.Matrix(short_basis)
    np.savetxt('T_1.txt', T, fmt='%d')
    np.savetxt('rand_1.txt', rand, fmt='%d')
    np.savetxt('trap_1.txt', trap, fmt='%d')
    np.savetxt('short_basis.txt', short_basis, fmt='%d')
    x0 = comm.solve_right(u)
    x1 = short_basis.solve_right(u)
    # sg.save(x0, "matrix_1.sobj")
    np.savetxt('x0_1.txt', np.array(x0), fmt='%d')
    np.savetxt('u_1.txt', np.array(u), fmt='%d')
    x = sg.matrix(Integer(9*N), Integer(N) )
    np.savetxt('comm_1.txt', np.array(comm), fmt='%d')
    
    # vv = DiscreteGaussianDistributionLatticeSampler(short_basis,alpha, x0)
    # temp = vv()
    # print("comm temp + x0")
    # test = np.array(comm)@(np.array(temp) + np.array(x0)) %q2
    # print(np.all(test == u))
    # test1 = np.array(T)@(np.array(temp) + np.array(x0)) %q2
    # print("T temp + x0")

    # print(np.all(test1 == u))
    # test2 = np.array(T)@(np.array(temp) + np.array(x1)) %q2
    # print("T temp + x1")

    # print(np.all(test2 == u))

    # exit()
    for i in range(x0.ncols()):
        center = sg.vector(x0.column(i))
        print(np.array(center).shape)
        x_col = DiscreteGaussianDistributionLatticeSampler(trap,alpha, -1*center)
        x.set_column(i, x_col())

    print("after")
    np.savetxt('x_1.txt', np.array(x), fmt='%d')
    # sg.save(x, "matrix_x.sobj")
    # sg.save(x0, "matrix_x0.sobj")
    # sg.save(trap, "matrix_trap.sobj")
    # sg.save(T, "matrix_T.sobj")
    # sg.save(rand, "matrix_rand.sobj")
    # sg.save(comm, "matrix_comm.sobj")
    # sg.save(u, "matrix_u.sobj")


    # res = DiscreteGaussianDistributionLatticeSampler(trap,alpha, -1*np.transpose(x0))
    # print(np.array(res).shape)
    # for i in range(N):
    #     print(i)
    #     v = DiscreteGaussianDistributionLatticeSampler(trap,alpha, -1*np.transpose(x0)[i])
    #     res.append(v())
    
    # print(np.array(res).shape, np.array(comm).shape, np.array(u).shape, np.array(x0).shape)
    test = np.array(comm)@(np.array(x) + np.array(x0)) %q2
    print("comm x + x0")
    print(np.all(test == u))
    test1 = np.array(T)@(np.array(x) + np.array(x0)) %q2
    print("T x + x0")
    print(np.all(test1 == u))
    x = sg.matrix(Integer(9*N), Integer(N) )
    for i in range(x1.ncols()):
        center = sg.vector(x1.column(i))
        print(np.array(center).shape)
        x_col = DiscreteGaussianDistributionLatticeSampler(short_basis,alpha, -1*center)
        x.set_column(i, x_col())
    test2 = np.array(T)@(np.array(x) + np.array(x1)) %q2
    print("T x + x1")
    print(np.all(test2 == u))

    print(np.array(test).shape, np.array(u).shape)

    u = np.round(np.array(u))  %q2
    print(test[:10])
    print(u[:10])
    print(np.all(test == u ))
    # print(np.all(test == u ))

    # test = np.round(test)
    exit()
    # print(res)
    # v = SampleD(trap, N, alpha, -1*x0) 
    e = poly_add_3D(x0, v, q2, poly_mod,N)
    return e

# def SamplePre_correct(comm, T, u, rand, alpha):
#     # print("...........................",len(comm))
#     # print(u)
#     # comm = sg.matrix(comm)
#     # u = sg.matrix(u)
#     # print(u.nrows(), u.ncols(), comm.nrows(), comm.ncols())
#     # print(comm.shape, u.shape)
#     # print(type(comm), type(u))
#     #
#     x0 = comm.solve_right(u)
#     v = SampleD(T, N, alpha, -1*x0) 
#     e = poly_add_3D(x0, v, q2, poly_mod,N)
#     #   
#     # x0 = sg.Matrix(R, x0.list())
#     # print(x0[0].list())
#     # print("x0 shape")
#     # print(type(x0),type(x0.polynomial().coefficients()),type(x0[0][0]),type(x0[0][0][0]))
#     # print(x0[0].polynomial().coefficients())
#     # print(x0[0].lift().list())
#     # x_list = [int(entry) for entry in x0.list()]
#     # print("x as list of integers:", x_list)
#     # print(len(v_poly.denominator().list()), len(v_poly.numerator().list()))
#     # print(type(x0), type(x0.lift()),type(x0[0]), type(x0[0].lift()), type(x0[0].lift().lift())) 
#     # print(type(np.array(x0.list())[0]))
#     # print(x0[0].coefficients())
#     # print(x0[0][0].lift().tolist())
#     # print(coeffs)
#     # T = sg.Matrix(R, T.to)
#     # print("+++++++++++++")
#     # print(T.shape)
#     # T = T.reshape((3,512))
#     T  =  rot(T, 3,1, 512, poly_mod = np.poly1d([1] + [0]*(512-1) + [1]))
#     rand = rot(rand,4,1, 512, poly_mod = np.poly1d([1] + [0]*(512-1) + [1]))
#     # T  =  rot([[T[0][0]]], 1,1, 512, poly_mod = np.poly1d([1] + [0]*(512-1) + [1]))

#     # print(T)
#     trap = get_trap(T,rand,512)
#     print("hereeeeeeeeeeeeeeeee")
#     # print(trap.shape)
#     # trap = comm*trap
#     trap = sg.Matrix(trap)
#     # print(T)
#     # print("Type:", type(T))
#     # print("Dimensions:", T.nrows(), "x", T.ncols())
#     # T = T.LLL()
#     # T_gs = T.gram_schmidt()[0]
#     # print(T_gs)
#     # print(np.array(T).shape)
#     print(alpha)
    
#     # x = trap.solve_right(u) 
#     # T = pad_t(T)
#     # D = DiscreteGaussianDistributionIntegerSampler(float(alpha))
#     # x = []
#     # for i in range(9):
#     #     z = D()
#     #     x += z + trap(i)
#     # print(x)
#     # print(type(u))
#     # print(u[0], u.ncols())
#     # print(len(u))
#     preimage = []
#     for i in range(512):
#         D = DiscreteGaussianDistributionLatticeSampler(trap, float(alpha))
#         preimage.append(list(D()))
#     print("---------------------------------")
#     preimage = np.transpose(np.array(preimage))
#     print(preimage.shape)
#     # P = DiscreteGaussianDistributionPolynomialSampler(T, float(alpha))
#     # print(D1())
#     # D2 = DiscreteGaussianDistributionLatticeSampler(T[1], alpha)
#     # D3 = DiscreteGaussianDistributionLatticeSampler(T[2], alpha)
#     # print(D())
#     # print("***********")
#     # print(P())
#     # exit()
#     # v = np.concatenate((np.array([[D() ]]), np.zeros((8,1,512))),axis=0)
#     # v = np.array(T.list()).reshape((3,1,512)) * [[D()]]

#     # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     # print(x0.ncols(), x0.nrows(), len(v), v[0])
#     # print(x0[0])
#     # print("***********")
#     # x = [D() for i in range(512)]
#     # x = sg.Matrix(np.transpose(x))
#     # test1 = x0 - preimage
#     preimage = unrot(preimage, 9,1,512)

#     x0 = unrot(x0,9,1,512)
#     x = poly_add_3D(x0, preimage, q2, poly_mod, 512)

#     return x

#     # test1 = trap*u
#     # print(len(test1))
#     # print(test1)
#     # uu = unrot(comm*test1, 9,1,512)
#     # checkk = np.all(uu == u)
#     # exit()
#     # x0 = D().solve_right(u) 

#     print(np.array(x0).shape)

#     x0 = unrot(test1, 9,1,512)
#     return x0
#     exit()
#     # v = []
#     # x = x0 + v
#     # poly_mod = np.poly1d([1] + [0]*(512-1) + [1])
#     # mod =  67108837
#     # print(poly_matmul(comm,x,mod, poly_mod, N=512))
#     # print(u)
#     # exit()
#     # assert poly_matmul() == t
#     return x
# def generate_f_comm(D, A0,A, B,c2,q2,n,l,poly_mod,N):
#     # split_A = np.split(A, [n+l])
#     # a1 = split_A[0]
#     # a2 = split_A[1]
#     a1 =  A[:n,:]
#     a2 =  A[n:,:]
#     f_temp1 = np.concatenate((D,A0), axis=1)
#     f_temp2 = poly_add_3D(B,[c2],q2,poly_mod,N)
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#     print(np.array(f_temp1[0]).shape,np.array(f_temp2[0]).shape,np.array(a2).shape)
#     f_comm = [np.concatenate((f_temp1[0], f_temp2[0],a2[0]), axis=0)]
#     print(np.array(f_comm).shape)
#     return f_comm

# D = [[[0,7,15],[12,1,10],[8,16,2]]]
# A = [[[1,1,1],[19,5,19]],[[0,0,0],[1,1,1]]]
# A0 = [[[3,10,3],[1.5341274,16.5341274,19.5341274],[15.06304496,12.06304496,19.06304496],[ 1.00877217, 19.00877217, 13.00877217],[18.00124848, 17.00124848, 15.00124848],[ 5.00017826, 19.00017826, 10.00017826],[16.00002546, 11.00002546, 17.00002546]]]
# B = [[[11,3,13],[17,10,13],[8,10,2],[17,11,7],[15,15,3],[16,3,11],[7,9,8]]]
# c2 = [[11,1,17],[0,0,13.06825481],[10,8,8.70709379],[10,18,19.22128158],[17,15,6.07836596],[18,8,8.99562482],[10,4,0.07345379]]
# q2_mod = 20
# n = 1
# l = 1
# N = 3
# result = generate_f_comm(D, A0, A, B,c2,q2_mod,n,l,poly_mod,N)
# print("$$$$$$$$$$$$$$$$$___________________")
# print(result)
# print("$$$$$$$$$$$$$$$$$___________________")















#HERE FOR VERIFY: TODO







# import oqs 
# import logging
# from pprint import pformat
# from sys import stdout
# from ac import *


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(stdout))

# logger.info("liboqs version: %s", oqs.oqs_version())
# logger.info("liboqs-python version: %s", oqs.oqs_python_version())
# logger.info(
#     "Enabled signature mechanisms:\n%s",
#     pformat(oqs.get_enabled_sig_mechanisms(), compact=True))


def cts_transfer_curr_for_ac(randomized_comm, D, A, A0, B,n, k,l, l_hat, tau, sig_comm, message, rand, rand_prime, N,poly_mod, q1,q2):
    split_sig = np.split(sig_comm, [(1+tau)*l+l_hat,((1+tau)*l+l_hat) + l*tau])
    s1 = split_sig[0] 
    s2 = split_sig[1]
    s3 = split_sig[2]
    # print("lhat", l_hat, l ,tau)
    # print([(1+tau)*l+l_hat,((1+tau)*l+l_hat) + l*tau])
    # print("-----sig", sig_comm.shape)
    # print("-----s3", s3.shape)
    # print("-----n and k", n, k)
    # new_comm = cts_commit(A0,message,rand,N, poly_mod, q1,q2, n1,l) #check params(A, message, rand,N,poly_mod, q1,q2, n1, l=1)
    # randomized_comm = cts_randomize(new_comm,A0,rand_prime,n1,n2, m, q1,q2,poly_mod,N)
    split_comm = np.split(randomized_comm, [n])
    c1 = split_comm[0]
    c2 = split_comm[1]
    rand_tilde = cts_combine(rand,rand_prime)
    split_rand = np.split(rand_tilde,[n])
    # print("-----R2", split_rand[1].shape)
    # print("-----s2", s2.shape)

    last_row_temp = poly_matmul_no_mod(split_rand[1],s2,poly_mod,N)
    # last_row = poly_add_2D_no_mod(s3,(-1)*last_row_temp,poly_mod,N)
    last_row = poly_add_3D_no_mod(s3,(-1)*last_row_temp,poly_mod,N)

    sig_comm = np.concatenate((s1, s2, last_row), axis=0)
    # intermediate_val = poly_add_2D(B,c2,q2,poly_mod,N)
    # f_comm_prime = generate_f_comm(D, A0, A, B,c2,q2,n,l,poly_mod,N)
    # A2 = A[1:]
    print("A",A.shape)
    print("A2",A[1:].shape)
    A2 = A[:1, 1:]

    f_comm = generate_f_comm(B,c2,D,A0,A2,q2, poly_mod,N)
    u = poly_matmul(f_comm, sig_comm, q2, poly_mod, N)

    #TODO: prove algorithm
    return f_comm, sig_comm, u

def cts_transfer_curr(D, A, A0, B,n, k,l, l_hat, tau, sig_comm, message, rand, rand_prime, N,poly_mod, q1,q2,n1,n2,m):
    split_sig = np.split(sig_comm, [(1+tau)*l+l_hat,((1+tau)*l+l_hat) + l*tau])
    s1 = split_sig[0] 
    s2 = split_sig[1]
    s3 = split_sig[2]
    new_comm = cts_commit(A0,message,rand,N, poly_mod, q1,q2, n1,l) #check params(A, message, rand,N,poly_mod, q1,q2, n1, l=1)
    randomized_comm = cts_randomize(new_comm,A0,rand_prime,n1,q1,q2,poly_mod,N)
    split_comm = np.split(randomized_comm, [n])
    c1 = split_comm[0]
    c2 = split_comm[1]
    rand_tilde = cts_combine(rand,rand_prime)
    split_rand = np.split(rand_tilde,[n])
    last_row_temp = poly_matmul_no_mod(split_rand[1],s2,poly_mod,N)
    last_row = poly_add_2D_no_mod(s3,(-1)*last_row_temp,poly_mod,N)
    sig_comm = np.concatenate((s1, s2, last_row), axis=0)
    # intermediate_val = poly_add_2D(B,c2,q2,poly_mod,N)
    # f_comm_prime = generate_f_comm(D, A0, A, B,c2,q2,n,l,poly_mod,N)
    # A2 = A[1:]
    f_comm = generate_f_comm(B,c2,D,A0,A[1:],q2, poly_mod,N)
    u = poly_matmul(f_comm, sig_comm, q2, poly_mod, N)

    #TODO: prove algorithm
    return f_comm, sig_comm

def cts_transfer(D, A, A0, B,n, k,l, l_hat, tau, sig_comm, message, rand, rand_prime, N,poly_mod, q1,q2,n1,n2,m, u, gamma_prime,G):
    split_sig = np.split(sig_comm, [(1+tau)*l+l_hat,((1+tau)*l+l_hat) + l*tau])
    s1 = split_sig[0] 
    s2 = split_sig[1]
    s3 = split_sig[2]
    print(s1.shape, s2.shape, s3.shape, sig_comm.shape)
    # cts_commit(A, message, rand, G, N,poly_mod, q1,q2, n1, l=1, tau=7)
    new_comm = cts_commit(A,message,rand,G,N, poly_mod, q1,q2, n1,l,tau=tau) #check params(A, message, rand,N,poly_mod, q1,q2, n1, l=1)
    # print("_____________--")
    # print(new_comm)
    randomized_comm = cts_randomize(new_comm,A,rand_prime,n1, q1,q2,poly_mod,N)
    split_comm = np.split(randomized_comm, [n])
    c2 = split_comm[1]
    # f_temp = poly_add_3D(B,c2,q2,poly_mod,N)
    rand_tilde = cts_combine(rand,rand_prime)
    split_rand = np.split(rand_tilde,[n])
    last_row_temp = poly_matmul_no_mod(split_rand[1],s2,poly_mod,N)
    last_row = poly_add_3D_no_mod(s3,(-1)*last_row_temp,poly_mod,N)
    new_sig_comm = np.concatenate((s1, s2, last_row), axis=0)
    # intermediate_val = poly_add_2D(B,c2,q2,poly_mod,N)
    # print(randomized_comm.shape)
    # f_comm_prime = np.concatenate((D,A0,f_temp,A[1:,l:]), axis=1)
    f_comm_prime = generate_f_comm(B,c2,D,A0,A[1:,l:],q2, poly_mod,N)


    sigalg = "Dilithium5"

    with oqs.Signature(sigalg) as signer, oqs.Signature(sigalg) as verifier:
        logger.info("Signature details:\n%s", pformat(signer.details))

        # Signer generates its keypair
        # signer_public_key = signer.generate_keypair()
        signer_public_key = f_comm_prime
        u = poly_matmul_no_mod(f_comm_prime, sig_comm, poly_mod, N)
        signature = u
        # Optionally, the secret key can be obtained by calling export_secret_key()
        # and the signer can later be re-instantiated with the key pair:
        # secret_key = signer.export_secret_key()

        # Store key pair, wait... (session resumption):
        # signer = oqs.Signature(sigalg, secret_key)

        # Signer signs the message
        # signature = signer.sign(message)

        # Verifier verifies the signature
        is_valid = verifier.verify(sig_comm, signature, signer_public_key)

        logger.info("Valid signature? %s", is_valid)

        return
    return nizk_prove(f_comm_prime, u, new_sig_comm, gamma_prime, q2, poly_mod, N)
 
    sigalg = "Dilithium5"

    f_comm_prime = generate_f_comm(D, A0, A, B,c2,q2,n,l,poly_mod,N)
    with oqs.Signature(sigalg) as signer, oqs.Signature(sigalg) as verifier:
        logger.info("Signature details:\n%s", pformat(signer.details))

        # Signer generates its keypair
        # signer_public_key = signer.generate_keypair()
        signer_public_key = f_comm_prime
        u = poly_matmul_no_mod(f_comm_prime, sig_comm, poly_mod, N)
        signature = u
        # Optionally, the secret key can be obtained by calling export_secret_key()
        # and the signer can later be re-instantiated with the key pair:
        # secret_key = signer.export_secret_key()

        # Store key pair, wait... (session resumption):
        # signer = oqs.Signature(sigalg, secret_key)

        # Signer signs the message
        # signature = signer.sign(message)

        # Verifier verifies the signature
        is_valid = verifier.verify(sig_comm, signature, signer_public_key)

        logger.info("Valid signature? %s", is_valid)

        return

def cts_verify(u,A0,A, B,D,n,l, comm, sig, sig_type, q2, poly_mod,N):
    split_comm = np.split(comm, [n])
    c2 = split_comm[1]
    # f_temp = poly_add_3D(B,c2,q2,poly_mod,N)
    # f_comm = np.concatenate((D,A0,f_temp,A[1:,l:]), axis=1)
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

# def sample_M():
#     return

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
    # usk = gen_message(l,1,N,w)
    # usk = np.random.choice([-1,0,1], (1,1,N))
    usk = np.zeros((1,1,N))
    indices = np.random.choice(N, w, replace=False)
    for i in indices:
        usk[0][0][i] = 1
    #organizations setup
    opk_a0, opk_B, opk_u, osk = cts_keygen(l,l_hat,tau,D,G,N,q1,q2,beta)
    return A, D, q1, q2, N, kappa, gamma, gamma_prime, alpha, opk_a0, opk_B, opk_u, osk,usk

def AC_registration(n,m,N,A,poly_mod,q1,q2,n1,l,usk, G,tau,beta=2):
    rand = generate_random(n,m,N,beta)
    # rand = generate_random(k,l*tau,N,2)

    comm = cts_commit(A, usk, rand,G,N,poly_mod, q1,q2, n1, l=l, tau=tau)
    # result = run_nizk_proof()
    # if result == False:
    #     nym = None
    #     return False, nym
    # verify_result = nizk_verify(nym)
    # if verify_result == True:
    #     nym = comm
    #     return True, nym
    # else:
    #     return False, None
    return comm,rand

# def AC_issue(D,A2, A0,B,T,u,n,l,alpha,comm,N,poly_mod,q2,rand):
    
#     sigma = cts_sign___(D,A0,B,comm, A2,u,N,10,1,q2,bound,poly_mod,)
#     return sigma

def AC_issue(D,A2, A0,B,u,comm,N,poly_mod,q2,bound):
    F_comm, sigma, u = cts_sign___(D,A0,B,comm, A2,u,N,10,1,q2,bound,poly_mod)
    return F_comm, sigma, u

def AC_prove(D,A,A0,B,n,k,l,l_hat,tau,message,rand,N,poly_mod,q1,q2,n1,m,comm,credential,u,gamma_prime,G,beta=2):
    # rand_prime = generate_random(n,m,N,beta)
    # comm_prime = cts_randomize(comm,A0,rand_prime,n1,n2,m,q1,q2,poly_mod,N)
    # sigma_prime = cts_transfer(D,A0,B,n,k,l,l_hat,tau,comm_prime,message,rand,rand_prime,N,poly_mod,q1,q2,n1,n2,m)
    # rand_prime = generate_random(n,m,N,beta)
    rand_prime = generate_random(4,2,N,beta)

    randomized_comm = cts_randomize(comm, A, rand_prime, n1, q1, q2, poly_mod, N)
    # transferred_sig = cts_transfer(D, A, A0, B,n, k,l, l_hat, tau, credential, message, rand, rand_prime, N,poly_mod, q1,q2,n1,n2,m, u, gamma_prime,G)
    transferred_f_comm, transferred_sig, u = cts_transfer_curr_for_ac(randomized_comm, D, A, A0, B,n, k,l, l_hat, tau, credential, message, rand, rand_prime, N,poly_mod, q1,q2)
    pi_prime = 0

    
    # proof = nizk_prove()

    return  transferred_f_comm, transferred_sig, randomized_comm, pi_prime, u

def AC_verify(u,A0,A,B,D,n,l, comm, sig, sig_type, q2, poly_mod,N, alpha):
    result = cts_verify(u,A0,A,B,D,n,l, comm, sig, sig_type, q2, poly_mod,N)
    split_comm = np.split(comm, [n])
    c2 = split_comm[1]
    # f_comm = generate_f_comm(D,A0,B,c2,q2,n,l,poly_mod, N)
    f_comm = generate_f_comm(B,c2,D,A0,A[1:,l:],q2, poly_mod,N)

    gamma_prime = (math.sqrt(k-n)+1)*N*alpha*math.sqrt(2) + alpha*math.sqrt(8)*N 
    result2 = nizk_prove(f_comm, u, sig, gamma_prime, q1, poly_mod, N)
    if result == True and result2 == True:
        return True
    else: 
        return False


# def matrix_multiplication_3D(n1,n2, m, q1_mod, q2_mod, N, matrix, vector):
#     new_matrix= []
#     for i in range(n1+n2):
#         if i >= n1:
#             mod = q2_mod
#         else:
#             mod = q1_mod
#         for j in range(m):
#             print(matrix[i][j], vector[0][j])
#             new_poly = np.multiply(matrix[i][j],vector[0][j])
#             element = np.remainder(new_poly, mod)
#             print(element)
#             new_matrix.append(element)
#     np.reshape(new_matrix, (n1+n2,m,N))
#     return
def ring_transpose(a):
    transpose = np.transpose(a, axes=(1,0,2))
    # transpose = []
    # a = np.array(a)
    # # print(a.shape)
    # for i in range(a.shape[1]):
    #     new_row = []
    #     for j in range(a.shape[0]):
    #         new_row.append(a[j][i])
    #     transpose.append(new_row)
    return transpose

##################TEST BDLOP COMMIT
# q1_mod = 5
# q2_mod = 5
# N=3
# m=4
# n2=1
# n1=1
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# a0 = bdlop_ckeygen(n=n1, k=m, l=n2 ,q1=q1_mod, q2=q2_mod, N=N)
# message = gen_matrix_ring(n2,m,q2_mod, poly_mod,N)
# rand = generate_random(n1+n2,m,N,10000)
# comm = bdlop_commit(a0, message,rand, q1_mod, q2_mod, n1,n2,m, N)
# print("--------a0----------")
# print(a0)
# print("--------rand----------")
# print(rand)
# print("--------msg----------")
# print(message)
# print("--------comm----------")
# print(comm)

# print(a0.shape)
# print(rand)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# new_matrix= []
# for i in range(n1+n2):
#     if i >= n1:
#         mod = q2_mod
#     else:
#         mod = q1_mod
#     for j in range(m):
#         print(a0[i][j], rand[0][j])
#         new_poly = np.multiply(a0[i][j],rand[0][j])
#         element = np.remainder(new_poly, mod)
#         print(element)
#         # break
#         new_matrix.append(element)
# new_matrix = np.reshape(new_matrix, (n1+n2,m,N))
# # print("A0")
# # print(a0)
# # print("rand")
# # print(rand)
# print("MESSAGE")
# print(message)
# print("MATRIX")
# print(new_matrix)
# # print(new_matrix.shape)
# comm1 = new_matrix[0:n1,0:m] #+ np.dstack([np.zeros(n1)]*N) 
# print("comm1")
# print(comm1)
# comm2 = np.mod(np.add(new_matrix[n1:n1+n2,0:m], message), q2_mod) 
# print("comm2")
# print(np.add(new_matrix[n1:n1+n2,0:m], message))
# comm = np.concatenate((comm1[0], comm2[0]), axis=0)
# print(comm)

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

import time 
def enc_key_gen(poly_mod, d = 512, q = 1437757, n = 2, m = 117, k = 113):
    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    A = gen_matrix_ring(n,m,q,d)
    beta = 2
    # S = generate_random_norm(k,n,d,beta)
    # E = generate_random_norm(k,m,d,beta)
    S = np.random.randint(-beta,beta+1, size=(k,n,d))
    E = np.random.randint(-beta,beta+1,size=(k,m,d))
    st = time.time()
    B_temp = poly_matmul(S,A, q, poly_mod, N=d)
    print("enc_key_gen", time.time()-st)
    # BB = poly_add_3D(B_temp, 3*E, q, poly_mod, N=d)
    B = mod_3D(len(B_temp), B_temp + 3*E,q)
    print("TEST ENC_KEY_GEN B values")
    print(B)
    B = B_temp + 3*E
    # print
    B = mod_3D(len(B),B, q)
    print(B)
    # print(B_temp.tolist()[0][:5], E.tolist()[0][:5])
    # print(B.tolist()[0][:5], BB.tolist()[0][:5])
    # exit()
    # print(A,B,S)
    return A,B,S

def enc_encrypt(A, B, miu, poly_mod, m = 117, d = 512, q = 1437757):
    beta = 2
    # r_pke = generate_random_norm(m,1,d,beta)
    r_pke = np.random.randint(-beta,beta+1,size=(m,1,d))
    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    print(np.array(A).shape, r_pke.shape)
    c0 = poly_matmul(A.astype(float), r_pke.astype(float), q, poly_mod, N=d)
    # print("HEREEEEEEEEEEEEEE")
    # print(np.array(B).shape, r_pke.shape, np.array(miu).shape)
    # c1 = poly_add_3D(poly_matmul(B, r_pke, q, poly_mod, N=d), ring_transpose(miu), q, poly_mod, d)
    # c1 = np.fmod(poly_matmul(B, r_pke, q, poly_mod, N=d)+ ring_transpose(miu),q)
    # print("TEST enc_encrypt")
    # print(c1)
    intermediate =poly_matmul(B, r_pke, q, poly_mod, N=d)+ ring_transpose(miu)
    c1 = mod_3D(len(intermediate), intermediate,q)
    print(c1)
    return r_pke , c0, c1

# k = 1
# A,B,S = enc_key_gen(d = 3, q = 1437757, n = 2, m = 7, k = 5)
# print("!!!PARAMS A!!!!!!")
# print(A)
# print("!!!PARAMS B!!!!!!")
# print(B)
# print("!!!PARAMS S!!!!!!")
# print(S)
# print(A.shape, B.shape, S.shape)
# print("!!!END!!!!!!")

# miu = []
# message = 
# r_ = generate_random()
# A, B, S = enc_key_gen()
# r_pke = enc_encrypt(A, B, miu, n, d, q)
# # print(r.shape)
# # message is 5, 7,1, 4
# message = [[[1,0,1], [1,1,1], [0,0,1], [1,0,0]]]
# miu = np.concatenate((r, message))
# d = 3
# for i in range(k):
#     miu = np.random.choice([-1,1], d)
#     message.append(miu)
# print(miu)
# r, c0, c1 = enc_encrypt(A,B,np.array([miu]), m=7, d =3, q=3)
# miu = np.concatenate((r, message))
A = [[[1082486, 60423, 1192360],
  [ 118524, 628992, 1254571],
  [ 216624, 804643, 828267],
  [1213802, 585861, 954312],
  [ 237555, 1199339, 45849],
  [1372662, 625067, 518845],
  [1192591, 1167416, 1012226]],
 [[ 967129,  227094, 204708],
  [ 758884, 1173303, 729038],
  [ 303314, 1201114, 1015810],
  [1142858, 1368850, 1104160],
  [1259926, 1156967, 1366016],
  [ 996209, 1185380, 402531],
  [ 613656, 1356069, 985405]]]

B = [[[215146, 680080, 358756],
  [ 381208, 376098,  70875],
  [ 979592,1093089, 897365],
  [ 732691, 746281, 371497],
  [ 263186, 371697,1094802],
  [1424153, 387303, 734137],
  [ 534535,1419896, 421972]],
 [[1360813,1141965, 680561],
  [ 830569, 466034, 993005],
  [1297003, 344579, 590411],
  [ 639866, 515160, 984738],
  [1413970,1401421,  12690],
  [ 591723,1348898, 804153],
  [ 402019, 396213, 563044]],
 [[ 313138,1119935, 78293],
  [1201831, 919273, 563688],
  [ 246858, 805854,1268588],
  [ 335080, 207148,1235246],
  [ 147228, 692367,1238392],
  [ 793183,1288225, 79700],
  [1315057,1167932, 281798]],
 [[1015918, 961068,1202504],
  [ 967689, 364666,1165424],
  [1014887,1216274, 253008],
  [ 350084,1226165,1159016],
  [ 943948,1316237, 742787],
  [1394284, 398057, 176099],
  [ 605734, 758031, 449850]],
 [[1142206,1316054, 531178],
  [ 690637, 475414, 704646],
  [ 991144,1230429, 128717],
  [ 734075, 371061, 152196],
  [ 368965, 886313, 100531],
  [1280862, 658664,1380323],
  [ 730858,1015009,1418617]]]

# miu = [[[895, -344, 99], [-967, 336, -37], [-250, -280, 492], [619, -501, 671]],[[-939, 710, -738], [-354, 885, 100],[-910, -476,40], [913, -49, 561]],[[-303, -115, -883],[474, -266, -942], [721, -342, -395], [13, -676, -452]],[[-335, -704, -403],[698, -435, 556],[966, -465, 495],[-137, -333, -64]],[[208, -583, -387],[826, -729, 224],[792, -833, 290],[607, 147, 176]],[[468, -784, 355],[863, -598, -803],[195, -701, -573],[376, -539, 366]],[[68, 832, -935],[-322, 314, -210], [104, -131, -913],[-181, 912, 596]],[[1,0,1],[1, 1, 1],[0,0,1],[1,0,0]]]
# print(np.array(miu).shape)
# r, c0, c1 = enc_encrypt(A,B,np.array(miu), m=7, d =3, q=3)
# print("!!!R!!!!!!")

# message = []
# msg = []
# d = 3
# for i in range(k):
#     miu = np.random.choice([-1,1], d)
#     message.append(miu)
# print(miu)
# r, c0, c1 = enc_encrypt(A,B,np.array([miu]), m=4, d =10, q=3)
# 
def enc_decrypt(A,B,S,c0,c1, q,d=512):
    poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    # miu_prime = poly_add_3D(np.array(c1), np.array(-1*(poly_matmul(S, c0, q, poly_mod, d))), q, poly_mod, d) % 3 
    print("TEST enc_decrypt")
    miu_prime = np.fmod(np.array(c1) + np.array(-1*(poly_matmul(S, c0, q, poly_mod, d))), q)
    print(miu_prime)
    intermediate = np.array(c1) + np.array(-1*(poly_matmul(S, c0, q, poly_mod, d)))
    miu_prime = mod_3D(len(intermediate), intermediate, q)
    print(miu_prime)
    return miu_prime

# A = [[[0,1,2,2,1,0],[0,0,2,1,0,1],[2,0,0,2,0,1],[2,2,2,0,2,2]],[[1,0,1,1,0,2],[0,0,0,2,2,1],[1,1,0,0,0,1],[0,2,2,0,1,2]]]
# B =  [[[1,0,1,2,0,2],[0,2,0,0,0,1],[0,2,0,2,1,2],[0,0,1,1,0,1]]]
# S = [[[-859,209,-694,136,970,-137],[-437,178,260,-298,-652,-458]]]
# message = [[[-1,0,0,0,1,1]]]
# d=6
# k=1
# r, c0, c1 = enc_encrypt(A,B,np.array(message), m=4, d =d, q=3)
# print("!!!PARAMS22!!!!!!")
# print(r, c0, c1)


# A = [[[0,1,2,2,1,0],[0,0,2,1,0,1],[2,0,0,2,0,1],[2,2,2,0,2,2]],[[1,0,1,1,0,2],[0,0,0,2,2,1],[1,1,0,0,0,1],[0,2,2,0,1,2]]]
# B =  [[[1,0,1,2,0,2],[0,2,0,0,0,1],[0,2,0,2,1,2],[0,0,1,1,0,1]]]
# S = [[[-859,209,-694,136,970,-137],[-437,178,260,-298,-652,-458]]]
# r = [[[ 574,452,-325,-242,935,216]],[[-327,-434,-295,405,-23,-292]],[[-545,-716,672,679,955,420]],[[390,930,-766,656,-767,-766]]]
# c0 = [[[1,0,2,2,2,2]],[[2,0,1,1,1,1]]]
# c1 = [[[0,0,1,2,1,1]]]

# result = enc_decrypt(A,B,S, c0,c1,q=3,d=6)
# print(result)

from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
from numpy import linalg as LA
import math
import random 
q = 1437757
# x, y = enc_key_gen()
# message = 
# r_pke = enc_encrypt(x[0], x[1], message, q)
l= 2
tau = 7
d = 512
poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
# s1 = [r_pke]
# message = 
# for i in range(l*tau):
#     r = gen_matrix_ring(1,1,poly_mod,d)
#     s1.append(r)
# s1.append(message)
m = [0] #TODO: make into 0 matrix with correct dimensions
v_bin = 0
E_bin = [0,1]


def rejection_sampling_1(z,v,s,M=1.1):
    u = np.random.rand()
    a = -2* np.sum(np.inner(ring_transpose(z),ring_transpose(v)))
    b = LA.norm(v)**2
    d = (a+b)/(2*s**2)
    # print("rej_samp",u, float(((1/M) * np.exp(d))))
    if u > ((1/M) * np.exp(d)):
        # print("rej_samp",u, float(((1/M) * np.exp(d))))

        return 0 #reject
    else:
        return 1


def rejection_sampling_2(z,v,s,M=6):
    print(np.inner(z,v))
    if np.inner(z,v).all() < 0:
        print("inside rej 2")
        return 1
    print("REJECTION SAMPLING 2")
    return rejection_sampling_1(z,v,s,M)

def rejection_sampling_0_int(z,v,s,q,M=6):
    u = np.random.rand()
    z = np.array(z)
    # print(u,s)
    # print(type(z),type(v),type(s))
    # norm = norm2_3D_poly(z,z.shape[0],z.shape[1],q)%q
    # print("INSIDE REJ 0", s)
    # print(z.shape, v.shape, z,v)
    norm = LA.norm(v)
    # norm1 = norm2_3D_poly(v,v.shape[0], v.shape[1], q)
    # print(norm, norm1)
    # print(z.shape, v.shape)
    inner_prod = np.dot(z,v)
    # print("innerprod")
    # print(inner_prod, np.inner(z,v))
    # exit()
    # inner_prod = custom_fmod_2D(inner_prod, inner_prod.shape[0], inner_prod.shape[1],q)
    # print("REJ SAM, inner prod, norm")
    # print(custom_mod(inner_prod,q),q)
    # print(z, v)
    # print("compar")
    # print(custom_mod(inner_prod,q)/(s**2),custom_mod(inner_prod,q)/(s**2))
    # print(norm)
    # print(s)
    # exit()
    print((-1*norm**2)/2*(s**2), math.exp((-1*norm**2)/2*(s**2)), np.cosh((inner_prod/(s**2))))
    print((M*math.exp((-1*norm**2)/2*(s**2))*np.cosh((inner_prod/(s**2)))))
    if u > (1/(M*math.exp((-1*norm**2)/2*s**2)*np.cosh((inner_prod/s**2)))):
        print('If')
        return 1
    else:
        print("ELSE")
        return 0   

def rejection_sampling_0(z,v,s,q,M=6):
    u = np.random.rand()
    z = np.array(z)
    # print(u,s)
    print(type(z),type(v),type(s))
    print("rejection sampling________________________________________")
    print(z.shape, v.shape)
    norm = norm2_3D_poly(v,v.shape[0],v.shape[1],q)%q
    print("rejection sampling")
    inner_prod = np.fmod(np.sum(z*v),q)
    # inner_prod = mod_
    print("REJ SAM, inner prod, norm")
    print(inner_prod)
    # print(z, v)
    print(norm)
    print(s)
    # exit()
    print((-1*norm**2)/2*(s**2), math.exp((-1*norm**2)/2*(s**2)), np.cosh((inner_prod/(s**2))))
    print((M*math.exp((-1*norm**2)/2*(s**2))*np.cosh((inner_prod/(s**2)))))
    if u > (1/(M*math.exp((-1*norm**2)/2*s**2)*np.cosh((inner_prod/s**2)))):
        print('If')
        return 1
    else:
        print("ELSE")
        return 0  
# def apply_automorphism(sigma, v, exp):
#     for i in range(len(v)):
#         for j in range(exp):
#             v[i] = sigma*v[i]
#     return v

# def apply_automorphism_v2(sigma, v, exp):
#     for i in range(len(v)):
#         for j in range(exp):
#             v[i] = v[i]**sigma
#     return v

# def apply_automorphism_neg_1(sigma, v, dim, exp):
#     if len(v) < dim:
#         v = np.pad(v, (dim-len(v),0), 'constant', constant_values=0)
#     if exp % 2 == 0:
#         return v
#     new_v = []
#     for i in range(len(v)-2, -1, -1):
#         # print(v[i], i)
#         new_v.append(np.float_power(v[i],sigma))
#     new_v.append(v[-1])
#     return new_v

def apply_automorphism_neg_1(sigma, v, dim, exp):
    # print(v)
    if len(v) < dim:
        v = np.pad(v, (dim-len(v),0), 'constant', constant_values=0)
    if exp % 2 == 0:
        return v
    new_v = []
    for i in range(len(v)):
        # tmp = np.float_power(v[i],sigma)
        # value = np.polydiv(tmp,poly_mod)[1]
        # element = [custom_mod(num, q) for num in value]
        new_v.append(np.float_power(v[i],sigma))
    new_v.append(v[-1])
    print(new_v, np.array(new_v).shape)
    return new_v



# def apply_automorphism_v1(sigma, v, dim, exp, poly_mod, q):
#     print(v)
#     if len(v) < dim:
#         v = np.pad(v, (dim-len(v),0), 'constant', constant_values=0)
#     if exp % 2 == 0:
#         return v
#     new_v = []
#     for i in range(len(v)-2, -1, -1):
#         tmp = np.float_power(v[i],sigma)
#         value = np.polydiv(tmp,poly_mod)[1]
#         element = [custom_mod(num, q) for num in value]
#         new_v.append(np.float_power(v[i],sigma))
#     new_v.append(v[-1])
#     return new_v

# def apply_automorphism_v1(sigma, v, dim, exp, poly_mod):
#     print("Apply autonmorphism and v below")
#     print(v)
#     if len(v) < dim:
#         v = np.pad(v, (dim-len(v),0), 'constant', constant_values=0)
#     if exp % 2 == 0:
#         return v
#     new_v = []
#     for i in range(len(v)):
#         new_val = P.polypow(v[i], sigma)
#         print(new_val)
#         # value = np.float_power(v[i],sigma)
#         value = np.polydiv(new_val,poly_mod)[1]
#         new_v.append(value)
#     new_v.append(v[-1])
#     return new_v

# print("automorphism")
# print(apply_automorphism_neg_1(-1, [5,4,3,2,1], 5, 1))
# print(apply_automorphism_neg_1(-1, [5,4,3,2,1], 5, 2))
# print(apply_automorphism_neg_1(-1, [5,4,3,2,1], 5, 3))
# print(sdfs)
#fig 6

def apply_automorphism_once(automorphism, v, d):
    if not isinstance(automorphism, int):
        print("APPLY AUTOMORPHISM FUNCTION -- automorphism is not an integer")
        return -1 
    new_v = []

    temp_v = copy.deepcopy(v)
    for j in range(len(temp_v)):
        for p in range(len(temp_v[j])):
            temp_v[j][p][:-1] = np.array([num*automorphism for num in temp_v[j][p][:-1]])
            element = temp_v
    # print('here', element)
        new_v.append(element)
    new_v = np.reshape(np.array(new_v), (len(v),1,d))
    return new_v

def apply_automorphism(automorphism, v, k, d):
    if not isinstance(automorphism, int):
        print("APPLY AUTOMORPHISM FUNCTION -- automorphism is not an integer")
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
            # print('here', element)
        new_v.append(element)
    new_v = np.reshape(np.array(new_v), (k*len(v),1,d))
    return new_v

def fig_6_prove_step_1(s1, s2, sigma1, sigma2, poly_mod, nu=23, k=4, l = 0, sigma=-1, d=512, m1=581, m2=18, gamma1= 17, n=2, gamma2=1.2, q=(2**24 - 75)*(2**79 - 67), v=1):
    # s1_auto = []
    # m_auto = []
    # print(s1.shape)
    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    # for i in range(k):
    #     if m1 != 0: s1_auto.append(neg_poly_matmul(sigma, s1, m1, i, poly_mod,q))  #DC
    #     if l != 0: m_auto.append(apply_automorphism_v1(sigma, m, l, i,poly_mod,q))
    # # exit()
    # s1_auto = np.reshape(s1_auto, (k*m1,1,d))
    # m_auto = np.reshape(m_auto, (k*l,1,d))
    # s = np.concatenate((s1_auto, m_auto))

    s = apply_automorphism(sigma, s1, k, d)
    # print(s)
    # exit()
    # print("S SHAPE")
    # print(s.shape)
    # s = np.reshape(s1_auto, (k*m1,1,d))
    # sigma1 = gamma1* *23
    # sigma2 = gamma2*1*23*math.sqrt(18*512)

    # v = 
    # T = LA.norm(v)
    # sigma1 = gamma1*T
    # sigma2 = gamma2*T
    # sigma1 = gamma1*alpha*nu
    # sigma2 = gamma2*v*nu*math.sqrt(m2*d)
    A1 = gen_matrix_ring(n,m1,q,d)
    A2 = gen_matrix_ring(n,m2,q,d)
    B_mat = gen_matrix_ring(l, m2, q,d)
    # y1_sampler = DiscreteGaussianDistributionIntegerSampler(sigma=sigma1)
    # y2_sampler = DiscreteGaussianDistributionIntegerSampler(sigma=sigma2)
    # y1 = []
    # for i in range(m1):
    #     y1.append([[y1_sampler() for j in range(d)]]) 
    # y2 = []
    # for i in range(m2):
    #     y2.append([[y2_sampler() for j in range(d)]])
    # R = PolynomialRing(Integers(q), "X")
    # X = R.gen()
    # S = R.quotient(X**d + 1)
    # y1_sampler = DiscreteGaussianDistributionPolynomialSampler(S,d,sigma1)
    # y2_sampler = DiscreteGaussianDistributionPolynomialSampler(S,d,sigma2)
    # y1 = []
    # y2 = []
    # for i in range(m1):
    #     y1.append([list(map(int, y1_sampler().list()))]) 
    #     # print("Y1 type", type(y1_sampler().list()))
    # for i in range(m2):
    #     y2.append([list(map(int, y2_sampler().list()))]) 
    
    y1 = sample_discrete_gaussian_distribution(m1, q, d, sigma1)
    y2 = sample_discrete_gaussian_distribution(m2, q, d, sigma2)

    
    
    # bound = (6*sigma).floor() #TO DO: Is this bound okay?
    # norm_factor = sum([exp(-x^2/(2*sigma^2)) for x in range(-bound,bound+1)])

    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    # print("Y1--------------------------")
    # print(y1)
    # print(np.array(y1).shape)
    # print(y1)
    # print("--------------")
    # y1 = ring_transpose([y1])
    # y2 = ring_transpose([y2])
    # print(y1)
    # print("--------------")
    w1 = poly_matmul(A1, y1, q, poly_mod, d)  
    w2 = poly_matmul(A2, y2, q, poly_mod, d)
    # w = poly_add_3D(w1,w2,q, poly_mod, d)
    w = np.fmod(w1 + w2,q)
    w = mod_3D(len(w1+w2),w1 + w2,q)
    # y1_auto = []
    # By2_auto =[]
    # for i in range(k):
    #     if m1 != 0: y1_auto.append(apply_automorphism_neg_1(sigma, y1, m1, i)) #DC
    #     if l != 0:
    #         By2 = poly_matmul(B_mat, y2)
    #         By2_auto.append(apply_automorphism_neg_1(sigma, By2, i))
    # By2_auto = np.reshape(By2_auto, (k*l,1,d))
    # y1_auto = np.reshape(y1_auto, (k*m1,1, d))
    # y = np.concatenate((y1_auto, -1*By2_auto))
    # y = np.reshape(y1_auto, (k*m1,1, d))

    y = apply_automorphism(-1, y1, k, d)


    R2 = gen_matrix_ring(k*(m1+l), k*(m1+l),q,d)
    # print("-------s , R2-------")
    # print(np.array(s).shape, np.array(R2).shape)
    sR2 = poly_matmul(ring_transpose(s),R2,q,poly_mod,d)
    # sR2 = poly_matmul(ring_transpose(s), R2[:,0:m1*(1)],q,poly_mod, d)

    # for i in range(1,k):
    #     if i % 2 == 0:
    #         print(np.array(s).shape, np.array(R2).shape, np.array(R2[:,i*m1:m1*(i+1)]).shape)
    #         element = poly_matmul(ring_transpose(s), R2[:,i*m1:m1*(i+1)],q,poly_mod, d)
    #     else:
    #         element = neg_poly_matmul(ring_transpose(s), R2[:,i*m1:m1*(i+1)],q,poly_mod, d)
    #     print("HERE")
    #     print(np.array(element).shape)
    #     sR2 = np.concatenate((sR2,element))
    R2s = poly_matmul(R2,s, q,poly_mod, d)
  
    # R2s = poly_matmul(R2[:,0:m1*(1)],s1, q,poly_mod, d)
    # for i in range(1,k):
    #     if i % 2 == 0:
    #         element = poly_matmul(R2[:,i*m1:m1*(i+1)],s1, q,poly_mod, d)
    #     else:
    #         element = neg_poly_matmul_right(R2[:,i*m1:m1*(i+1)],s1, q,poly_mod, d)
    #     R2s = np.concatenate((R2s,element))

    # sR2y = poly_matmul(sR2,y,q,poly_mod,d)
    # print(np.array(sR2.shape))
    sR2y = poly_matmul(sR2 ,y, q,poly_mod, d)
    # for i in range(1,k):
    #     if i % 2 == 0:
    #         element = poly_matmul(sR2[:,i*m1:m1*(i+1)],y, q,poly_mod, d)
    #     else:
    #         element = neg_poly_matmul_right(sR2[:,i*m1:m1*(i+1)],y, q,poly_mod, d)
    #     sR2y = np.concatenate((sR2y,element))      

    yR2s = poly_matmul(ring_transpose(y), R2s, q,poly_mod, d)
    # for i in range(1,k):
    #     if i % 2 == 0:
    #         element = poly_matmul(ring_transpose(y), R2s[:,i*m1:m1*(i+1)], q,poly_mod, d)
    #     else:
    #         element = neg_poly_matmul(ring_transpose(y), R2s[:,i*m1:m1*(i+1)], q,poly_mod, d)
    #     yR2s = np.concatenate((yR2s,element))    

    # print("-------sR2 , y-------")
    # print(np.array(sR2).shape, np.array(y).shape)
    # sR2y = poly_matmul(sR2,y,q,poly_mod,d)
    # print("-------y , R2, s-------")
    # print(np.array(y).shape, np.array(R2).shape, np.array(s).shape)
    # yR2s = poly_matmul(ring_transpose(y),R2s, q,poly_mod,d)
    r1 = gen_matrix_ring(k*(m1+l),1,q,d)
    r1y = poly_matmul(ring_transpose(r1),y,q,poly_mod,d)
    # r1y = neg_poly_matmul_right(ring_transpose(r1),y,q,poly_mod,d)
    # g1 = poly_add_3D(poly_add_3D(sR2y, yR2s, q, poly_mod, d), r1y, q, poly_mod, d)
    g1 = np.fmod(sR2y + yR2s + r1y,q)
    b_vec= gen_matrix_ring(m2,1,q, d)
    # s2 = gen_matrix_ring_with_norm(m2, 1, q, d, nu)
    # t = poly_add_3D(poly_matmul(ring_transpose(b_vec),s2, q, poly_mod, d), g1, q, poly_mod, d)
    t = np.fmod(poly_matmul(ring_transpose(b_vec),s2, q, poly_mod, d) + g1, q)
    # yR2y = poly_matmul(poly_matmul(ring_transpose(y),R2,q,poly_mod,d),y,q,poly_mod,d)
    # yR2y = poly_matmul(poly_matmul(ring_transpose(y),R2,q,poly_mod,d),y,q,poly_mod,d)
    
    yR2 = poly_matmul(ring_transpose(y),R2, q,poly_mod, d)
    # for i in range(1, k):
    #     if i % 2 == 0:
    #         element = poly_matmul(ring_transpose(y),R2[:,i*m1:m1*(i+1)], q,poly_mod, d)
    #     else:
    #         element = neg_poly_matmul(ring_transpose(y), R2[:,i*m1:m1*(i+1)], q,poly_mod, d)
    #     yR2 = np.concatenate((yR2,element))     
    yR2y = poly_matmul(yR2, y, q,poly_mod, d)
    # for i in range(1,k):
    #     if i % 2 == 0:
    #         element = poly_matmul(yR2[:,i*m1:m1*(i+1)], y, q,poly_mod, d)
    #     else:
    #         element = neg_poly_matmul_right(yR2[:,i*m1:m1*(i+1)], y, q,poly_mod, d)
    #     yR2y = np.concatenate((yR2y,element))   

    by2 = poly_matmul(ring_transpose(b_vec),y2,q,poly_mod,d)
    # v = poly_add_3D(yR2y, by2 ,q,poly_mod,d)
    v = np.fmod(yR2y+ by2 ,q)
    return b_vec, s,w,t,v,y1,y2,A1,A2, R2, r1

##################################
#FIG 6 PROVE 1 TESTING FUNCTIONS
# d = 10
# r = generate_random(1,112, d, 100000000000)
# print("r gen")
# message = gen_matrix_ring(1,1,1,d)
# print("msg gen")
# A, B, S = enc_key_gen(d=d)
# print("keygen")
# miu = np.concatenate((r, message), axis=1)
# print("miu gen")
# r_pke, c0, c1 = enc_encrypt(A, B, miu,d=d)
# print("rpke gen")
# print(np.array(r_pke).shape, np.array(miu).shape)
# s1 = np.concatenate((r_pke, r_pke, r_pke, r_pke, ring_transpose(miu)))
# print("s1 concat")
# print(s1.shape)
# # print(LA.norm(ring_transpose(s1)[0],2))
# alpha = LA.norm(ring_transpose(s1)[0],2)
# result = fig_6_prove_step_1(s1=s1, alpha = alpha,d=d)
# # result = fig_6_prove_step_1(s1=s1, alpha = 500, nu=1, k=1, l = 0, sigma=-1, d=512, m1=581, m2=18, gamma1= 17, n=23, gamma2=1.2, q_LNP=2**103)
# print("result")
# print(result)
#################################
#HERE
# r = generate_random(4,7,8,20000000)
# print("r gen")
# message = gen_matrix_ring(1,1,1000, 2)
# print("msg gen")
# A, B, S = enc_key_gen(2,248)
# print("keygen")
# print(r.shape, message.shape)
# print(r, message)
# print(r.shape)
# miu = np.concatenate((np.reshape(r, (1,28*4,2)), message), axis=1)
# miu = np.transpose(miu, axes=[1,0,2])

# def ring_transpose(a):
#     transpose = []
#     a = np.array(a)
#     print(a.shape)
#     for i in range(a.shape[1]):
#         new_row = []
#         for j in range(a.shape[0]):
#             new_row.append(a[j][i])
#         transpose.append(new_row)
#     return transpose

# a = [[[1,0,1], [1,1,1]], [[3,2,3], [3,3,3]]]
# b = ring_transpose(a)
# print(b)

# miu = np.array(ring_transpose(miu))
# print("miu gen")
# r_pke, c0, c1 = enc_encrypt(A, B, miu, A.shape[1], 2, 100000)
# print("rpke gen")
# print(np.array(r_pke).shape, np.array(miu).shape)
# s1 = np.concatenate((r_pke, miu))
# print(np.array(s1).shape)
# print("s1 concat")
# result = fig_6_prove_step_1(s1=s1, alpha = 500, nu=1, k=1, l = 0, sigma=-1, d=2, m1=230, m2=18, gamma1= 17, n=23, gamma2=1.2, q_LNP=103)
# print("result")
# print(result)

# r = generate_random(4,7,8,20000000)
# print("r________________________")
# print(repr(r))
# message = gen_matrix_ring(1,1,1000, 2)
# print("message________________________")
# print(repr(message))
# A, B, S = enc_key_gen(2,248)
# miu = np.concatenate((np.reshape(r, (1,28*4,2)), message), axis=1)
# miu = np.array(ring_transpose(miu))
# print("miu________________________")
# print(repr(miu))
# r_pke, c0, c1 = enc_encrypt(A, B, miu, A.shape[1], 2, 100000)
# print("r_pke________________________")
# print(repr(r_pke))
# s1 = np.concatenate((r_pke, miu))
# print("s1________________________")
# print(repr(s1))
# w,t,v = fig_6_prove_step_1(s1=s1, alpha = 500, nu=1, k=1, l = 0, sigma=-1, d=2, m1=230, m2=18, gamma1= 17, n=23, gamma2=1.2, q_LNP=103)
# print("result________________________")
# print(w)
# print(t)
# print(v)

def sample_challenge(d, k,q, n=23):
    poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    c = [random.randrange(-k,k+1) for i in range(d)]
    # S = poly.polydiv(c, poly_mod)[1]
    # rand = LA.norm(S, np.inf)
    exp = [i**k for i in c]
    # tmp = apply_automorphism_neg_1(1, exp,1,1) #TODO
    # print(np.array(exp).shape, np.array(tmp).shape)
    print(d, k , q, "SAMPLE CHALLENGE THANGS", exp)
    tmp_val = LA.norm(neg_poly_matmul([[exp]], [[exp]] , q, poly_mod,d )[0][0],1)
    val = tmp_val**(1/(2*k))
    # print(val, n)
    while val > n:
        c = [random.randrange(-k, k+1) for i in range(d)]
        exp = [i**k for i in c]
        tmp = apply_automorphism_neg_1(-1, exp,1,1)
        print("TMP")
        print(tmp)
        exit()
        tmp_val = LA.norm(poly_matmul([[tmp]],[[exp]],q, poly_mod,d )[0][0],1)
        val = tmp_val**(1/(2*k))
    # print(rand, val)
    return c

import copy
def sample_challenge_v2(d, kappa, q, nu=23):
    poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    val = nu + 1
    c = []
    while val > nu:
        print('in')
        half_c = [random.randrange(-kappa,kappa+1) for i in range(int(d/2))]
        print(half_c)
        other_half_c = copy.deepcopy(half_c)
        c = [num*-1 for num in np.flip(other_half_c[:-1])] + [0] + half_c
        sigma_c = [num*-1 for num in c[:-1]]
        sigma_c.append(c[-1])
        c_k = [i**k for i in c]
        sigma_c_k = [i**k for i in sigma_c]
        print(half_c)
        print(c)
        product = poly_matmul([[sigma_c_k]],[[c_k]],q, poly_mod,d)
        print(product)
        tmp_val = np.sum(np.abs(product[0][0]))
        val = tmp_val**(1/(2*k))
    return [[c]]

    # print(c)
    # # print(halfc)
    # exit()
    # S = poly.polydiv(c, poly_mod)[1]
    # rand = LA.norm(S, np.inf)
    # exp = [i**k for i in c]
    # tmp = apply_automorphism_neg_1(1, exp,1,1) #TODO
    # print(np.array(exp).shape, np.array(tmp).shape)
    # print(d, k , q, "SAMPLE CHALLENGE THANGS", exp)
    # tmp_val = LA.norm(neg_poly_matmul([[exp]], [[exp]] , q, poly_mod,d )[0][0],1)
    # val = tmp_val**(1/(2*k))
    # print(val, n)
    while val > n:
        c = [random.randrange(-k, k+1) for i in range(d)]
        exp = [i**k for i in c]
        tmp = apply_automorphism_neg_1(-1, exp,1,1)
        print("TMP")
        print(tmp)
        exit()
        tmp_val = LA.norm(poly_matmul([[tmp]],[[exp]],q, poly_mod,d )[0][0],1)
        val = tmp_val**(1/(2*k))
    # print(rand, val)
    return c

# def sample_challenge(d, k,q, n=23):
#     poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
#     c = [random.randrange(-k,k+1) for i in range(d)]
#     # S = poly.polydiv(c, poly_mod)[1]
#     # rand = LA.norm(S, np.inf)
#     exp = [i**k for i in c]
#     tmp = apply_automorphism_neg_1(1, exp,1,1) #TODO
#     print(np.array(exp).shape, np.array(tmp).shape)
#     tmp_val = LA.norm(poly_matmul([[tmp]],[[exp]],q, poly_mod,d )[0][0],1)
#     val = tmp_val**(1/(2*k))
#     # print(val, n)
#     while val > n:
#         c = [random.randrange(-k, k+1) for i in range(d)]
#         exp = [i**k for i in c]
#         tmp = apply_automorphism_neg_1(-1, exp,1,1)
#         print("TMP")
#         print(tmp)
#         exit()
#         tmp_val = LA.norm(poly_matmul([[tmp]],[[exp]],q, poly_mod,d )[0][0],1)
#         val = tmp_val**(1/(2*k))
#     # print(rand, val)
#     return c

# def sample_challenge(d,max_coefficient, k,q, n=23):
#     poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
#     c = [random.randrange(-max_coefficient, max_coefficient+1) for i in range(d)]
#     # S = poly.polydiv(c, poly_mod)[1]
#     # rand = LA.norm(S, np.inf)
#     exp = [i**k for i in c]
#     tmp = apply_automorphism_neg_1(-1, exp,1,1)
#     tmp_val = LA.norm(poly_matmul([[tmp]],[[exp]],q, poly_mod,d )[0][0],1)
#     val = tmp_val**(1/(2*k))
#     print(val, n)
#     while val > n:
#         c = [random.randrange(-k, k+1) for i in range(d)]
#         exp = [i**k for i in c]
#         tmp = apply_automorphism_neg_1(-1, exp,1,1)
#         tmp_val = LA.norm(poly_matmul([[tmp]],[[exp]],q, poly_mod,d )[0][0],1)
#         val = tmp_val**(1/(2*k))
#     # print(rand, val)
#     return c

# def alt_sample_challenge(d,max_coefficient, k,q, n=23):
#     poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
#     c = [random.randrange(-max_coefficient, max_coefficient+1) for i in range(d)]
#     # S = poly.polydiv(c, poly_mod)[1]
#     # rand = LA.norm(S, np.inf)
#     exp = np.poly1d(c)**k
#     s = np.polydiv(exp, poly_mod)[1]
#     if len(s) < d:
#         end = np.pad(s, (d-len(s.c),0), 'constant', constant_values=0)
#     # print(k, c, exp, s.c, end)
#     # exit()
#     tmp = apply_automorphism_neg_1(-1, end,1,1)
#     tmp_val = LA.norm(poly_matmul([[tmp]],[[end]],q, poly_mod,d )[0][0],1)
#     val = tmp_val**(1/(2*k))
#     while val > n:
#         tmp_val = LA.norm(poly_matmul([[tmp]],[[end]],q, poly_mod,d )[0][0],1)
#         val = tmp_val**(1/(2*k))
#     # print(rand, val)
#     print(c, val)
#     exit()
#     return c

# result = sample_challenge(5, 2,55)
# print("SAMPLE CHALLENGE", result)

def get_challenge(d,k,q,n):
    c_1 = sample_challenge(d,k,q,n)
    c_2 = sample_challenge(d,k,q,n)
    poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    for i in range(len(c_1)):
        if c_1[i] != c_2[i]:
            print(np.array(c_1).shape, np.array(c_2).shape)
            return poly_add_2D([c_1], [np.array(c_2)*-1], q, poly_mod, d)
    # for i in range(len(c_1)):
    #     for j in range(len(c_1[0])):
    #         for k in range(len(c_1[0][0])):
    #             if c_1[i][j][k] != c_2[i][j][k]:
    #                 return poly_add_2D(c_1, -1*c_2)
    return False

# result = get_challenge(123, 12,18, 500)
# print("GET CHALLENGE", result)

def fig_6_prove_step_2(c,s1,s2,y1,y2, sigma1, sigma2, q, d, poly_mod):
    # print("INNNNNNNNNNNNNNNNNNNNNNNN", sigma1, sigma2)
    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    # print("FIG 6 PROVER 2 ------------------------------")
    # print(np.array(s1).shape , np.array(y1).shape)
    # exit()
    print("IN FIG 6 Prove step 2")
    # print(np.array(s1).shape)
    # print(np.array(c).shape)
    # print(np.array(s2).shape)
    # print(np.array(y1).shape)
    # print(np.array(y2).shape)
    # print(poly_matmul(np.array(c),np.array(s1)),q,poly_mod,d)
    # exit()
    z1 = np.fmod(ring_transpose(np.array(poly_matmul(np.array(c),ring_transpose(np.array(s1)),q,poly_mod,d))) + np.array(y1), q)
    z2 = np.fmod(ring_transpose(np.array(poly_matmul(np.array(c),ring_transpose(np.array(s2)),q,poly_mod,d))) + np.array(y2),q)
    # print(z1.shape, poly_matmul(np.array(c),np.array(s1),q,poly_mod,d).shape)
    # print(garb)
    # print("Z1 shape --------------------------------")
    # print(np.array(z1).shape)
    # print("v --------------------------------")
    # a = np.array(c)
    # b = ring_transpose(np.array(s1))
    # mul= poly_matmul(a,b,q,poly_mod,d)
    # print("v --------------------------------")
    # print(a.shape, np.array(b).shape, np.array(mul).shape)

    rej1_res = rejection_sampling_1(z1, ring_transpose(poly_matmul(np.array(c),ring_transpose(np.array(s1)),q,poly_mod,d)), sigma1)
    # print("INSIDE FIG 6 PROVE STEP 2, shapes")
    # print(np.array(z2).shape,  np.array(ring_transpose(poly_matmul(np.array(c),np.array(s2),q,poly_mod,d))).shape)
    rej2_res = rejection_sampling_2(z2, ring_transpose(poly_matmul(np.array(c), ring_transpose(np.array(s2)),q,poly_mod,d)),sigma2)
    # print(rej1_res, rej2_res)
    # return z1, z2 #TODO
    # print("****************************************************", z1, z2)
    # exit()
    if rej1_res == 1 or rej2_res ==1:
        print("FAILED 2088")
        # exit()
        # print(z1,z2)
        print(rej1_res, rej2_res)
        return -1, -1 
    else:
        return z1, z2

# #HERE FOR THE FIG 6 STUFF
# d= 100
# # c = get_challenge(d, 12,18, d)
# c= sample_challenge(d,2,500)
# r = generate_random(16,7,d,1)
# # print(r)
# message = gen_matrix_ring(1,1,1000, d)
# A, B, S = enc_key_gen(d=d)
# miu = np.concatenate((np.reshape(r, (1,28*4,d)), message), axis=1)
# miu = np.array(ring_transpose(miu))
# # print(B.shape, r.shape)
# # print(brk)
# r_pke, c0, c1 = enc_encrypt(A, B, miu, d=d)# A.shape[1], 2, 100000)
# s1 = np.concatenate((r_pke, miu))
# gamma1 = 17
# alpha = 500
# gamma2 = 1.2
# vi = 1
# n =23
# m2 = 18
# m1 = 117 +112 +1
# # m1 = 581
# # d = 2
# q = 1000
# w, t,v, y1, y2 = fig_6_prove_step_1(s1=s1, m1 =m1 ,alpha = alpha, d=d)
# sigma1 = gamma1*alpha*n
# sigma2 = gamma2*vi*n*math.sqrt(m2*d)
# print("SIGMAAAAAS", sigma1, sigma2)
# s2 = gen_matrix_ring_with_norm(m2,1, q,d,1)
# print("-----------")
# print(np.array(c).shape, np.array(s1).shape)
# print(c)
# print("CHALLEGE ABOVE")
# result = fig_6_prove_step_2([[c]],s1,ring_transpose(s2),y1,y2, sigma1, sigma2,q,d)
# print("-----------")
# print(result)
# print(ex)
# from numpy import linalg as LA
def check_zkp_1(A1,A2,w,t,v,c,z1,z2,std1,std2,m1,m2,l,d,t1,t2,b,r0,r1, R2,m,s2,q,poly_mod,k=4):
    # print("R2------------------", R2)

    # z1_auto = []
    # z2_auto =[]
    # # t_b = poly_add_3D(poly_matmul(B,s2, q, poly_mod, d), m, q, poly_mod, d)
    # # temp1 = poly_matmul(c, t_b, q,poly_mod,d)
    # # temp2 = -1*poly_matmul(B, z2,q,poly_mod,d)
    # # second = poly_add_3D(temp1, temp2, q, poly_mod,d)

    # for i in range(k):
    #     z1_auto.append(apply_automorphism_neg_1(-1, z1,m1, i))
    #     # val = poly_matmul(np.array(c),np.array(t2),q, poly_mod, d)
    #     val = np.array(c)
    #     # val2 = poly_matmul(B, z2, q, poly_mod, d)
    #     val2 = np.array(z2) #, q, poly_mod, d)
    #     # res = poly_add_3D(val, -1*val2, q, poly_mod, d)
    #     res = val2
    #     z2_auto.append(apply_automorphism_neg_1(-1, res,l, i))
    # z1_auto = np.reshape(z1_auto, (k*m1,1,d))
    # z2_auto = np.reshape(z2_auto, (k*l,1,d))
    # z = np.concatenate((z1_auto, z2_auto), axis=1)
    # z = z1_auto

    z = apply_automorphism(-1, z1, k, d)
    # print("Z---------------------", z)
    # print("r1----------------------", r1)
    # print("r0------------------------", r0)
    # print("v------------------------", v )
    # print("HERE", np.array(c).shape, np.array(t).shape, np.array(b).shape, np.array(z2).shape)
    # print( poly_matmul(ring_transpose(b), z2, q, poly_mod,d))
    # print( poly_matmul(ring_transpose(b), z2, q, poly_mod,d)*-1)
    # exit()
    # f = poly_add_3D(poly_matmul(c,t, q, poly_mod, d), poly_matmul(ring_transpose(b), z2, q, poly_mod,d)*-1, q, poly_mod, d)
    f = np.fmod(poly_matmul(c,t, q, poly_mod, d) + poly_matmul(ring_transpose(b), z2, q, poly_mod,d)*-1, q)
    # print("f------------------------", f)
    # exit()
    # print(np.array(z1).shape)
    # norm_z1 = 0
    # for i in range(len(z1)):
    #     norm = LA.norm(z1[i][0])
    #     norm_z1 = norm_z1 + norm**2
    # norm_z1 = math.sqrt(norm_z1)
    # print(z1.shape)
    new_list = [LA.norm(poly[0])**2 for poly in z1]
    norm_z1 = math.sqrt(np.sum(new_list))

    new_list = [LA.norm(poly[0])**2 for poly in z2]
    norm_z2 = math.sqrt(np.sum(new_list))
    # print("COMPARE NORMS")
    # print(new_list)
    # print(norm_z1)
    # print(np.sum(np.inner(ring_transpose(z1),ring_transpose(z1))))
    # exit()
    # if np.sum(np.inner(ring_transpose(z1),ring_transpose(z1))) > std1*math.sqrt(2*m1*d):
    if norm_z1 > std1*math.sqrt(2*m1*d):

        print("inner product",np.sum(np.inner(ring_transpose(z1),ring_transpose(z1))))
        print("math sqrt",std1*math.sqrt(2*m1*d))
        print("FIRST IF")
        return False
    # print(LA.norm(ring_transpose(z1)[0],2), std1*math.sqrt(2*m1*d))
    # if np.sum(np.inner(ring_transpose(z2),ring_transpose(z2)))  > std2*math.sqrt(2*m2*d):
    if norm_z2  > std2*math.sqrt(2*m2*d):

        print("inner product",np.sum(np.inner(ring_transpose(z2),ring_transpose(z2))))
        print("math sqrt",std2*math.sqrt(2*m2*d))
        print("SECOND IF")
    # print(LA.norm(ring_transpose(z2)[0],2),std2*math.sqrt(2*m2*d))
        return False
    # lhs = poly_add_3D(poly_matmul(A1,z1, q,poly_mod, d), poly_matmul(A2,z2, q, poly_mod,d), q, poly_mod,d)
    lhs = np.fmod(poly_matmul(A1,z1, q,poly_mod, d)+ poly_matmul(A2,z2, q, poly_mod,d), q)
    # rhs = poly_add_3D(w, ring_transpose(poly_matmul(np.array(c),ring_transpose(np.array(t1)),q,poly_mod,d)), q, poly_mod,d)
    rhs = np.fmod(w + ring_transpose(poly_matmul(np.array(c),ring_transpose(np.array(t1)),q,poly_mod,d)), q)
    # print("w", w)
    # print("ct1",  ring_transpose(poly_matmul(np.array(c),ring_transpose(np.array(t1)),q,poly_mod,d)))

    # print("A1z1", poly_matmul(A1,z1, q,poly_mod, d))
    # print("A2z2", poly_matmul(A2,z2, q,poly_mod, d))

    # print("EQ 3")
    # print(lhs,rhs)
    for i in range(len(lhs)):
        for j in range(len(lhs[0])):
            for k in range(len(lhs[0][0])):
                if lhs[i][j][k]%q != rhs[i][j][k] %q:
                    print("THIRD CHECK FAIL")
                    return False
    zR2 = poly_matmul(ring_transpose(z),R2, q, poly_mod,d)
    zR2z = poly_matmul(zR2, z,q, poly_mod,d)
    cR1 =poly_matmul(c,ring_transpose(r1), q, poly_mod,d)
    cR1z =poly_matmul(cR1,z,q, poly_mod,d) 
    # step5=poly_add_3D(zR2z, cR1z, q, poly_mod,d)
    step5=np.fmod(zR2z+ cR1z, q)
    cc=poly_matmul(c,c,q,poly_mod,d)
    ccr0=poly_matmul(cc,r0, q, poly_mod,d)
    # step7= poly_add_3D(ccr0, -1*f, q, poly_mod,d)
    step7 =np.fmod(ccr0 + -1*f, q)
    # lhs2 = poly_add_3D(step5,step7, q, poly_mod,d)
    lhs2= np.fmod(step5+step7, q)
    print("EQ 4")
    print(lhs2,v)    
    for i in range(len(lhs2)):
        for j in range(len(lhs2[0])):
            for k in range(len(lhs2[0][0])):
                if lhs2[i][j][k]%q != v[i][j][k]%q:
                    print("FOURTH CHECK")
                    return False
    return True

# print("----------------------FIG 6 VERIFIER CHECKS----------------------")

# q= 10000
# d = 10
# poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
# c= sample_challenge(d,2,500)
# r = generate_random(16,7,d,1)
# message = gen_matrix_ring(1,1,1000, d)
# A, B, S = enc_key_gen(d=d)
# miu = np.concatenate((np.reshape(r, (1,28*4,d)), message), axis=1)
# miu = np.array(ring_transpose(miu))
# r_pke, c0, c1 = enc_encrypt(A, B, miu, d=d)
# s1 = np.concatenate((r_pke, miu))
# gamma1 = 17
# alpha = 500
# gamma2 = 1.2
# vi = 1
# n =23
# m2 = 18
# m1 = 117 +112 +1
# w, t,v, y1, y2, A1, A2,R2,r1 = fig_6_prove_step_1(s1=s1, m1 =m1 ,alpha = alpha, d=d)
# sigma1 = gamma1*alpha*n
# sigma2 = gamma2*vi*n*math.sqrt(m2*d)
# s2 = gen_matrix_ring_with_norm(m2,1, q,d,1)
# print("-----------")
# print(np.array(c).shape, np.array(s1).shape)
# print(c)
# print("CHALLEGE ABOVE")
# z1, z2 = fig_6_prove_step_2([[c]],s1,ring_transpose(s2),y1,y2, sigma1, sigma2,q,d)

# # A1 = gen_matrix_ring(n, m1, q, d)
# # A2 = gen_matrix_ring(n, m2, q, d)
# t1 = poly_add_3D(poly_matmul(A1, s1, q, poly_mod,d), poly_matmul(A2, s2, q, poly_mod,d), q, poly_mod,d)
# # B_ = gen_matrix_ring(l, m)
# # t2 = poly_matmul(B_, s2, q, poly_mod,d) 
# t2 = [[[]]]
# b = gen_matrix_ring(m2, 1, q, d)
# r0 = gen_matrix_ring(1,1,q, d)
# mmm = check_zkp_1(A1,w,t,v,[[c]],z1,z2,sigma1,sigma2,m1,m2,l,d,t1,t2,b,r0,r1, R2,m,s2,q,poly_mod)
# print(mmm)
# print(ex)

# print("END FIG 6 CHECK")


# def prove_2(c, s1, s2, y1, y2):
#     z1 = c*s1 + y1
#     z2 = c*s2 + y2
#     x = rejection_sampling_1(z1, c*s1, s1_prime)
#     y = rejection_sampling_2(z2,c*s2,s2_prime)
#     if x == 1:
#         z1 = None
#     if y == 1:
#         z2 = None
#     return z1,z2
    
def fig_7_verifier(q,d,N):
    # mus = []
    # for i in range(N):
    #     mu = gen_matrix_ring(1,1,q,d)
    #     mus.append(mu[0])
    # print(np.array(mus).shape)
    mus = [np.random.randint(-q+1,q, size = (1,d)) for i in range(N)]
    # print(np.array(mus).shape)
    # exit()
    return mus

def fig_7_prover(N, mus,fs,q,poly_mod,d):
    # print("INSIDE")
    # print(mus.shape, fs.shape)
    f = poly_matmul([[mus[0][0]]],[[fs[0][0]]],q,poly_mod,d)
    # print("FSHAPE",f.shape)
    for i in range(1,N):
        # print("FOR LOOP",np.array(mus[i]).shape, np.array([[fs[0][0]]]).shape)
        # print("SUM OF MU, FS", [[mus[i][0]]],[[fs[0][i]]])
        # f = poly_add_3D(f, poly_matmul([[mus[i][0]]],[[fs[0][i]]],q,poly_mod,d),q, poly_mod, d)
        f = np.fmod(f+ poly_matmul([[mus[i][0]]],[[fs[0][i]]],q,poly_mod,d),q)
        # print("SUMMMMMMM", f)
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

# def convert_R_to_S_k(r,n,m, N, d, k):
#     s = []
#     for i in range(n):
#         row = []
#         for j in range(m):
#             for l in range(0,k):
#                 poly = []
#                 for o in range(0,N,k):
#                     s_i = r[i][j][l+o]
#                     poly.append(s_i)
#                 row.append(poly)
#         s.append(row)
#     return np.array(s)
# mus = fig_7_verifier(2**103, 512, 4*581)

# print("--------------F/IG 7 TESTS-----------------")
# d= 512
# q = 15
# q_pke = 10
# m1 = 581
# N= 512
# # N=2048
# rand = generate_random(4,7,N,1) 
# message = gen_matrix_ring(1,1,q,N)
# A,D, q1, q2, N, k_prime, gamma, gamma_prime, alpha = cts_setup(n=1, k=4 ,l=1, l_prime=2,q1=75,q2=67,N=N)
# poly_mod_N = np.poly1d([1] + [0]*(N-1) + [1])
# poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
# comm = cts_commit(A, message, rand, N, poly_mod_N, q1, q2, 1, 1)
# fs = np.reshape(comm,(1,14,N))
# # print(comm.shape)
# # print(exitttt)
# # for i in range():

# result = fig_7_prover(14, mus, fs,q,poly_mod,d)
# print(result)

# def apply_automorphism(sigma, v, exp):
#     for i in range(len(v)):
#         for j in range(exp):
#             v[i] = sigma*v[i]
#     return v

def fig_8_prover_1(sigma, s1, lmda, q,d, s2, poly_mod, k = 4,m1 = 581,m2 = 18, l = 1):
    # s1_auto = []
    # m_auto = []
    s = apply_automorphism(sigma, s1, k, d)
    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    print("IN FIG 8 PROVER")
    # print(np.array(s1).shape)
    # print(k*m1,1,d)
    # for i in range(k):
        # s1_auto.append(apply_automorphism_neg_1(sigma, s1,m1 ,i))
    # s = np.reshape(s1_auto, (k*m1,1,d))
    g = gen_matrix_ring(lmda,1, q,d)
    # print("pre change g", g)
    for i in range(lmda):
        g[i][0][-1] = 0
    Bg = gen_matrix_ring(lmda, m2, q, d)
    tg = poly_add_3D(poly_matmul(Bg, s2, q, poly_mod, d), g, q, poly_mod, d)
    # tg = np.fmod(poly_matmul(Bg, s2, q, poly_mod, d)+ g, q)
    return s, g, tg

# r = generate_random(4,7,12,1)
# message = gen_matrix_ring(1,1,1000, 3)
# A, B, S = enc_key_gen(3,248)
# miu = np.concatenate((np.reshape(r, (1,28*4,3)), message), axis=1)
# miu = np.array(ring_transpose(miu))
# print("PARAMS", A.shape, B.shape, miu.shape)
# r_pke, c0, c1 = enc_encrypt(A, B, miu, A.shape[1], 3, 100000)
# s1 = np.concatenate((r_pke, miu))
# s2 = gen_matrix_ring_with_norm(4,1,21,3,3000)
# tg = fig_8_prover_1(-1, s1, 14, 21, 3, s2,k = 4,m1 = 230,m2 = 4, l = 1 )
# print("RESULT~~~~~~~~~~")
# # print(g)
# print(tg)
# print(g)

def fig_8_verifier_1(q, lmda, M=1):
    gamma = [random.randrange(-q+1, q) for i in range(lmda*M)]
    gamma = np.reshape(gamma,(lmda,M))
    return gamma

# m1 = 230
# k = 4
# r = generate_random(4,7,12,1)
# message = gen_matrix_ring(1,1,1000, 3)
# A, B, S = enc_key_gen(3,248)
# miu = np.concatenate((np.reshape(r, (1,28*4,3)), message), axis=1)
# miu = np.array(ring_transpose(miu))
# print("PARAMS", A.shape, B.shape, miu.shape)
# r_pke, c0, c1 = enc_encrypt(A, B, miu, A.shape[1], 3, 100000)
# s1 = np.concatenate((r_pke, miu))
# s2 = gen_matrix_ring_with_norm(4,1,21,3,3000)
# tg = fig_8_prover_1(-1, s1, 14, 21, 3, s2,k = 4,m1 = 230,m2 = 4, l = 1 )
# gamma = fig_8_verifier_1(q,14,k*m1)
# print("Fig 8 prover and verifier RESULT~~~~~~~~~~")
# # print(g)
# print(gamma)
# print(g)


# def neg_poly_multiplication(neg_poly, neg_poly_exp, poly, poly_exp, poly_mod, d):
#     print(np.array(neg_poly).shape, np.array(neg_poly_exp).shape, np.array(poly).shape, np.array(poly_exp).shape)
#     res = np.array([0]*(d*2 +1))
#     print("LENGTHS", len(neg_poly), len(poly), d*d-1)
#     for i in range(len(neg_poly)):
#         # temp = np.array([0]*(len(poly)*len(neg_poly)))
#         temp = np.array([0]*(d*2))
#         print("temp", temp)
#         for j in range(d+1):
#             # print(poly_exp)
#             print("---------",i,j,neg_poly_exp[i], poly_exp[j])
#             exp = neg_poly_exp[i] + poly_exp[j] + d 
#             print("indices",exp)
#             # print(temp[exp])
#             temp[exp] = temp[exp] + neg_poly[i]*poly[j]
#         print("temp all",temp)
#         # temp_neg_poly = np.poly1d(temp[:len(poly)])
#         # temp_poly = np.poly1d(temp)
#         # print("temp poly", temp_poly)
#         res = res + temp
#     res_neg = np.flip(np.append(res[:d], [0]))
#     res_pos = res[d:]
#     print("RES NEG RES POS")
#     print(res_neg, res_pos)
#     print(np.poly1d(res_neg), np.poly1d(res_pos))
#     element_neg = np.polydiv(res_neg, poly_mod)[1]*-1
#     element_neg[0] = element_neg[0]*-1
#     element_pos = np.polydiv(res_pos,poly_mod)[1]
#     print("HERE", element_neg, element_pos)
#     # print(res, np.poly1d(element))
#     # print("---------")
#     # print(element, len(np.array(element)))
#     element = element_neg + element_pos
#     # print("ELEMENT", element)
#     if len(np.array(element)) < d:
#         element = np.pad(element, (d-len(np.array(element)),0), 'constant', constant_values=0)  
#     # print(element)
#     # print(np.array(element).shape, res.shape)
#         # res = res + element
#     # print("RES")
#     print(np.poly1d(element))
#     return element


# def neg_poly_multiplication(neg_poly, neg_poly_exp, poly, poly_exp, poly_mod, d):
#     print(np.array(neg_poly).shape, np.array(neg_poly_exp).shape, np.array(poly).shape, np.array(poly_exp).shape)
#     res = np.array([0]*(d*2+1))
#     print("LENGTHS", len(neg_poly), len(poly), d*d-1)
#     for i in range(len(neg_poly)):
#         # temp = np.array([0]*(len(poly)*len(neg_poly)))
#         temp = np.array([0]*(d*2+1))
#         print("temp", temp)
#         for j in range(d+1):
#             # print(poly_exp)
#             print("---------",i,j,neg_poly_exp[i], poly_exp[j])
#             exp = neg_poly_exp[i] + poly_exp[j] + d 
#             print("indices",exp)
#             # print(temp[exp])
#             temp[exp] = temp[exp] + neg_poly[i]*poly[j]
#         print("temp all",temp)
#         # temp_neg_poly = np.poly1d(temp[:len(poly)])
#         # temp_poly = np.poly1d(temp)
#         # print("temp poly", temp_poly)
#         res = res + temp
#     element = np.polydiv(res,poly_mod)[1]
#     print("HERE")
#     print(res, np.poly_1d(element))
#     # print("---------")
#     # print(element, len(np.array(element)))
#     if len(np.array(element)) < d:
#         element = np.pad(element, (d-len(np.array(element)),0), 'constant', constant_values=0)  
#     # print(element)
#     # print(np.array(element).shape, res.shape)
#         # res = res + element
#     print("RES")
#     print(np.poly1d(element))
#     return res

# res = neg_poly_multiplication(np.array([1,1]),np.array([-2,-1]), np.array([1,1]),np.array([1,2]), np.poly1d([1] + [0]*(2-1) + [1]),2)
# print(res)
def fig_8_prover_2(g, F1, gamma, lmda, d,q, M=1):
    # h = []
    # for i in range(lambda):
    #     sum = 0
    #     for j in range(M):
    #         sum = poly_add_2D(sum, poly_matmul(gamma[i][j], F[j])) #TODO: Fj(S) implement
    #     val = poly_add_2D(g[i], sum)
    #     h.append(val)
    h = []
    # sum = 0
    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    # print("g , f1 ,gamma")
    # print(g.shape, F1.shape, gamma.shape)
    # print("=====================================")
    # print(g)
    # print(g.shape)
    for i in range(lmda):
        sum = 0
        # for j in range(M):
            # print(gamma[i][j]*np.array(F1))
            # sum = sum + gamma[i][j]*np.array(F1)

        sum = np.sum([gamma[i][j]*np.array(F1) for j in range(M)])
            # print("inside loop multi:")#, np.array(gamma[i]*[F1]).shape)
            # print(gamma[i], [F1])
            # print(g[i].shape)
            # print(sum)
        # print("*******************************")
        # print(g[i].shape, sum.shape)
        # exit()
        # add_tmp = poly_add_3D([g[i]], sum, q,poly_mod, d)
        add_tmp = np.fmod([g[i]]+ sum, q)
        h.append(add_tmp[0])

    # np.sum(gamma[i][j])*np.array(F1)
    # np.sum([gamma[i][j]*np.array(F1) for i in range(lmda) for j in range(M)],axis=2)
    
    # []


    print("h  -------------------- ", h)
    return h

def fig_8_verifier_2(h):
    h = np.array(h)
    print("verifier 2 fig 8", h.shape)
    for i in range(len(h)):
        if h[i][0][-1] != 0:
            return False
    return True

# print("~~~~~~~~~~~~Fig 8 PVPV~~~~~~~~~~")
# m1 = 230
# k = 4
# d = 5
# r = generate_random(16,7,d,1)
# # message = gen_matrix_ring(1,1,1, d)
# message = [[[1,1,1,0,1]]]
# A, B, S = enc_key_gen(d,248)
# miu = np.concatenate((np.reshape(r, (1,28*4,d)), message), axis=1)
# miu = np.array(ring_transpose(miu))
# print("PARAMS", A.shape, B.shape, miu.shape)
# r_pke, c0, c1 = enc_encrypt(A, B, miu, A.shape[1], d, 100000)
# s1 = np.concatenate((r_pke, miu))
# s2 = gen_matrix_ring_with_norm(4,1,21,d,3000)
# g, tg = fig_8_prover_1(-1, s1, 14, 21, d, s2,k = 4,m1 = 230,m2 = 4, l = 1 )
# gamma = fig_8_verifier_1(q,14,k*m1)
# sigma = -1
# neg_poly = [1]*d
# neg_exp = list(range(0,-d,-1))
# message_exp = list(range(d,0,-1))
# poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
# w = 4
# # print(message, message_exp)
# print(message)
# # print(gg)
# F1 = neg_poly_multiplication(neg_poly, neg_exp, message[0][0], message_exp, poly_mod,d)+(w*-1)
# h = fig_8_prover_2(g, F1, gamma, 14, k*m1, d,q)
# res = fig_8_verifier_2(h)
# print("Fig 8 RESULT~~~~~~~~~~")
# print(res)
# print(ggg)

# r = generate_random(4,7,12,20000000)
# message = gen_matrix_ring(1,1,1000, 3)
# A, B, S = enc_key_gen(3,248)
# miu = np.concatenate((np.reshape(r, (1,28*4,3)), message), axis=1)
# miu = np.array(ring_transpose(miu))
# # print("PARAMS", A.shape, B.shape, miu.shape)
# r_pke, c0, c1 = enc_encrypt(A, B, miu, A.shape[1], 3, 100000)
# s1 = np.concatenate((r_pke, miu))
# s2 = gen_matrix_ring_with_norm(4,1,21,3,3000)
# lmda = 14
# g, tg = fig_8_prover_1(-1, s1, lmda, 21, 3, s2,k = 4,m1 = 230,m2 = 4, l = 1 )
# # print("RESULT~~~~~~~~~~")
# # print(g)
# # print(tg)
# d= 3
# M = 4
# gamma = fig_8_verifier_1(q, lmda, M)
# neg_poly = [1]*d
# neg_exp = list(range(0,-d,-1))
# message_exp = list(range(d,0,-1))
# w=  5#l1 norm of message
# poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
# neg_poly_multi = neg_poly_multiplication(neg_poly, neg_exp, message[0][0], message_exp, poly_mod, d)
# # print(neg_poly_multi)
# F1 = np.array(neg_poly_multi) + w*(-1) 
# # print(F1)
# res = fig_8_prover_2(g, F1, gamma, lmda, M, d, 400)
# print("RESULT~~~~~~~~~~")
# print(res)

# l = 1
# tau = 7 
# N = 2048
# d = 512
# n = 1
# pho = l*tau*N/d*(n+l)
# pho_eval = 1
# v_e = l*tau +1
# v_d = 1
# k_bin = 1 
# s1 = 
# m = []
# A1 = 
# A2 = 
# R =
# q1 = 2**24 - 75
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# f1 = poly_matmul(A1, R, q1, poly_mod, N)
# q2 = 2**79 - 67
# f2 = poly_add_2D(poly_matmul(A2, R, q2, poly_mod, N), poly_matmul(m,G, q2, poly_mod, N), q2, poly_mod, N)
# message =  
# w = 
# F_1 = 
# v = [0]*(l*tau)
# m_pke = 117
# n_pke = 2
# q_pke = 1437757
# k_pke = 113
# beta_e = [2*math.sqrt(d*m_pke)] + [math.sqrt(N*k)]*(l*tau-1)
# print(beta_e)
# E1 = np.identity(m_pke)+
# x = 
# m2 = 18
# q = 2**103
# s2 = generate_random(m2, 1, q, 1)

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
    # alpha_d = 2**20.14
    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    # bin_b_d = np.full((1,1,d),np.random.choice([-1,1], 1))
    # bin_b_e = np.full((1,1,d),np.random.choice([-1,1], 1))
    # bin_b_d = np.random.choice([-1,1], 1)
    # bin_b_e = np.random.choice([-1,1], 1)
    bin_b_d = np.zeros((1,1,d))
    bin_b_d[0][0][-1] = np.random.choice([-1,1], 1)
    bin_b_e = np.zeros((1,1,d))
    bin_b_e[0][0][-1] = np.random.choice([-1,1], 1)
    # print(bin_b_d, bin_b_e)
    # exit()
    # y1_sampler = DiscreteGaussianDistributionIntegerSampler(sigma=std_d)
    # y2_sampler = DiscreteGaussianDistributionIntegerSampler(sigma=std_e)
    # y_d = []
    # y_e = []

    # for i in range(y_dim):
    #     y_d.append([[y1_sampler()]*d])
    #     y_e.append([[y2_sampler()]*d])
    # print("y_d -------------")
    # print(y_d)

    y_d = sample_discrete_gaussian_distribution(y_dim, q, d, std_d)
    y_e = sample_discrete_gaussian_distribution(y_dim, q, d, std_e)
    # print(y_d.shape)
    # exit()

    mat_B_d = gen_matrix_ring(y_dim, m2, q, d)
    mat_B_e = gen_matrix_ring(y_dim, m2, q, d)
    # print("SDFJID")
    # print(mat_B_d.dtype,mat_B_e.dtype, y_d.dtype)
    t_d = poly_add_3D(poly_matmul(mat_B_d, s2, q, poly_mod, d), np.array(y_d), q, poly_mod, d)
    # vvvvv = poly_matmul(mat_B_d, s2, q, poly_mod, d)+ np.array(y_d)
    # print(type(vvvvv), np.dtype(vvvvv))
    # print("JERRE")
    # val = np.array(poly_matmul(mat_B_d, s2, q, poly_mod, d)+ np.array(y_d))
    # t_d = custom_fmod_3D(val,val.shape[0],val.shape[1],d, q)
    t_e = poly_add_3D(poly_matmul(mat_B_e, s2, q, poly_mod, d), np.array(y_e), q, poly_mod, d)
    # t_e = custom_mod(poly_matmul(mat_B_e, s2, q, poly_mod, d)+ np.array(y_e),q)
    # val2 = np.array(poly_matmul(mat_B_e, s2, q, poly_mod, d)+ np.array(y_e))
    # t_e =  custom_fmod_3D(val2,val2.shape[0],val2.shape[1],d,q)
    vec_b_d = gen_matrix_ring(m2, 1,q, d)
    vec_b_e = gen_matrix_ring(m2, 1,q, d)
    # val3= poly_matmul(ring_transpose(vec_b_d), s2, q, poly_mod, d) + bin_b_d
    # t_d_prime =custom_fmod_3D(val3, val3.shape[0],val3.shape[1],d,q)
    # t_d_prime = 
    print("HERE++++++++++++++++==")
    # ttt_d_prime = poly_matmul(ring_transpose(vec_b_d), s2, q, poly_mod, d) 
    # start = time.time()
    # print(np.remainder(ttt_d_prime + bin_b_d,q))
    # end = time.time()
    # print(end-start)
    # start = time.time()
    # print(poly_add_3D(ttt_d_prime, bin_b_d,q,poly_mod,d))
    # end = time.time()
    # print(end-start)
    # print(t_d_prime, ttt_d_prime , ttt_d_prime + bin_b_d)
    # exit() 
    # t_e_prime = np.fmod(poly_matmul(ring_transpose(vec_b_e), s2, q, poly_mod, d) + bin_b_e,q)
    # val4= poly_matmul(ring_transpose(vec_b_e), s2, q, poly_mod, d) + bin_b_e
    # t_e_prime = custom_fmod_3D(val4, val4.shape[0], val4.shape[1],d,q)
    t_d_prime = poly_matmul(ring_transpose(vec_b_d), s2, q, poly_mod, d) + bin_b_d 
    t_e_prime = poly_matmul(ring_transpose(vec_b_e), s2, q, poly_mod, d) + bin_b_e
    return t_d, t_d_prime, t_e, t_e_prime, bin_b_d, bin_b_e, y_d, y_e

# d= 256
# m2 = 18
# s2 = gen_matrix_ring_with_norm(m2, 1, 500, d, 1000)
# res = fig_10_prove_1(s2, d=d)
# print(res)

#cd = 512*115
#ce = 512*(1+8*(117+1))
def fig_10_verifier_1(cd, ce, dim=512):
    # R_d = np.random.choice([-1,0,1], p=[0.25,0.5,0.25] , size=(256,cd,d))
    # R_e = np.random.choice([-1,0,1], p=[0.25,0.5,0.25], size=(256,ce,d))
    R_d = np.random.choice([-1,0,1], p=[0.25,0.5,0.25] , size=(dim,cd))
    R_e = np.random.choice([-1,0,1], p=[0.25,0.5,0.25], size=(dim,ce))
    return R_d, R_e

# d= 256
# m2 = 18
# s2 = gen_matrix_ring_with_norm(m2, 1, 500, d, 1000)
# t_d, t_d_prime, t_e, t_e_prime = fig_10_prove_1(s2, d=d)
# r_d, r_e = fig_10_verifier_1(t_d, t_d_prime, t_e, t_e_prime)
# print(r_d, r_e)

# e_d = poly_matmul( ,s) 
# e_e = 
# f1 = poly_matmul(A1, R, q1, poly_mod, N)
# q2 = 2**79 - 67
# f2 = poly_add_2D(poly_matmul(A2, R, q2, poly_mod, N), poly_matmul(m,G, q2, poly_mod, N), q2, poly_mod, N)
# comm = cts_commit(A, message, rand, N, poly_mod, q1, q2, n+l, l*tau)
# f = np.reshape(comm, (t0, 1, l*tau))

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
        print("TMAP: a_prime, a_exp, b_prime, b_exp, poly_mod,d")
        print(a_prime, a_exp, b_prime, b_exp, poly_mod,d)
        print("a,b,k,d", a,b,k,d)
        res = neg_poly_multiplication(a_prime, a_exp, b_prime, b_exp, poly_mod,d)
        sum = sum + res
    return sum

# a = [0,1,2,3,4,5]
# b = [0,10,20,30,40,50]
# k = 3
# d = 2
# res = T_map_computation(a,b, k,d)
# print(res)
# print(asds)
import time

def norm_3D_poly(v, norm, n,m):
    # print(np.array(v).shape, norm, n,m)
    # print("INSIDE NORM")
    # s = time.time()
    # w = []
    # for i in range(n):
    #     for j in range(m):
    #         # print("indices", v[i][j], i, j)
    #         val  = LA.norm(v[i][j],norm)
    #         # print(val)
    #         w.append(val)
    # print(time.time()-s)
    # s = time.time()
    w = [LA.norm(v[i][j],norm) for i in range(n) for j in range(m)]
    # print(time.time()-s)

    # print(w, u)

    return LA.norm(w, norm)

# norm2_3D_poly(z, 256,1,q)
def norm2_3D_poly(v, n,m, q):

    return LA.norm(v)
    # [LA.norm(poly[0])**2 for poly in v]
    # norm_z1 = math.sqrt(np.sum(new_list))
    # print(np.array(v).shape, norm, n,m)
    # print("INSIDE NORM")
    # s = time.time()
    # w = []
    # # print(v)
    # for i in range(n):
    #     for j in range(m):
    #         # x = [custom_mod(num, q) for num in v[i][j]]
    #         x = np.fmod(v[i][j], q)
    #         val = LA.norm(x, 2)
    #         # print("x,val---------", x,val)
    #         w.append(val)
    # print(time.time()-s)
    # s = time.time()
    w = [LA.norm(np.fmod(v[i][j],q),2) for i in range(n) for j in range(m)]
    # print(time.time()-s)
    # print(w, u)

    # print("w==================", w, LA.norm(w,2))
    return LA.norm(w, 2)

# def map_ring_n_to_d(element_N, k):
#     s = []
#     for i in range(k):
#         s.append([element_N[j] for j in range(i, len(element_N), k)])
#     return s

def map_ring_n_to_d(element_N, N, k):
    s = []
    if len(element_N) < N:
        element_N = np.pad(element_N, (N-len(element_N),0), 'constant', constant_values=0)

    for i in range(k):
        s.append([element_N[j] for j in range(i, len(element_N), k)])
    return s

# s = (s1, message)
# def fig_10_prover_2(s, s1,s2,w, message, comm_f, e_d, e_e, e_e_array, x,b_d,R_d,y_e,y_d,b_e,R_e,s_d, s_e,d, m_pke, m1, N, k=4 , l=1, tau=7, ve = 8,k_bin = 1):

def fig_10_prover_2(e_d, e_e, b_d,R_d,y_e,y_d,b_e,R_e,s_d, s_e,d,q, y_dim=1):

    # poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    # tmp = poly_matmul(b_d, R_d, q, poly_mod,d)
    # print("INSIDE")
    # print(b_d, R_d)
    # print(np.array(b_d[0]*R_d).shape, np.array(e_d).shape)
    # print(-1*R_d)
    # exit()
    # bdRded = poly_matmul(poly_matmul(b_d,R_d,q,poly_mod,d), e_d, q, poly_mod, d)
    # beReee =poly_matmul(poly_matmul(b_e,R_e,q,poly_mod,d), e_e, q, poly_mod, d)
    # print(b_d.shape, b_e.shape, b_d, b_e)
    # y_d = np.reshape(y_d, (y_dim*d))
    # y_e = np.reshape(y_e, (y_dim*d))
    print("y vecs")
    print(y_d, y_e)
    # exit()
    # bdRded = poly_matmul(b_d * R_d, e_d, q, poly_mod, d)
    # bdRded = matmul(b_d * R_d, e_d, q, d)
    bdRded = np.matmul(b_d[0][0][-1]*R_d, e_d) 
    
    # bdRded = custom_fmod_2D(bdRded,bdRded.shape[0], bdRded.shape[1], q)
    # [custom_mod(num, mod) for num in intermediate_element]
    # print(bdRded.tolist())
    # print("FIG PROVER 10  bdrded, temp")
    # print(bdRded.shape, temp.shape)
    # print("--------------------------")
    # print(bdRded.tolist())
    
    # print(bdRded, temp)
    # exit()
    # beReee =poly_matmul(b_e*R_e, e_e, q, poly_mod, d)
    beReee = np.matmul(b_e[0][0][-1]*R_e, e_e)
    print(beReee.shape)
    print(bdRded.shape)
    # beReee = np.fmod(beReee,q).astype(float)
    # beReee = custom_fmod_2D(beReee,beReee.shape[0],beReee.shape[1], q)
    # print("--------------------------")
    # print(beReee.tolist())
    # z_d = poly_add_3D(bdRded, y_d, q, poly_mod, d)
    # z_e =  poly_add_3D(beReee, y_e, q, poly_mod, d)
    # new_zd = np.fmod(bdRded + y_d,q) 
    # val = bdRded + y_d %q
    # new_zd = custom_fmod_2D(val,val.shape[0],val.shape[1],q)

    # new_ze = np.fmod(beReee + y_e,q)
    new_ze = beReee + y_e 
    new_zd = bdRded + y_d 
    # new_ze = custom_fmod_2D(val,val.shape[0],val.shape[1],q)
    
    # new_zd = []
    # for i in range(z_d.shape[0]):
    #     row = []
    #     for j in range(z_d.shape[1]):
    #         row.append([custom_mod(num, q2) for num in z_d[i][j]])
    #     new_zd.append(row)

    # new_ze = []
    # for i in range(z_e.shape[0]):
    #     row = []
    #     for j in range(z_e.shape[1]):
    #         row.append([custom_mod(num, q2) for num in z_e[i][j]])
    #     new_ze.append(row)

    # oop = b_d* R_d

    # print(oop[0].tolist(),e_d[:,0].tolist(), np.array(new_zd).shape, z_d[0], new_zd[0])    
    # print(z_d.shape,z_e.shape)
    # exit()

    # z_d = [custom_mod(num, q2) for num in f[0][0]]
    # z_e = [custom_mod(num, q2) for num in f[0][0]]
    # print("PROBLEMO AREA")
    # print(np.array(z_d[0]), np.array(z_d).shape)
    # print(np.array(bdRded[0]), np.array(bdRded).shape)
    # print(np.array(s_d))
    #UNCOMMENT LATER
    # print(z_d.shape, bdRded.shape, z_e.shape, beReee.shape)
    # exit()
    if rejection_sampling_0_int(new_zd, bdRded, s_d,q) !=0:
        print("rejection sampling failed for zd")
        exit()
        return False, None, None #,None, None, None, None, None
    
    if rejection_sampling_0_int(new_ze, beReee, s_e,q) != 0: 
        print("rejection sampling failed for ze")
        exit()
        return False, None, None #None, None, None, None, None
    
    # beta_1 = 2*math.sqrt(d*m_pke)
    # beta_i = math.sqrt(N*k)
    # print("PROBLEMO AREA")
    # print(np.array(e_e).shape,np.array(e_e_array).shape)
    # index_e = e_e_array[0]
    # # print(norm_3D_poly(e_e[:117], 2, 117, 1)**2)
    # x_1 = beta_1**2 - norm_3D_poly(e_e[:index_e], 2, index_e, np.array(e_e).shape[1])**2 #LA.norm(e_e_array[0],2)**2)]
    # print(e_e_array)
    # print("****************************")
    # print(x_1)
    # x_is = [x_1]
    # for i in range(1,len(e_e_array)):
    #     x_i = beta_i**2 - norm_3D_poly(e_e[e_e_array[i-1]:e_e_array[i]], 2, e_e_array[0], np.array(e_e).shape[1])**2 
    #     x_is.append(x_i)
    # # x_1 = [(beta_1 - LA.norm(e_e_array[0],2)**2)]
    # # x_i = [(beta_i - LA.norm(val,2)**2) for val in e_e_array[1:]]
    # # x_is = np.append(np.array(x_1), np.array(x_i))
    # print("_________SHAPES________X________")
    # # print(np.array(x_1).shape) 
    # # print(x_i)
    # # print(np.array(x_is).shape)
    # # print(x_is)
    # # for i in range(len(x_is)):
    # #     x= np.array([int(bit) for bit in np.binary_repr(x_i[i])])
    # x = np.array([])
    # max_abs_val = max(x_is, key=abs)
    # bin_size = int(max_abs_val).bit_length()
    # print(max_abs_val, bin_size)
    # for i in range(len(x_is)):
    #     x_i_binval = np.array([int(bit) for bit in np.binary_repr(abs(int(x_is[i])))]) #TODO abs -- remove
    #     # print(x_i_binval)
    #     if len(x_i_binval) < bin_size:
    #         x_i_binval = np.pad(x_i_binval, (bin_size-len(x_i_binval),0), 'constant', constant_values=0)
    #     x = np.concatenate((x, x_i_binval))
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
    # print(x)
    # print(np.array(x).shape)

    # exit()

    ################################################################################# part used to define S* but is not actually used in other figures/protocols eep
    # s_star = (s2, (s1,x), (m, y_d, y_e, b_d, b_e)) #HERE
    # sigma = -1
    # # print("COMM_F", comm_f, comm_f.shape)
    # # print(b_d, b_e)
    # g_d = [(np.array(b_d)-1)*(np.array(b_d)+1)]*d
    # g_e = [(np.array(b_e)-1)*(np.array(b_e)+1)]*d
    # phi = [comm_f, g_d, g_e]
    # print("PHI", phi)
    # neg_poly = [1]*d
    # neg_exp = list(range(0,-d,-1))
    # message_exp = list(range(d-1,-1,-1))
    # print(np.array(message).shape, neg_exp)
    # F1 = neg_poly_multiplication(neg_poly, neg_exp, message[0][0], message_exp, poly_mod, d)-1*w
    # id = np.identity(m_pke)
    # # E = np.zeros((m_pke, 2*(m1+l)))
    # # E[:m_pke, :m_pke] = id
    # # # value = poly_matmul(E, s, q, poly_mod, N=d)
    # # # norm = LA.norm(value, 2)**2
    # # # x_i = np.array([int(bit) for bit in np.binary_repr(x_i)])
    # # # for i in range(2,l*tau):
    # # #     id = np.identity(k*N/d)
    # # #     E = np.zeros((m_pke, 2*(m1+l)))
    # # #     E[i:, i:] = id
    # # #     poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
    # # #     value = poly_matmul(E, s, q, poly_mod, N=d)
    # # #     norm = LA.norm(value, 2)**2
    # # #     binval= np.array([int(bit) for bit in np.binary_repr(beta_1-norm)])
    # # #     np.concatenate((x, binval), axis = 0)
    # # Ebin = np.zeros((1,(m1),d))#COMMENT OUT INCORRECT
    # # # Ebin = np.zeros((1,23,2)) #COMMENT OUT INCORRECT
    # # print("CHECK============================================")
    # # print(Ebin)

    # # Ebin[:1,-1:] = [1]*d
    # # print(Ebin)
    # # # print("HERE")
    # # # print(Ebin, np.array(s).shape)
    # # # print(poly_matmul(Ebin, s, q, poly_mod, d))
    # # # print(x, np.array(x).shape)
    # # print("poly_matmul(Ebin, s, q, poly_mod, d)[0][0]", poly_matmul(Ebin, s, q, poly_mod, d), poly_matmul(Ebin, s, q, poly_mod, d).shape)
    # # x_prime =np.concatenate((x,poly_matmul(Ebin, s, q, poly_mod, d)[0][0]))

    # # print(np.array(x_prime).shape)
    # # G = T_map_computation(x_prime, x_prime - np.identity((ve+k_bin)*d),(ve+k_bin)*d, (ve+k_bin)*d) #TODO
    # # G = T_map_computation(x_prime, x_prime - np.identity(np.array(x_prime).shape[0]),np.array(x_prime).shape[0], np.array(x_prime).shape[0]) #TODO
    # G = [] 
    # H_e = []
    # H_d = [] 
    # # for i in range(256): #TODO
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # # print(z_e, np.array(z_e).shape)
    # # print(np.array(b_e).shape, np.array(R_e[0]).shape)
    # # print(y_e[0], np.array(y_e[0]).shape)
    # # bbb=T_map_computation(poly_matmul(b_e,[R_e[0]],q, poly_mod,d), e_e)
    # # print(bbb, np.array(bbb).shape)
    # # print(y_d)
    # b_e = [[b_e[0][0]*len(e_e)]]
    # b_d = [[b_d[0][0]*len(e_d)]]
    # print(np.array(b_e).shape, b_e[0][0], len(e_e))
    # print(b_e)

    # for i in range(len(z_e)): #TODO
    #     H_e.append(np.array([z_e[i]])+T_map_computation(poly_matmul(b_e,[R_e[i]],q, poly_mod,d), e_e, 1,1, poly_mod)*-1+ y_e[i])
    #     H_d.append(np.array([z_d[i]])+T_map_computation(poly_matmul(b_d,[R_d[i]],q, poly_mod,d), e_d, 1,1, poly_mod)*-1+ y_d[i])
    #     # H_d.append(poly_add_3D(poly_add_3D([z_d[i]], T_map_computation(poly_matmul(b_d,[R_d[i]],q, poly_mod,d), e_d,1,1)*-1,q, poly_mod,d),y_d[i],q, poly_mod,d)) 
    # print(z_e)
    # print(H_e)
    # I = []
    # beta_e = [2*math.sqrt(d*m_pke),math.sqrt(N*k),math.sqrt(N*k),math.sqrt(N*k),math.sqrt(N*k),math.sqrt(N*k),math.sqrt(N*k)]
    # p_1 = [2**j for j in range(2*int(math.log(beta_1,2)) - 1)] + [0]*(len(x) - len([2**j for j in range(2*int(math.log(beta_1,2))-1)]))
    
    # # print(p_1, x)
    # # exit()
    # I.append(T_map_computation(e_e_array[0],e_e_array[0],23,1, poly_mod) + T_map_computation(p_1,x[0],23,1, poly_mod) + beta_e[0]**2)
    # print("SHAPES--------------------------------------")
    # # print(np.array(e_e_array).shape, np.array(x).shape, np.array(beta_e).shape)
    # # print(ve)
    # for i in range(1,ve):
    #     # print(math.log(beta_i,2))
    #     p_i = [2**j for j in range(2*int(math.log(beta_i,2))-1)] + [0]*(len(x) - len([2**j for j in range(2*int(math.log(beta_i,2)-1))]))
    #     # print(p_i)
    #     # print("IN FOR LOOP", np.array(e_e_array[i]).shape, np.array(p_i).shape, np.array(x[i]).shape)
    #     # print(T_map_computation(e_e_array[i],e_e_array[i],23,1) + T_map_computation(p_i,x[i],23,1) + beta_e[i]**2)
    #     print("INDEX", i, ve)
    #     I.append(T_map_computation(e_e_array[i],e_e_array[i],23,1, poly_mod) + T_map_computation(p_i,x[i],23,1,poly_mod) + beta_e[i]**2)
    # print(d)
    # delta_d = [0]*d
    # print(delta_d)
    # delta_e = [0]*d
    # J_e = []
    # J_d = []
    # for i in range(d):
    #     delta_e[i] = 1
    #     delta_d[i] = 1
    #     J_e.append(T_map_computation(delta_d, b_d, d,1, poly_mod))
    #     J_d.append(T_map_computation(delta_e, b_e, d,1,poly_mod))
    #     delta_d = [0]*d
    #     delta_e = [0]*d
    # psi = [F1, G, H_d, H_e, I, J_d, J_e]
    return True, np.array(new_zd), np.array(new_ze) #, s_star, sigma, phi, psi
#HERE IS THE FIG 10 CODE
# n_pke= 2
# m_pke= 117
# k_pke= 113
# n_pke= 2
# m_pke= 1
# k_pke= 3
# d= 2
# q = 15
# q_pke = 10
# m1 = 581
# N= 8
# print("STEP 1--------------------------------------------------")
# A_pke, B_pke, S = enc_key_gen(n=n_pke,m=m_pke,k=k_pke,d=d,q = q)
# print("post key gen")
# # print(A_pke, B_pke)
# D = np.zeros((n_pke+k_pke,m_pke+k_pke,d))
# D[:n_pke, :m_pke] = A_pke
# D[n_pke: , :m_pke]= B_pke
# identity_ring_n = np.dstack([np.identity(k_pke)]*d)
# # print(identity_ring_n)
# D[n_pke:, m_pke:] = identity_ring_n
# D1 = (q_pke**-1) * D
# print("STEP 2--------------------------------------------------")
# rand = generate_random(4,7,d,10000000) #beta is 1
# print("post rand")
# message = gen_matrix_ring(1,1,q,d)
# print("post message")
# A,D, q1, q2, N, k_prime, gamma, gamma_prime, alpha = cts_setup(n=1, k=4 ,l=1, l_prime=2,q1=75,q2=67,N=d)
# print("A.shape", A.shape)
# # poly_mod_N = np.poly1d([1] + [0]*(N-1) + [1])
# poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
# comm = cts_commit(A, message, rand, d, poly_mod, q1, q2, n, l)
# print("post comm")
# f0 = [comm[:n, :l*tau]]*28
# f1 = [comm[n:, :l*tau]]*28
# # u1 = (q_pke**-1) * comm #TODO
# u1 = np.array([[[1,2]],[[3,4]],[[5,6]],[[7,8]],[[9,10]]])
# print("post u1")
# r = generate_random(2,1, d,20000000)
# message = gen_matrix_ring(1,1,q,d)
# print("STEP 3--------------------------------------------------")
# miu = np.concatenate((r, message))
# r_pke, c0, c1 = enc_encrypt(A_pke, B_pke, miu, m=1, d=2)
# print("MIU,RPKE")
# print(miu.shape, r_pke.shape)
# s1 = np.concatenate((r_pke, miu))
# # s =  [s1,apply_automorphism_neg_1(-1, s1,m1 ,1)]
# s= s1
# poly_mod = np.poly1d([1] + [0]*(d-1) + [1])
# D1s = poly_matmul(D1, s, q, poly_mod, d)
# e_d = poly_add_3D(D1s, -1*u1, q, poly_mod, d)
# k =4 
# # N = 2048
# # N = 16
# # d= 512
# # E = []
# # E = np.zeros((m_pke+7,m_pke+7,d))
# E = np.zeros((23,23,d))
# print(np.array(s).shape, np.array(s1).shape)
# print(s1)
# # tmp = np.array(apply_automorphism_neg_1(-1, np.array(s1),8 ,1))
# tmp = np.array(s1)
# print(tmp)
# s =  np.concatenate((np.array(s1), tmp, tmp,tmp,tmp, [[[1,2]],[[3,4]],[[4,5]]]))
# E[ :m_pke, :m_pke] =  np.dstack([np.identity(m_pke)]*d)
# e_e = poly_matmul(E, s, q, poly_mod, d) 
# e_e_array = e_e
# for i in range(1,7):
#     E_i = np.zeros((23,23,d))
#     identity_ring = np.dstack([np.identity(16)]*d)
#     E_i[i:16+i,i:16+i] = identity_ring
#     e_e_i = poly_matmul(E_i, s, q, poly_mod, d) 
#     e_e = np.concatenate((e_e,e_e_i))
#     np.append(e_e_array, e_e_i)
# print("E_E_ARRAY-----------------------", np.array(e_e_array).shape)
# l=1
# tau = 7
# g = []
# for i in range(tau):
#     val = 17**(i/tau)
#     g.append(val)
# id = np.identity(1)
# G = [[i]*2 for i in g]
# m2 = 18
# s2 = gen_matrix_ring_with_norm(m2, 1, 500, d, 1000)
# t_d, t_d_prime, t_e, t_e_prime, b_d, b_e, y_d, y_e = fig_10_prove_1(s2, d=d, N=N,y_dim=1)
# R_d, R_e = fig_10_verifier_1(t_d, t_d_prime, t_e, t_e_prime, d=d, N=N)
# # print("R_D SHAPE & b_d")
# # print(R_d.shape, b_d.shape)
# n = 2
# # m2 = 18
# A2 = gen_matrix_ring(n, m2, q, d)
# # R =  
# # rand = 
# # f2 = poly_add_2D(poly_matmul(A2, R, q2, poly_mod, N), poly_matmul(m,G, q2, poly_mod, N), q2, poly_mod, N)
# # S=1

# comm_f = np.concatenate((f0,f1))[0]
# alpha_d = (d*m_pke+1)*math.sqrt((n_pke+1)*d)
# k_bin = 8
# v_e = 1
# alpha_e = math.sqrt(4*d*m_pke + 6*N*k + d*(k_bin+v_e))
# s_d = 12*math.sqrt(337)*alpha_d
# s_e = 2.5*math.sqrt(337)*alpha_e
# w = 1000

# print("______________________________FIG 10 Prover two____________________________________")
# # print(np.array(e_e).shape)
# # res = fig_10_prover_2(s, s1,s2,w, message, comm_f, e_d, e_e, b_d[0], R_d, y_e, y_d, b_e[0], R_e, s_d, s_e, d, m_pke, m1, N,k=4 , l=1, tau=7, ve = 15,k_bin = 1)
# R_d = np.random.choice([-1,0,1],(1,1,d))
# R_e = np.random.choice([-1,0,1], (1,1,d))
# e_e = gen_matrix_ring(1,1,q,d)
# e_d = gen_matrix_ring(1,1,q,d)
# print("---------------------------", b_e.shape, R_e.shape, e_e.shape)
# print(R_e)
# # print(poly_matmul([[b_e]],[R_e], q,poly_mod, d))
# # print("---------------------------", b_e, R_e.shape, e_d.shape)

# z_d, z_e, s_star, sigma, phi, psi = fig_10_prover_2(s, s1,s2,w, message, comm_f, e_d, e_e, e_e_array, [[b_d]], R_d, y_e, y_d, [[b_e]], R_e, s_d, s_e, d, m_pke, m1, N, k=4 , l=1, tau=7, ve = 7,k_bin = 1)
# print("FIG 10 Prover 2 results", z_d, z_e, s_star, sigma, phi, psi)
# k=4
# alpha_d = (d*m_pke+1)*math.sqrt((n_pke+1)*d)
# alpha_e = math.sqrt(4*d*m_pke + 6*N*k + d*(k_bin+v_e))
# std_d = 12*math.sqrt(337)*alpha_d
# std_e = 2.5*math.sqrt(337)*alpha_e
# bounds_inf = 14*std_d
# #TODO: unsure what this t is
# bounds_euc = t*math.sqrt(256)*std_e

# def check_norm_inf(a, bounds):
#     if LA.norm(a, np.inf) <= bounds:
#         return True
#     return False

# def check_norm_euclidean(a, bounds):
#     if LA.norm(a, 2) <= bounds:
#         return True
#     return False

# def izk_prove(x,s1,m):
#     b_d = np.random.choice([-1,1], 1)
#     b_e = np.random.choice([-1,1], 1) 
#     d = 512
#     m2 = 18
#     q = 
#     q2 =  
#     s2 = 
#     q = 
#     alpha_d = 
#     alpha_e = 
#     #TODO: double check if gamma values for std_d and std_e are right
#     std_d = 16*math.sqrt(337)*alpha_d
#     std_e = 1*math.sqrt(337)*alpha_d
#     y1_sampler = DiscreteGaussianDistributionIntegerSampler(sigma=sigma1)
#     y2_sampler = DiscreteGaussianDistributionIntegerSampler(sigma=sigma2)
#     y_d = []
#     for i in range(256/d):
#         y_d[i] = y1_sampler() 
#     y_e = []
#     for i in range(256/d):
#         y_e[i] = y2_sampler()
#     t_d = poly_add_2D(poly_matmul_no_mod(gen_matrix_ring(256/d, m2, q2, d), s2, poly_mod, d), y_d)
#     t_e = poly_add_2D(poly_matmul_no_mod(gen_matrix_ring(256/d, m2, q, d), s2, poly_mod, d), y_e)
#     t_d_prime = poly_add_2D(poly_matmul_no_mod(gen_matrix_ring(m2, 1, d), s2, q, poly_mod, d), b_d)
#     t_e_prime = poly_add_2D(poly_matmul_no_mod(gen_matrix_ring(m2, 1, d), s2, q, poly_mod, d), b_e)
#     #sends above ts to verifier
#     # ve = l*tau + 1
#     vd = 1
#     ve = 15
#     cd = d*115
#     ce = d*( 1 +15*(k*N/d+1))
#     R_d = np.random.choice([-1,0,1],(256,cd))
#     R_e = np.random.choice([-1,0,1], (256,ce))
#     s = [s1, m]
#     # e_d = poly_matmul_no_mod(D1,s,poly_mod,d)

#     z_d = poly_add_2D(poly_matmul(poly_matmul(b_d, R_d), e_d), y_d)
#     z_e =  poly_add_2D(poly_matmul(poly_matmul(b_e, R_e), e_e), y_e)
#     if rejection_sampling_0(z_d, poly_matmul(poly_matmul(b_d, R_d), e_d), s_d) != 0 or rejection_sampling_0(z_e, poly_matmul(poly_matmul(b_e, R_e), e_e), s_e) 1= 0:
#         return "ABORT"
#     s_star = (s2, (s1,x), (m, y_d, y_e, b_d, b_e))
#     sigma =  
#     phi = 
#     psi =
#     return
import time
def accept_fig_10(z_d, z_e, s_d, t_bound, s_e):
    # print(z_d) #, z_e)
    # new_d = []
    # new_e = []
    # s = time.time()
    # for i in range(z_d.shape[0]):
    #     new_d.append(np.max(z_d[i]))
    # for i in range(z_e.shape[0]):
    #     new_e.append(np.max(z_e[i]))
    # print(time.time()-s)

    # s = time.time()
    new_d = [np.max(z_d[i]) for i in range(z_d.shape[0])]
    new_e = [np.max(z_e[i]) for i in range(z_e.shape[0])]

    # print(new_d)
    # print(LA.norm(new_d, np.inf))
    # exit
    # print(14*s_d)
    if LA.norm(new_d, np.inf) <= 14*s_d and LA.norm(new_e) <=t_bound*math.sqrt(256)*s_e:
        return True
    else:
        return False

# t=1000000
# print("ACCEPT FIG 10___________________________________")
# print(np.array(z_d).shape, np.array(z_e).shape)
# print(accept_fig_10(z_d[0], z_e[0], s_d, t, s_e))

def nizk_prove(f_comm, u, sig, gamma_prime, mod, poly_mod, N):
    print(poly_matmul(f_comm, sig, mod, poly_mod, N))
    print(u)
    if np.all(poly_matmul(f_comm, sig, mod, poly_mod, N) == np.array(u)):
        print("equals")
        if LA.norm(sig,2) <= gamma_prime:
            print("within bound")
            return True
    return False

def nizk_verify(crs, x, pi):
    return

def nizk_simsetup():
    return

#table 10
def disclosure_attributes_example(xi, N, d, t, k, k_hat, poly_mod, q1, q2, n):
    bound = xi*math.sqrt(2*k*N) 
    B = gen_matrix_ring(k_hat, k, q2,N)
    r1 = generate_random(k,1,N,1) 
    # to disclose 3 (ie first two indices):
    disclose_indices = [0,1]
    # message: 14, 5 in binary (over 4 bits each)
    # message = np.array([[np.concatenate(([1,1,1,0,0,1,0,1],[0]*(N-8)))]])
    # m_prime = np.array([[np.concatenate(([0,0,0,0,0,1,0,1],[0]*(N-8)))]])
    # m_att = np.array([[np.concatenate(([1,1,1,0,0,0,0,0],[0]*(N-8)))]])
    message = np.array([[np.concatenate(([1,1,1,0],[0]*(N-4)))]])
    m_prime = np.array([[np.concatenate(([0,0,1,0],[0]*(N-4)))]])
    m_att = np.array([[np.concatenate(([1,1,0,0],[0]*(N-4)))]])
    A,D, q1, q2, N, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=1, l_prime=2,q1=q1,q2=q2,N=N)
    A1 = A[:n, :]
    A2 = A[n:,:]
    poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
    comm = cts_commit(np.array(A), np.array(message), np.array(r1), N, poly_mod, q1, q2, n, l)
    t11 = comm[:n]
    t21 = comm[n:]
    g= []

    for i in range(k_hat):
        f = gen_matrix_ring(1,1,q2,N)
        print("f in for loop", f, np.array(f).shape)
        for j in disclose_indices:
            f[0][0][j] = 0
        g.append(f[0])

    tmp = poly_matmul(B, r1,q2,poly_mod, N)
    # t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)
    t_g = np.fmod(tmp+np.array(g), q2)
    
    t_g = mod_3D(len(tmp+np.array(g)),tmp+np.array(g),q2)

    gammas = [random.randrange(0, q2) for i in range(k_hat)]

    y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**N, sigma=xi)
    y = []
    for i in range(k):
        y.append([y_sampler()])
    h = []
    w = []
    for i in range(k_hat):
        # h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
        h_i = np.fmod([g[i]]+ np.multiply(gammas[i],m_prime), q2)

        # w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A[1:]),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
        w_i = poly_matmul(np.fmod(np.multiply(gammas[i],A[1:])+B[[i],:], q2), y, q2, poly_mod, N)

        h.append(h_i[0])
        print(np.array(w_i).shape)
        w.append(w_i[0])

    k_challenge = 2
    challenge = generate_random(1,1,N,1) 
    w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))
    # z = np.array(poly_add_3D(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod, N))
    z = np.fmod(np.array(y)+ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1)
    if math.sqrt(np.sum(np.inner(z,z))) > bound:
        # print("first check failed 2385 and index", index)
        print(np.sum(np.inner(z,z)), bound)
        return False

    res_1 = poly_matmul(A1,z, q1,poly_mod,N) % q1
    # inter = poly_add_3D(poly_matmul(A1,y,q1, poly_mod,N),poly_matmul(A1,ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod,N),q1,poly_mod, N)
    inter = np.fmod(poly_matmul(A1,y,q1, poly_mod,N) + poly_matmul(A1,ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod,N),q1)
    w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))
    # res_2 = poly_add_3D(np.array(w_bold), poly_matmul(np.array(challenge),np.array(t11),q1, poly_mod, N),q1, poly_mod, N) %q1
    res_2 = np.fmod(np.array(w_bold)+ poly_matmul(np.array(challenge),np.array(t11),q1, poly_mod, N),q1) %q1
    if res_1.all() != res_2.all():
        if poly_matmul(A1,z, q1,poly_mod,N).all() == poly_add_3D(poly_matmul(A1,y, q1,poly_mod,N),poly_matmul(A1,np.array(challenge)*r1, q1,poly_mod,N),q1, poly_mod, N).all():
            print('235')
            print(poly_matmul(A1,z, q1,poly_mod,N))
            print(poly_add_3D(poly_matmul(A1,y, q1,poly_mod,N),poly_matmul(A1,np.array(challenge)*r1, q1,poly_mod,N),q1, poly_mod, N))
            print(True)
        print("second of the or statement 2388")
        return False

    for i in range(k_hat):
        if h[0][0][0] != 0:
            print("failed 2392")
            exit()
        print("B SHAPE",np.array(B).shape)
        print("B[1] SHAPE",np.array(B[[1],:]).shape)
        
        # lhs_1 = poly_add_3D(np.array(gammas[i])*np.array(A2), B[[i],:], q2, poly_mod, N)
        lhs_1 = np.fmod(np.array(gammas[i])*np.array(A2)+ B[[i],:], q2)
        lhs = poly_matmul(lhs_1, z, q2, poly_mod, N) %q2

        # temp0 = poly_add_3D(np.array(gammas[i])*np.array(poly_add_3D(t21, -1*m_att, q2, poly_mod, N)), [t_g[i]], q2, poly_mod, N)
        temp0 = np.fmod(np.array(gammas[i])*np.array(np.fmod(t21 + -1*m_att, q2)) + [t_g[i]], q2)
        temp1 = poly_matmul(challenge,poly_add_3D(temp0, [-1*h[i]], q2, poly_mod, N), q2, poly_mod, N)
        # actual_rhs = poly_add_3D([w[i]], temp1, q2, poly_mod, N) %q2
        actual_rhs = np.fmod([w[i]]+ temp1, q2) %q2
        # print("------------------------FINAL RESULT LHS")
        # print(lhs)
        # print("------------------------FINAL RESULT RHS")
        # print(actual_rhs)
        # print("------------------------FINAL RESULT SHAPES")
        # print(np.array(rhs).shape, np.array(lhs).shape)   
        if np.array(lhs).all() != np.array(actual_rhs).all():
            return False
        return True

# def disclosure_not_attribute():
#     N = 4
#     q1 = 100
#     q2 = 100
#     poly_mod = np.array([1,0,0,0,1])
#     # q = np.array([1, -65, -57, -120, -121, -55, -63])


#     # m0 and m1 is NOT 01
#     for index in range(100):
#         xi = 50 #TODO
#         N= 4
#         d = 512
#         t = 4
#         k = 4
#         k_hat = 2
#         poly_mod = np.poly1d([1] + [0]*(N-1) + [1])

#         # q1 = 2**24 - 75
#         q1 = 100
#         q2 = 100
#         # q2 = 2**79 - 67
#         bound = xi*math.sqrt(2*k*N) 
#         B = gen_matrix_ring(k_hat, k, q2,N)
#         n = 1
#         r1 = generate_random(k,1,N,1) 
#         # m0 is not 16 (ie first two indices):
#         val = 16
#         att = 1
#         disclose_indices = [0]
#         # message: 14, 5 in binary (over 4 bits each)
#         # message = np.array([[np.concatenate(([1,1,1,0,0,1,0,1],[0]*(N-8)))]])
#         # m_prime = np.array([[np.concatenate(([0,0,0,0,0,1,0,1],[0]*(N-8)))]])
#         # m_att = np.array([[np.concatenate(([1,1,1,0,0,0,0,0],[0]*(N-8)))]])
#         message = np.array([[np.concatenate(([1,1,1,0],[0]*(N-4)))]])
#         m_prime = np.array([[np.concatenate(([1-val,1,1,0],[0]*(N-4)))]])
#         m_att = np.array([[np.concatenate(([val,0,0,0],[0]*(N-4)))]])
#         A,D, q1, q2, N, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=1, l_prime=2,q1=q1,q2=q2,N=N)
#         A1 = A[:n, :]
#         A2 = A[n:,:]
#         poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
#         comm = cts_commit(np.array(A), np.array(message), np.array(r1), N, poly_mod, q1, q2, n, l)
#         t11 = comm[:n]
#         t21 = comm[n:]
#         g= []

#         for i in range(k_hat):
#             f = gen_matrix_ring(1,1,q2,N)
#             print("f in for loop", f, np.array(f).shape)
#             for j in disclose_indices:
#                 f[0][0][j] = f[0][0][j]*(val-att)
#             # f = [custom_mod(num, q2) for num in f[0][0]]
#             # f = np.fmod(f[0][0], q2)
#             f = np.fmod(f[0][0] ,q2)
#             g.append([f])
#         # exit()
#         # g = [custom_mod(num, q2) for num in g]
#         tmp = poly_matmul(B, r1,q2,poly_mod, N)
#         # t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)
#         t_g = np.fmod(tmp+np.array(g), q2)

#         gammas = [random.randrange(0, q2) for i in range(k_hat)]

#         y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**N, sigma=xi)
#         y = []
#         for i in range(k):
#             y.append([y_sampler()])
#         h = []
#         w = []
#         for i in range(k_hat):
#             # h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
#             h_i = np.fmod([g[i]]+ np.multiply(gammas[i],m_prime), q2)
#             # w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A[1:]),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
#             w_i = poly_matmul(np.fmod(np.multiply(gammas[i],A[1:])+B[[i],:], q2), y, q2, poly_mod, N)

#             h.append(h_i[0])
#             print(np.array(w_i).shape)
#             w.append(w_i[0])

#         k_challenge = 2
#         challenge = generate_random(1,1,N,1) 
#         w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))
#         # z = np.array(poly_add_3D(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod, N))
#         z = np.array(np.fmod(np.array(y)+ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1))

#         if math.sqrt(np.sum(np.inner(z,z))) > bound:
#             print("first check failed 2385 and index", index)
#             print(np.sum(np.inner(z,z)), bound)
#             exit()

#         res_1 = poly_matmul(A1,z, q1,poly_mod,N) % q1
#         # inter = poly_add_3D(poly_matmul(A1,y,q1, poly_mod,N),poly_matmul(A1,ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod,N),q1,poly_mod, N)
#         inter= np.fmod(poly_matmul(A1,y,q1, poly_mod,N)+poly_matmul(A1,ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod,N),q1)
#         w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))
#         # res_2 = poly_add_3D(np.array(w_bold), poly_matmul(np.array(challenge),np.array(t11),q1, poly_mod, N),q1, poly_mod, N) %q1
#         res_2 = np.fmod(np.array(w_bold)+ poly_matmul(np.array(challenge),np.array(t11),q1, poly_mod, N),q1)%q1
#         if res_1.all() != res_2.all():
#             if poly_matmul(A1,z, q1,poly_mod,N).all() == poly_add_3D(poly_matmul(A1,y, q1,poly_mod,N),poly_matmul(A1,np.array(challenge)*r1, q1,poly_mod,N),q1, poly_mod, N).all():
#                 print('235')
#                 print(poly_matmul(A1,z, q1,poly_mod,N))
#                 print(poly_add_3D(poly_matmul(A1,y, q1,poly_mod,N),poly_matmul(A1,np.array(challenge)*r1, q1,poly_mod,N),q1, poly_mod, N))
#                 print(True)
#             print("second of the or statement 2388")
#             exit()

#         for i in range(k_hat):
#             if h[0][0][0] == 0:
#                 print("failed h is non zero check", h, index, g,gammas)
#                 exit()
#             print("B SHAPE",np.array(B).shape)
#             print("B[1] SHAPE",np.array(B[[1],:]).shape)
            
#             # lhs_1 = poly_add_3D(np.array(gammas[i])*np.array(A2), B[[i],:], q2, poly_mod, N)
#             lhs_1 = np.fmod(np.array(gammas[i])*np.array(A2)+ B[[i],:], q2)
#             lhs = poly_matmul(lhs_1, z, q2, poly_mod, N) %q2

#             # temp0 = poly_add_3D(np.array(gammas[i])*np.array(poly_add_3D(t21, -1*m_att, q2, poly_mod, N)), [t_g[i]], q2, poly_mod, N)
#             temp0 = np.fmod(np.array(gammas[i])*np.array(t21 + -1*m_att)+ [t_g[i]], q2)
#             # temp1 = poly_matmul(challenge,poly_add_3D(temp0, [-1*h[i]], q2, poly_mod, N), q2, poly_mod, N)
#             temp1 = poly_matmul(challenge, np.fmod(temp0 + [-1*h[i]], q2), q2, poly_mod, N)
#             # actual_rhs = poly_add_3D([w[i]], temp1, q2, poly_mod, N) %q2
#             actual_rhs = np.fmod([w[i]]+ temp1, q2)%q2
#             print("------------------------FINAL RESULT LHS")
#             print(lhs)
#             print("------------------------FINAL RESULT RHS")
#             print(actual_rhs)
#             print("------------------------FINAL RESULT SHAPES")
#             # print(np.array(rhs).shape, np.array(lhs).shape)   
#             if np.array(lhs).all() != np.array(actual_rhs).all():
#                 print("failed 2416")
#                 exit()
#             print("TRUEEEEEEE")

# xi = 50 #TODO
# N= 4
# d = 512
# t = 4
# k = 4
# k_hat = 2
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# # q1 = 2**24 - 75
# # q1 = 100
# # q2 = 100
# # q2 = 2**79 - 67
# n=1
# disclosure_attributes_example(xi, N, d, t, k, k_hat, poly_mod, q1, q2,n)

#disclosure of properties attributes m0 NOT 1
# xi = 7.6*1000 #TODO
# N= 6
# d =6
# k = 4
# k_hat = 2
# q2 = 2**79-67
# bound = xi*math.sqrt(2*k*N)
# B = gen_matrix_ring(k_hat, k, q2,N)

# r1 = generate_random(4,1,N,1)
# # message = gen_matrix_ring(1,1,q,N)
# attr_index = 0
# message = [[[2,0,0,0,1]]]
# m_prime = [[[2-1,0,0,0,1]]]
# m_att = [[[1,0,0,0,1]]]
# A,D, q1, q2, N, k_prime, gamma, gamma_prime, alpha = cts_setup(n=1, k=4 ,l=1, l_prime=2,q1=75,q2=67,N=d)
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# comm = cts_commit(np.array(A), np.array(message), np.array(r1), d, poly_mod, q1, q2, n, l)
# t11 = comm[:1]
# t21 = comm[1:]

# print("**********")
# print(A.shape, A[1:])
# #CHEW
# # g = []
# #disclose first (0th) attribute
# g = gen_matrix_ring(1,1,q2,N)
# g[0][0][0] = 0
# for i in range(1, k_hat):
#     f = gen_matrix_ring(1,1,q2,N)
#     f[0][0][attr_index] = f[0][0][0]*16-1 
#     g = np.concatenate((g, f))
# print(g)
# N = 5
# q2 = 999
# # poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# print(B.shape, r1.shape, np.array(g).shape)
# tmp = poly_matmul(B, r1,q2,poly_mod, N)
# print("HERE")
# t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)

# gammas = [random.randrange(0, q2) for i in range(k_hat)]

# y_sampler = DiscreteGaussianDistributionIntegerSampler(sigma=xi)
# y = []
# for i in range(k):
#     y.append([[y_sampler()]*N])
# print("HERRRRRRRRRRRRRRRRRRRR")
# # y = [y]
# # print(y)

# w_bold = poly_matmul(A[:1], y, q1, poly_mod, N)
# h = []
# w = []
# for i in range(k_hat):
#     print(g[i], np.multiply(gammas[i],m_prime))
#     h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
#     print("FOR LOOP CHCKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
#     print(np.multiply(gammas[i],ring_transpose(A[1:])).shape) 
#     print(np.array(B[[i],:]).shape)
#     print(np.array(y).shape)
#     w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A[1:]),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
#     h.append(h_i)
#     w.append(w_i)
# # print("&&&&&&&&&&&77")
# # print(np.array(w).shape)
# # print(w)
# # print(sdfd)

# k_challenge = 2
# # challenge = [random.randint(-5*k_challenge,5*k_challenge) for i in range(N)]
# # norm_inf = LA.norm(challenge, np.inf) #TODO
# # norm_one = LA.norm(challenge, 1)
# # while norm_inf != 1 and norm_one != k_challenge:
# #     coefficients = [random.randint(-5*k_challenge,k_challenge) for i in range(N)] 
# #     norm_inf = LA.norm(challenge, np.inf)
# #     norm_one = LA.norm(challenge, 1)
# challenge = [[[-1,0,1,0,1]]]
# # matrix = np.reshape(coefficients,(1,1,N))

# z = poly_add_3D_no_mod(ring_transpose(y), poly_matmul_no_mod(challenge,ring_transpose(r1),poly_mod, N), poly_mod, N)
# rej_outcome = rejection_sampling_1(z,poly_matmul_no_mod(challenge,ring_transpose(r1),poly_mod, N), xi, 6)
# print(rej_outcome)

# print(np.array(z).shape, w_bold)
# if np.sum(np.inner(ring_transpose(z),ring_transpose(z))) > bound or z != poly_add_3D(w_bold, poly_matmul(challenge,t11,q1, poly_mod, N), poly_mod, N):
#     print("failed 1", bound, np.sum(np.inner(ring_transpose(z),ring_transpose(z))))
# # exit()
# for i in range(k_hat):
#     if h_i[0][0][0] != 0:
#         print("failed")
#     print("B SHAPE",np.array(B).shape)
#     print("B[1] SHAPE",np.array(B[[1],:]).shape)

#     lhs_1 = poly_add_3D_no_mod(np.multiply(gammas[i],ring_transpose(A[1:])), ring_transpose(B[[i],:]), poly_mod, N)
#     lhs = poly_matmul(lhs_1, z, q2, poly_mod, N)
#     print(lhs.shape)
#     temp0 = poly_add_3D(np.multiply(gammas[i], poly_add_3D(t21, np.multiply(-1, m_att), q2, poly_mod, N)), [t_g[i]], q2, poly_mod, N)
#     temp1 = poly_add_3D(temp0, -1*h[i], q2, poly_mod, N)

#     print("w", w)
#     print("w[i]", w[i])
#     print("______________")
#     print(poly_matmul(challenge,temp1, q2, poly_mod, N).shape)
#     temp2 = poly_matmul(challenge,temp1, q2, poly_mod, N)
#     print("______________")
#     print(temp2.shape)
#     rhs = poly_add_3D(w[i], poly_matmul(challenge,temp2, q2, poly_mod, N),q2, poly_mod,N)
#     # temp2 = poly_add_3D([t_g[i]], -1*h[i], q2, poly_mod, N)
#     # rhs = poly_add_3D(temp1, temp3, q2, poly_mod, N)
#     print("FINAL RESULT")
#     print(lhs, rhs)
#     if lhs.all() != rhs.all():
#         print("failed")
#     # for i in range(len(lhs)):
#     #     for j in range(len(lhs[i])):
#     #         for k in range(len(lhs[i][j])):
#     #             if lhs[i][j][k] == rhs[i][j][k]:
#     #                 print("failed")
#     #                 break


# print("it's over")


# disclosure of properties attributes m0 OR
# xi = 20 #TODO
# N= 5
# d =5 
# k = 4
# k_hat = 2
# q2 = 2**79-67
# bound = xi*math.sqrt(2*k*N)
# B = gen_matrix_ring(k_hat, k, q2,N)

# r1 = generate_random(4,1,N,10000000) #beta is 1
# # message = gen_matrix_ring(1,1,q,N)
# message = [[[1,2,3,4,5]]]
# m_prime = [[[1-16,2,3,4,5]]]
# m_att = [[[16,0,0,0,0]]]
# A,D, q1, q2, N, k_prime, gamma, gamma_prime, alpha = cts_setup(n=1, k=4 ,l=1, l_prime=2,q1=75,q2=67,N=d)
# poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# comm = cts_commit(np.array(A), np.array(message), np.array(r1), d, poly_mod, q1, q2, n, l)
# t11 = comm[:1]
# t21 = comm[1:]

# print("**********")
# print(A.shape, A[1:])
# #CHEW
# # g = []
# #disclose first (0th) attribute
# g = gen_matrix_ring(1,1,q2,N)
# g[0][0][0] = 0
# for i in range(1, k_hat):
#     f = gen_matrix_ring(1,1,q2,N)
#     f[0][0][0] = f[0][0][0]*16-1 
#     g = np.concatenate((g, f))
# print(g)
# N = 5
# q2 = 999
# # poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
# print(B.shape, r1.shape, np.array(g).shape)
# tmp = poly_matmul(B, r1,q2,poly_mod, N)
# print("HERE")
# t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)

# gammas = [random.randrange(0, q2) for i in range(k_hat)]

# y_sampler = DiscreteGaussianDistributionIntegerSampler(sigma=xi)
# y = []
# for i in range(k):
#     y.append([[y_sampler()]*N])
# print("HERRRRRRRRRRRRRRRRRRRR")
# # y = [y]
# # print(y)

# w_bold = poly_matmul(A[:1], y, q1, poly_mod, N)
# h = []
# w = []
# for i in range(k_hat):
#     print(g[i], np.multiply(gammas[i],m_prime))
#     h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
#     print("FOR LOOP CHCKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
#     print(np.multiply(gammas[i],ring_transpose(A[1:])).shape) 
#     print(np.array(B[[i],:]).shape)
#     print(np.array(y).shape)
#     w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A[1:]),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
#     h.append(h_i)
#     w.append(w_i)
# print("&&&&&&&&&&&77")
# print(np.array(w).shape)
# print(w)
# # print(sdfd)

# k_challenge = 2
# # challenge = [random.randint(-5*k_challenge,5*k_challenge) for i in range(N)]
# # norm_inf = LA.norm(challenge, np.inf) #TODO
# # norm_one = LA.norm(challenge, 1)
# # while norm_inf != 1 and norm_one != k_challenge:
# #     coefficients = [random.randint(-5*k_challenge,k_challenge) for i in range(N)] 
# #     norm_inf = LA.norm(challenge, np.inf)
# #     norm_one = LA.norm(challenge, 1)
# challenge = [[[-1,0,1,0,1]]]
# # matrix = np.reshape(coefficients,(1,1,N))

# z = poly_add_3D_no_mod(ring_transpose(y), poly_matmul_no_mod(challenge,ring_transpose(r1),poly_mod, N), poly_mod, N)
# rej_outcome = rejection_sampling_1(z,poly_matmul_no_mod(challenge,ring_transpose(r1),poly_mod, N), xi, 6)
# print(rej_outcome)

# print(np.array(z).shape, w_bold)
# if LA.norm(z[0], 2) > bound or z != poly_add_3D(w_bold, poly_matmul(challenge,t11,q1, poly_mod, N)):
#     print("failed")

# for i in range(k_hat):
#     if h_i[0][0][0] != 0:
#         print("failed")
#     print("B SHAPE",np.array(B).shape)
#     print("B[1] SHAPE",np.array(B[[1],:]).shape)

#     lhs_1 = poly_add_3D_no_mod(np.multiply(gammas[i],ring_transpose(A[1:])), ring_transpose(B[[i],:]), poly_mod, N)
#     lhs = poly_matmul(lhs_1, z, q2, poly_mod, N)
#     print(lhs.shape)
#     temp0 = poly_add_3D(np.multiply(gammas[i], poly_add_3D(t21, np.multiply(-1, m_att), q2, poly_mod, N)), [t_g[i]], q2, poly_mod, N)
#     temp1 = poly_add_3D(temp0, -1*h[i], q2, poly_mod, N)

#     print("w", w)
#     print("w[i]", w[i])
#     print("______________")
#     print(poly_matmul(challenge,temp1, q2, poly_mod, N).shape)
#     temp2 = poly_matmul(challenge,temp1, q2, poly_mod, N)
#     print("______________")
#     print(temp2.shape)
#     rhs = poly_add_3D(w[i], poly_matmul(challenge,temp2, q2, poly_mod, N),q2, poly_mod,N)
#     # temp2 = poly_add_3D([t_g[i]], -1*h[i], q2, poly_mod, N)
#     # rhs = poly_add_3D(temp1, temp3, q2, poly_mod, N)
#     print("FINAL RESULT")
#     print(lhs, rhs)
#     if lhs.all() != rhs.all():
#         print("failed")
#     # for i in range(len(lhs)):
#     #     for j in range(len(lhs[i])):
#     #         for k in range(len(lhs[i][j])):
#     #             if lhs[i][j][k] == rhs[i][j][k]:
#     #                 print("failed")
#     #                 break


# print("it's over")

def disclosure_prover_1(B, r1):
    return

def disclosure_verifier_1():
    return

def disclosure_prover_2():
    return

def disclosure_verifier_2():
    return

def disclosure_prover_3():
    return

def disclosure_check():
    return