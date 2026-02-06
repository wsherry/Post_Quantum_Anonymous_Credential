
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ac import *
import time 
from sage.all import *
import random
from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionLatticeSampler
import csv

def write_file(file_name, field_names, data):
    # Write data to CSV after the loop
    with open(file_name, mode='w', newline='') as file:
        # Define the field names for the CSV header
        writer = csv.DictWriter(file, fieldnames=field_names)
        # Write the header
        writer.writeheader()
        # Write the data rows
        writer.writerows(data)

N = 512
eta = 7.6
k = 4
T = math.sqrt(2*k*N)
xi = eta*T
poly_mod = np.poly1d([1] + [0]*(N-1) + [1])

def verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g):

    for i in range(k_hat):
    
        lhs_1 = poly_add_3D(np.array(gammas[i])*np.array(A2), B[[i],:], q2, poly_mod, N)
        lhs_ = poly_matmul(lhs_1, z, q2, poly_mod, N) 
        lhs = mod_3D(len(lhs_),lhs_,q2)
        
        temp0 = poly_add_3D(np.array(gammas[i])*np.array(poly_add_3D(t21, -1*m_att, q2, poly_mod, N)), [t_g[i]], q2, poly_mod, N)
        temp1 = ring_transpose(poly_matmul(challenge, ring_transpose(poly_add_3D(temp0, [-1*h[i]], q2, poly_mod, N)), q2, poly_mod, N))
        rhs_ = poly_add_3D([w[i]], temp1, q2, poly_mod, N) 
        rhs = mod_3D(len(rhs_),rhs_,q2)

        if np.array_equal(lhs, rhs):
            return time.perf_counter(), True, 0  

    return time.perf_counter(), False, 4

def verifier_checks_2(k_hat, disclose_indices, check_zero, h):
    for i in range(k_hat):
        h_coeffs = [int(h[i][0][ind]) for ind in disclose_indices]

        if check_zero: #need to check that coefficients are all zeroes
            if not all(x == 0 for x in h_coeffs):
                print("h_coeffs", h_coeffs)
                return time.perf_counter(), False, 3
        else:
            if 0 in h_coeffs:
                print("h_coeffs", h_coeffs)
                return time.perf_counter(), False, 3
    return time.perf_counter(), True, 0  
    
def verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11):
    if math.sqrt(np.sum(np.inner(z,z))) > bound:
        return time.perf_counter(), False, 1

    res_1 = poly_matmul(A1,z, q1,poly_mod,N) 
    res_1 = mod_3D(len(res_1), res_1, q1)
    res_2 = poly_add_3D(np.array(w_bold), ring_transpose(poly_matmul(np.array(challenge),ring_transpose(np.array(t11)),q1, poly_mod, N)),q1, poly_mod, N) 
    res_2 = mod_3D(len(res_2), res_2, q1)

    if np.array_equal(res_1, res_2):
        return time.perf_counter(), True, 0  
    return time.perf_counter(), False, 2

def or_property_size_info(p_size, v_size, prover_sizes, verifier_sizes, index,random_index, bit, message, m_prime, m_att, val, other_val, A1, A2, q1, q2, N, k_hat, t11, t21, B, disclose_indices ,r1, bound, setup_start, protocol_start, num_attributes,G):

    rejection_fail_count = 0
    rejection_abort = True
    while rejection_abort:
        prover_gamma_prime = [random.randrange(0, q2) for i in range(k_hat)]
        g= []
        g_2 =[]

        for i in range(k_hat):
            f = gen_matrix_ring(1,1,q2,N)
            f_2 = gen_matrix_ring(1,1,q2,N)
            for j in disclose_indices:
                if random_index == 1: #False value OR True other_value
                    diff = val - other_val
                    f[0][0][j] = -1*diff*prover_gamma_prime[i] 
                    f_2[0][0][j] = 0
                else:
                    diff = val - other_val
                    f[0][0][j] = 0
                    f_2[0][0][j] = -1*diff*prover_gamma_prime[i] 
            f = mod_3D(len(f),f, q2)
            f_2 = mod_3D(len(f_2),f_2, q2)

            g.append(f[0])
            g_2.append(f_2[0])
        g = poly_matmul(g, np.array(G), q2, poly_mod,N)
        g_2 = poly_matmul(g_2, np.array(G), q2, poly_mod,N)

        tmp = poly_matmul(B, r1,q2,poly_mod, N)
        t_g_1 = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)
        t_g_2 =  poly_add_3D(tmp,np.array(g_2), q2, poly_mod, N)
        p_size += t_g_1.nbytes + t_g_2.nbytes 

        gammas = [random.randrange(0, q2) for i in range(k_hat)]
        gammas_double_prime = np.array(gammas) - np.array(prover_gamma_prime)
    
        v_size += np.array(gammas).nbytes 
        p_size += np.array(gammas_double_prime).nbytes + np.array(prover_gamma_prime).nbytes

        if random_index == 1:
            gammas_all = [prover_gamma_prime,gammas_double_prime]
        else:
            gammas_all = [gammas_double_prime, prover_gamma_prime]
        y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**N, sigma=xi)
        y = [[y_sampler(),y_sampler()] for j in range(k)]
        w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))

        h_1= []
        h_2 = []
        w_1 = []
        w_2 = []

        for i in range(k_hat):
            h_i_1 = poly_add_3D([g[i]], np.multiply(gammas_all[0][i],m_prime[0]), q2, poly_mod, N)
            h_i_2 = poly_add_3D([g_2[i]], np.multiply(gammas_all[1][i],m_prime[1]), q2, poly_mod, N)

            w_i_1 = poly_matmul(poly_add_3D(np.multiply(gammas_all[0][i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
            w_i_2 = poly_matmul(poly_add_3D(np.multiply(gammas_all[1][i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)

            h_1.append(h_i_1[0])
            h_2.append(h_i_2[0])
            w_1.append(w_i_1[0])
            w_2.append(w_i_2[0])

        p_size += w_bold.nbytes + np.array(h_1).nbytes+ np.array(h_2).nbytes + np.array(w_1).nbytes+ np.array(w_2).nbytes

        challenge = generate_random(2,2,N,2)
        v_size += np.array(challenge).nbytes

        z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))
        p_size += z.nbytes
        prover_sizes.append(p_size)
        verifier_sizes.append(v_size)        
        if rejection_sampling_1(z, poly_matmul(np.array(challenge),ring_transpose(np.array(r1)),q2, poly_mod, N), xi) == 0:
            rejection_fail_count = rejection_fail_count + 1
        else:
            rejection_abort = False
            time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
            
            for i in range(k_hat):
                if gammas[i] - prover_gamma_prime[i] != gammas_double_prime[i]:
                    rt_with_setup = time.perf_counter() - setup_start
                    rt = time.perf_counter() - protocol_start
                    return False, rt, rt_with_setup,{"Index": index, "Time":rt, "Time (with Setup)":rt_with_setup , "Rej Count":rejection_fail_count ,"Number Attributes": num_attributes,"Attribute Value":val, "Not Attribute Value":other_val, "Attribute Index":disclose_indices[0], "Success":False, "Error": 3.1,  "message": message, "m_prime_1":m_prime[0], "m_prime_2":m_prime[1], "m_att_1": m_att[0], "m_att_2": m_att[1]}
        

            if status == True:
                # we want to check that the coefficients are zero
                time_value, status, error = verifier_checks_2(k_hat, disclose_indices, True, h_1)
            if status == True:
                time_value, status, error = verifier_checks_2(k_hat, disclose_indices, True, h_2)
            if status == True:
                time_value, status, error = verifier_checks_3(k_hat,gammas_all[0], A2, B, q2, poly_mod, N, z, challenge, t21, m_att[0], h_1,w_1, t_g_1)
            if status == True:
                time_value, status, error = verifier_checks_3(k_hat, gammas_all[1], A2, B, q2, poly_mod, N, z, challenge, t21, m_att[1], h_2,w_2, t_g_2)
                      
            rt_with_setup = time_value - setup_start
            rt = time_value - protocol_start

            return prover_sizes, verifier_sizes, status, rt,rt_with_setup, {"Index": index, "Time":rt, "Time (with Setup)":rt_with_setup , "Rej Count":rejection_fail_count ,"Number Attributes": num_attributes,"Attribute Value":val, "Not Attribute Value":other_val, "Attribute Index":disclose_indices, "Success":status, "Error": error,  "message": message, "m_prime_1":m_prime[0], "m_prime_2":m_prime[1], "m_att_1": m_att[0], "m_att_2": m_att[1]}


def or_property(index,random_index, bit, message, m_prime, m_att, val, other_val, A1, A2, q1, q2, N, k_hat, t11, t21, B, disclose_indices ,r1, bound, setup_start, protocol_start, num_attributes,G):

    rejection_fail_count = 0
    rejection_abort = True
    while rejection_abort:
        prover_gamma_prime = [random.randrange(0, q2) for i in range(k_hat)]
        g= []
        g_2 =[]

        for i in range(k_hat):
            f = gen_matrix_ring(1,1,q2,N)
            f_2 = gen_matrix_ring(1,1,q2,N)
            for j in disclose_indices:
                if random_index == 1: #False value OR True other_value
                    diff = val - other_val
                    f[0][0][j] = -1*diff*prover_gamma_prime[i] 
                    f_2[0][0][j] = 0
                else:
                    diff = val - other_val
                    f[0][0][j] = 0
                    f_2[0][0][j] = -1*diff*prover_gamma_prime[i] 
            f = mod_3D(len(f),f, q2)
            f_2 = mod_3D(len(f_2),f_2, q2)

            g.append(f[0])
            g_2.append(f_2[0])
        g = poly_matmul(g, np.array(G), q2, poly_mod,N)
        g_2 = poly_matmul(g_2, np.array(G), q2, poly_mod,N)

        tmp = poly_matmul(B, r1,q2,poly_mod, N)
        t_g_1 = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)
        t_g_2 =  poly_add_3D(tmp,np.array(g_2), q2, poly_mod, N)

        gammas = [random.randrange(0, q2) for i in range(k_hat)]
        gammas_double_prime = np.array(gammas) - np.array(prover_gamma_prime)
        if random_index == 1:
            gammas_all = [prover_gamma_prime,gammas_double_prime]
        else:
            gammas_all = [gammas_double_prime, prover_gamma_prime]
        y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**N, sigma=xi)
        y = [[y_sampler(),y_sampler()] for j in range(k)]
        w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))

        h_1= []
        h_2 = []
        w_1 = []
        w_2 = []

        for i in range(k_hat):
            h_i_1 = poly_add_3D([g[i]], np.multiply(gammas_all[0][i],m_prime[0]), q2, poly_mod, N)
            h_i_2 = poly_add_3D([g_2[i]], np.multiply(gammas_all[1][i],m_prime[1]), q2, poly_mod, N)

            w_i_1 = poly_matmul(poly_add_3D(np.multiply(gammas_all[0][i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
            w_i_2 = poly_matmul(poly_add_3D(np.multiply(gammas_all[1][i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)

            h_1.append(h_i_1[0])
            h_2.append(h_i_2[0])
            w_1.append(w_i_1[0])
            w_2.append(w_i_2[0])

        challenge = generate_random(2,2,N,2)
        z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))
        
        if rejection_sampling_1(z, poly_matmul(np.array(challenge),ring_transpose(np.array(r1)),q2, poly_mod, N), xi) == 0:
            rejection_fail_count = rejection_fail_count + 1
        else:
            rejection_abort = False
            time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
            
            for i in range(k_hat):
                if gammas[i] - prover_gamma_prime[i] != gammas_double_prime[i]:
                    rt_with_setup = time.perf_counter() - setup_start
                    rt = time.perf_counter() - protocol_start
                    return False, rt, rt_with_setup,{"Index": index, "Time":rt, "Time (with Setup)":rt_with_setup , "Rej Count":rejection_fail_count ,"Number Attributes": num_attributes,"Attribute Value":val, "Not Attribute Value":other_val, "Attribute Index":disclose_indices[0], "Success":False, "Error": 3.1,  "message": message, "m_prime_1":m_prime[0], "m_prime_2":m_prime[1], "m_att_1": m_att[0], "m_att_2": m_att[1]}
        

            if status == True:
                # we want to check that the coefficients are zero
                time_value, status, error = verifier_checks_2(k_hat, disclose_indices, True, h_1)
            if status == True:
                time_value, status, error = verifier_checks_2(k_hat, disclose_indices, True, h_2)
            if status == True:
                time_value, status, error = verifier_checks_3(k_hat,gammas_all[0], A2, B, q2, poly_mod, N, z, challenge, t21, m_att[0], h_1,w_1, t_g_1)
            if status == True:
                time_value, status, error = verifier_checks_3(k_hat, gammas_all[1], A2, B, q2, poly_mod, N, z, challenge, t21, m_att[1], h_2,w_2, t_g_2)
                      
            rt_with_setup = time_value - setup_start
            rt = time_value - protocol_start

            return status, rt,rt_with_setup, {"Index": index, "Time":rt, "Time (with Setup)":rt_with_setup , "Rej Count":rejection_fail_count ,"Number Attributes": num_attributes,"Attribute Value":val, "Not Attribute Value":other_val, "Attribute Index":disclose_indices, "Success":status, "Error": error,  "message": message, "m_prime_1":m_prime[0], "m_prime_2":m_prime[1], "m_att_1": m_att[0], "m_att_2": m_att[1]}

def disclose(disclose_indices, m_prime,r1, bound, setup_start, protocol_start,  A1, q1, q2, num_attributes, vals,message, m_att,index,t11, t21, poly_mod,N,B,A, A2, k_hat):
    
    rejection_fail_count = 0
    rejection_abort = True
    while rejection_abort:
        g= []
        for i in range(k_hat):
            f = gen_matrix_ring(1,1,q2,N)
            for j in disclose_indices:
                f[0][0][j] = 0
            g.append(f[0])

        G = np.zeros((1,2,N))
        G[0][0][-1] = 1
        G[0][1][-1] = np.rint(np.sqrt(q2))
        g = poly_matmul(g, np.array(G), q2, poly_mod,N)

        tmp = poly_matmul(B, r1,q2,poly_mod, N)
        t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)

        gammas = [random.randrange(0, q2) for i in range(k_hat)]

        y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**N, sigma=xi)
        y = [[y_sampler(),y_sampler()] for i in range(k)]
        
        h = []
        w = []
        for i in range(k_hat):
            h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
            w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A[1:]),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
            h.append(h_i[0])
            w.append(w_i[0])
        w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))

        challenge = generate_random(2,2,N,2)

        z = np.array(poly_add_3D(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod, N))
        if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
            rejection_fail_count = rejection_fail_count + 1
        else:
            rejection_abort = False

            time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
            if status == True:
                time_value, status, error = verifier_checks_2(k_hat, disclose_indices, True, h)
            if status == True:
                time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g)

            rt_with_setup = time_value - setup_start
            rt = time_value - protocol_start
            return status, rt,rt_with_setup,{"Index": index, "Time":rt,  "Time(with Setup)": rt_with_setup, "Number Attributes": num_attributes, "Attribute Value":vals, "Attribute Index":disclose_indices, "Success":status, "Error": error, "message": message, "m_prime":m_prime, "m_att": m_att}

def multi_NOT_v2(num_attributes, k_hat, q2, N, diffs, disclose_indices, B, r1, poly_mod, m_prime, A1, A2, setup_start, protocol_start, index, vals, not_vals, message, m_att, bound, t11, t21, G, tau):
  
    rejection_fail_count = 0
    rejection_abort = True
    h_coeffs_zero = True
    h_coeffs_zero_count = 0
    while rejection_abort and h_coeffs_zero:  
        g = []
        for i in range(k_hat):
            f = gen_matrix_ring(1,1,q2,N)
            for j in range(num_attributes):
                while f[0][0][disclose_indices[j]] == 0:
                    f[0][0][disclose_indices[j]] = (diffs[j])*random.randint(-(q2-1)//2, (q2-1)//2)

            g.append(f[0])
        g = poly_matmul(g, np.array(G), q2, poly_mod,N)

        tmp = poly_matmul(B, r1,q2,poly_mod, N)
        t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)

        gammas = [random.randint(-(q2-1)//2, (q2-1)//2) for i in range(k_hat)]

        y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**Integer(N), sigma=xi)
        y = [[y_sampler(),y_sampler()] for i in range(k)]

        h = []
        w = []
        for i in range(k_hat):
            h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
            w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
            h.append(h_i[0])
            w.append(w_i[0])
        
        w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))

        challenge = generate_random(2,2,N,2)

        z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))

        if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
            rejection_fail_count += 1
        else:
            rejection_abort = False

            time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
            if status == True:
                time_value, status, error = verifier_checks_2(k_hat, disclose_indices, False, h)
                if status == False:
                    h_coeffs_zero_count += 1
                    continue
            if status == True:
                h_coeffs_zero = False
                time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g)
    
            rt_with_setup = time_value - setup_start
            rt = time_value - protocol_start
            entry = {"Index": index, "Time":rt,"Time (with Setup)":rt_with_setup , "Rej Count":rejection_fail_count , "Number Attributes": num_attributes,"Attribute Value":vals, "Not Attribute Value": not_vals, "Attribute Index":disclose_indices , "Success":status, "Error": error, "message": message, "m_prime_1":"","m_prime_2": m_prime, "m_att_1": "", "m_att_2": m_att}

            return status, rt, rt_with_setup, entry
        
def multi_NOT(num_attributes, k_hat, q2, N, diffs, disclose_indices, B, r1, poly_mod, m_prime, A1, A2, setup_start, protocol_start, index, vals, not_vals, message, m_att, bound, t11, t21, G, tau):
    rejection_fail_count = 0
    rejection_abort = True

    while rejection_abort:   
    
        g= []

        for i in range(k_hat):
            f = gen_matrix_ring(1,1,q2,N)
            for j in range(num_attributes):
                f[0][0][disclose_indices[j]] = (diffs[j])*f[0][0][disclose_indices[j]]
            g.append(f[0])
        g = poly_matmul(g, np.array(G), q2, poly_mod,N)

        tmp = poly_matmul(B, r1,q2,poly_mod, N)
        t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)

        gammas = [random.randrange(0, q2) for i in range(k_hat)]

        y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**Integer(N), sigma=xi)
        y = [[y_sampler(),y_sampler()] for i in range(k)]
        
        h = []
        w = []
        for i in range(k_hat):
            h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
            w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
            h.append(h_i[0])
            w.append(w_i[0])
        
        w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))
        challenge = generate_random(2,2,N,2)

        z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))
        if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
            rejection_fail_count = rejection_fail_count + 1   
        else:
            rejection_abort = False
            time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
            if status == True:
                time_value, status, error = verifier_checks_2(k_hat, disclose_indices, False, h)
            if status == True:
                time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g)

            rt_with_setup = time_value - setup_start
            rt = time_value - protocol_start

            return status, rt, rt_with_setup, {"Index": index, "Time":rt,"Time (with Setup)":rt_with_setup , "Rej Count":rejection_fail_count , "Number Attributes": num_attributes,"Attribute Value":vals, "Not Attribute Value": not_vals, "Attribute Index":disclose_indices , "Success":status, "Error": error, "message": message, "m_prime_1":"","m_prime_2": m_prime, "m_att_1": "", "m_att_2": m_att}
        
def not_property(index, setup_start, protocol_start, r1, m_prime, A1, A2, t21, bound, num_attributes, vals, not_vals, message, m_att, t11, B, k_hat,q2,N,A,poly_mod,q1,disclose_indices ): 
    
    g = []
    for i in range(k_hat):
        f = gen_matrix_ring(1,1,q2,N)
        for j in range(len(disclose_indices)):
            f[0][0][disclose_indices[j]] = (not_vals[j]-vals[j])*f[0][0][disclose_indices[j]]
        g.append(f[0])
    tmp = poly_matmul(B, r1,q2,poly_mod, N)
    t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)

    gammas = [random.randrange(0, q2) for i in range(k_hat)]

    y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**Integer(N), sigma=xi)
    y = []
    for i in range(k):
        y.append([y_sampler()])
    h = []
    w = []
    for i in range(k_hat):
        h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
        w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
        h.append(h_i[0])
        w.append(w_i[0])

    w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))

    challenge = generate_random(2,2,N,2)

    z = np.array(poly_add_3D(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod, N))
    if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
        rt_with_setup = time.perf_counter() - setup_start
        rt = time.perf_counter() - protocol_start
        return False, rt, rt_with_setup, {"Index": index, "Time":rt,"Time (with Setup)":rt_with_setup ,"Number Attributes": num_attributes,"Attribute Value":vals, "Not Attribute Value": not_vals, "Attribute Index":disclose_indices , "Success":False, "Error": -2.1, "message": message, "m_prime_1":"","m_prime_2": m_prime, "m_att_1": "", "m_att_2": m_att}
   
    time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
    if status == True:
        time_value, status, error = verifier_checks_2(k_hat, disclose_indices, False, h)
    if status == True:
        time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g)
  
    rt_with_setup = time_value - setup_start
    rt = time_value - protocol_start
    return status, rt, rt_with_setup, {"Index": index, "Time":rt,"Time (with Setup)":rt_with_setup ,"Number Attributes": num_attributes,"Attribute Value":vals, "Not Attribute Value": not_vals, "Attribute Index":disclose_indices , "Success":status, "Error": error, "message": message, "m_prime_1":"","m_prime_2": m_prime, "m_att_1": "", "m_att_2": m_att}
