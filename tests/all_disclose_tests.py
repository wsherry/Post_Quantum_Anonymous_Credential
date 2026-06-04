
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ac import *
import csv
import time
from sage.all import *
import random
from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionLatticeSampler
from tests.helpers import *
import sys
from itertools import chain

np.set_printoptions(threshold=sys.maxsize)
foldername = "RESULTS"
N = 512
eta = 7.6
tau = 2
# eta = 19.32
k = 4
T = math.sqrt(2*k*N)
xi = eta*T
d = 128
t = 4
k_hat = 2
q1 = 4294966997
q2 = 67108837
# q2 = 101
bound = xi*T*2
poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
n = 1  
l = 1
l_prime = 2
# G = 1
G = np.zeros((1,2,N))
G[0][0][-1] = 1
G[0][1][-1] = np.rint(np.sqrt(q2))

# def get_random_vals_NOT_OR(num_attributes, max_val, m=100):
#     random_vals = [np.random.choice(m, 2, replace=False) for att in range(num_attributes)]
#     vals = [vals[0] for vals in random_vals]
#     not_vals =  [vals[1] for vals in random_vals]
#     diffs = np.subtract(np.array(vals), np.array(not_vals))
#     # for i in range(num_attributes):
#         # diffs.append(vals[i] - not_vals[i])
#     disclose_indices =  np.random.choice(max_val, num_attributes, replace=False)
#     return vals, not_vals, diffs, disclose_indices

def run_multi_NOT(num_attributes, iterations = 100):
    run_times_setup =[]
    run_times = []
    data = []
    prover_sizes = []
    verifier_sizes = []
    message_sizes = []
    for index in range(iterations):
        p_size = 0
        v_size = 0
        setup_start = time.perf_counter()
        r1 = generate_random(k,l*tau,N,2) 
        random_vals = [np.random.choice(100, 2, replace=False) for att in range(num_attributes)]
        vals = [vals[0] for vals in random_vals]
        not_vals =  [vals[1] for vals in random_vals]
        diffs = np.subtract(np.array(vals), np.array(not_vals))
        disclose_indices =  np.random.choice(N, num_attributes, replace=False)
        message = np.random.choice(2, (1,1,N))
        for i in range(len(disclose_indices)):
            message[0][0][int(disclose_indices[i])] = vals[i]
        m_prime = message.copy()
        m_att = [[[0]*N]]
        for i in range(len(disclose_indices)):
            m_prime[0][0][int(disclose_indices[i])] = diffs[i]
            m_att[0][0][int(disclose_indices[i])] = not_vals[i]
        
        message_sizes.append(np.array(message).nbytes)
        
        m_prime = poly_matmul(np.array(m_prime), G, q2, poly_mod, N)
        m_att = poly_matmul(np.array(m_att),G, q2, poly_mod, N)
        A,D, q1_val, q2_val, N_val, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=l, l_prime=l_prime,q1=q1,q2=q2,N=N)
        B = gen_matrix_ring(k_hat, k, q2,N)
        A1 = A[:n, :]
        A2 = A[n:,:]
        comm = cts_commit(np.array(A), np.array(message), np.array(r1),G, N, poly_mod, q1, q2, n, l, tau = tau)
        t11 = comm[:n]
        t21 = comm[n:]

        protocol_start = time.perf_counter()

        rejection_fail_count = 0
        rejection_abort = True
        h_coeffs_zero = True
        h_coeffs_zero_count = 0
        # print("_______________")
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
            p_size += t_g.nbytes

            gammas = [random.randint(0,q2) for i in range(k_hat)]
            v_size += np.array(gammas).nbytes

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
            p_size += w_bold.nbytes + np.array(h).nbytes + np.array(w).nbytes

            challenge = generate_random(l*tau,l*tau,N,2)
            v_size += np.array(challenge).nbytes

            # z = np.array(poly_add_3D(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod, N))
            z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))
            p_size += z.nbytes
            prover_sizes.append(p_size)
            verifier_sizes.append(v_size)
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
                data.append({"Index": index, "Prover size": p_size, "Verifier size": v_size, "Message size": np.array(message).nbytes, "Time":rt, "Time (with Setup)":rt_with_setup ,"Rej Count": rejection_fail_count, "H Coeffs Rerun": h_coeffs_zero_count, "Number Attributes": num_attributes,"Attribute Value":vals, "Not Attribute Value": not_vals, "Attribute Index":disclose_indices , "Success":status, "Error": error, "message": message, "m_prime":m_prime, "m_att": m_att})
                run_times.append(rt)
                run_times_setup.append(rt_with_setup)            


    avg_not = np.average(np.array(run_times))
    med_not = np.median(np.array(run_times))
    min_not = np.min(np.array(run_times))
    max_not = np.max(np.array(run_times))
    avg_not_setup = np.average(np.array(run_times_setup))
    med_not_setup = np.median(np.array(run_times_setup))
    min_not_setup = np.min(np.array(run_times_setup))
    max_not_setup = np.max(np.array(run_times_setup))

    p_avg = np.average(np.array(prover_sizes))
    p_med = np.median(np.array(prover_sizes))
    p_min = np.min(np.array(prover_sizes))
    p_max = np.max(np.array(prover_sizes))

    v_avg = np.average(np.array(verifier_sizes))
    v_med = np.median(np.array(verifier_sizes))
    v_min = np.min(np.array(verifier_sizes))
    v_max = np.max(np.array(verifier_sizes))

    m_avg = np.average(np.array(message_sizes))
    m_med = np.median(np.array(message_sizes))
    m_min = np.min(np.array(message_sizes))
    m_max = np.max(np.array(message_sizes))
    print("--------------------------------------------------------------------------")
    print("NOT RESULTS with multi attributes: ", num_attributes)
    print("NO SETUP TIME: (avg, med, min, max)", avg_not, med_not, min_not, max_not)
    print("WITH SETUP TIME: (avg, med, min, max)", avg_not_setup, med_not_setup, min_not_setup, max_not_setup)
    print("PROVER SIZES avg, med, min, max: ",p_avg, p_med, p_min, p_max)
    print("VERIFIER SIZES avg, med, min, max: ", v_avg,v_med, v_min, v_max)
    print("MESSAGE SIZES avg, med, min, max: ", m_avg,m_med, m_min, m_max)

    return  run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes

def run_NOT_false(iterations = 100):

    run_times_setup =[]
    run_times = []
    data = []
    prover_sizes = []
    verifier_sizes = []
    message_sizes = []
    for index in range(iterations):
        p_size = 0
        v_size = 0
        setup_start = time.perf_counter()
        r1 = generate_random(k,l*tau,N,2) 
        random_vals = np.random.choice(100, 1, replace=False)
        val = random_vals[0]
        not_val = val
        diff = val - not_val
        disclose_indices =  np.random.choice(N, 1, replace=False)
        message = np.random.choice(2, (1,1,N))
        message[0][0][int(disclose_indices[0])] = val
        m_prime = message.copy()
        m_prime[0][0][int(disclose_indices[0])] = diff
        m_att = [[[0]*N]]
        m_att[0][0][int(disclose_indices[0])] = not_val
        m_prime = np.array(m_prime)
        m_att = np.array(m_att)            
        m_att = poly_matmul(m_att, G, q2, poly_mod, N)
        m_prime = poly_matmul(m_prime,G, q2, poly_mod, N)
        
        A,D, q1_val, q2_val, N_deg, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=l, l_prime=l_prime,q1=q1,q2=q2,N=N)
        B = gen_matrix_ring(k_hat, k, q2,N)
        A1 = A[:n, :]
        A2 = A[n:,:]
        comm = cts_commit(np.array(A), np.array(message), np.array(r1),G, N, poly_mod, q1, q2, n, l,tau = tau)
        t11 = comm[:n]
        t21 = comm[n:]
        message_sizes.append(np.array(message).nbytes)

        protocol_start = time.perf_counter()

        rejection_fail_count = 0
        rejection_abort = True

        while rejection_abort:  
            g= []

            for i in range(k_hat):
                f = gen_matrix_ring(1,1,q2,N)
                for j in disclose_indices:
                    f[0][0][j] = diff*f[0][0][j]
                g.append(f[0])
            g = poly_matmul(g, np.array(G), q2, poly_mod,N)

            tmp = poly_matmul(B, r1,q2,poly_mod, N)
            t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)
            p_size += t_g.nbytes 

            gammas = [random.randrange(0, q2) for i in range(k_hat)]
            v_size += np.array(gammas).nbytes

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
            p_size += w_bold.nbytes + np.array(h).nbytes + np.array(w).nbytes

            # challenge = generate_random(1,1,N,1)
            challenge = generate_random(l*tau,l*tau,N,2)
            v_size += np.array(challenge).nbytes

            z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))
            p_size += z.nbytes
            prover_sizes.append(p_size)
            verifier_sizes.append(v_size)
            
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
                data.append({"Index": index, "Prover size": p_size, "Verifier size": v_size, "Message size": np.array(message).nbytes,"Time":rt, "Time (with Setup)":rt_with_setup , "Rej Count": rejection_fail_count, "Attribute Value":val, "Not Attribute Value": not_val, "Attribute Index":disclose_indices , "Success":status, "Error": error, "message": message, "m_prime":m_prime, "m_att": m_att})
                run_times.append(rt)
                run_times_setup.append(rt_with_setup)   

    avg_not = np.average(np.array(run_times))
    med_not = np.median(np.array(run_times))
    min_not = np.min(np.array(run_times))
    max_not = np.max(np.array(run_times))
    avg_not_setup = np.average(np.array(run_times_setup))
    med_not_setup = np.median(np.array(run_times_setup))
    min_not_setup = np.min(np.array(run_times_setup))
    max_not_setup = np.max(np.array(run_times_setup))

    
    p_avg = np.average(np.array(prover_sizes))
    p_med = np.median(np.array(prover_sizes))
    p_min = np.min(np.array(prover_sizes))
    p_max = np.max(np.array(prover_sizes))

    v_avg = np.average(np.array(verifier_sizes))
    v_med = np.median(np.array(verifier_sizes))
    v_min = np.min(np.array(verifier_sizes))
    v_max = np.max(np.array(verifier_sizes))

    m_avg = np.average(np.array(message_sizes))
    m_med = np.median(np.array(message_sizes))
    m_min = np.min(np.array(message_sizes))
    m_max = np.max(np.array(message_sizes))
 
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print("=========NOT FALSE attributes: ")
    print("NO SETUP TIME: (avg, med, min, max)", avg_not, med_not, min_not, max_not)
    print("WITH SETUP TIME: (avg, med, min, max)", avg_not_setup, med_not_setup, min_not_setup, max_not_setup)
    print("PROVER SIZES avg, med, min, max: ",p_avg, p_med, p_min, p_max)    
    print("VERIFIER SIZES avg, med, min, max: ", v_avg,v_med, v_min, v_max)
    print("MESSAGE SIZES avg, med, min, max: ", m_avg,m_med, m_min, m_max)

    return run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes

def run_multi_OR_random(num_attributes, iterations = 100):
    data = []
    run_times = []
    run_times_setup = []
    prover_sizes = []
    verifier_sizes = []
    message_sizes = []
    for index in range(iterations):
        # print(index)
        setup_start = time.perf_counter()
        r1 = generate_random(k,l*tau,N,2) 
        # vals, other_vals, diffs, disclose_indices = get_random_vals_NOT_OR(num_attributes,N)
        random_vals = [np.random.choice(100, 2, replace=False) for att in range(num_attributes)]
        vals = [vals[0] for vals in random_vals]
        other_vals =  [vals[1] for vals in random_vals]
        # for i in range(num_attributes):
            # diffs.append(vals[i] - not_vals[i])
        disclose_indices =  np.random.choice(N, num_attributes, replace=False)
        message = np.random.choice(2, (1,1,N))
        random_indices = np.random.choice(2, num_attributes)
        for i in range(num_attributes):
            message[0][0][int(disclose_indices[i])] = vals[i]
        message_sizes.append(np.array(message).nbytes)

        A,D, q1_val, q2_val, N_deg, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=l, l_prime=l_prime,q1=q1,q2=q2,N=N)
        A1 = A[:n, :]
        A2 = A[n:,:]
        comm = cts_commit(np.array(A), np.array(message), np.array(r1),G, N, poly_mod, q1, q2, n, l,tau = tau)
        t11 = comm[:n]
        t21 = comm[n:]
        B = gen_matrix_ring(k_hat, k, q2,N)
        p_size = 0
        v_size = 0
        count = 0
        protocol_start = time.perf_counter()

        for i in range(num_attributes):
            # prover_gamma_prime = [random.randrange(0, q2) for j in range(k_hat)]
            # p_size += np.array(prover_gamma_prime).nbytes

            m_prime_1 = message.copy()
            m_prime_2 = message.copy()
            if random_indices[i] == 1: #False value OR True value
                # diff = other_vals[i] - vals[i]
                diff = vals[i] - other_vals[i]

                m_prime_1[0][0][int(disclose_indices[i])] = diff
                m_prime_2[0][0][int(disclose_indices[i])] = 0
            else:
                diff = vals[i] - other_vals[i]
                m_prime_1[0][0][int(disclose_indices[i])] = 0
                m_prime_2[0][0][int(disclose_indices[i])] = diff
            m_att_1 = [[[0]*N]]
            m_att_2 = [[[0]*N]]

            if random_indices[i] == 1: #False value OR True value
                m_att_1[0][0][int(disclose_indices[i])] = other_vals[i]
                m_att_2[0][0][int(disclose_indices[i])] = vals[i]  
            else:
                m_att_1[0][0][int(disclose_indices[i])] = vals[i]
                m_att_2[0][0][int(disclose_indices[i])] = other_vals[i]  

            m_att_1 = poly_matmul(m_att_1, G, q2, poly_mod, N)
            m_att_2 = poly_matmul(m_att_2,G, q2, poly_mod, N)
            m_prime_1 = poly_matmul(m_prime_1, G, q2, poly_mod, N)
            m_prime_2 = poly_matmul(m_prime_2,G, q2, poly_mod, N)
            m_att = [m_att_1, m_att_2]
            m_prime = [m_prime_1, m_prime_2]


            prover_sizes, verifier_sizes, p_size,v_size,success_flag, rt, rt_with_setup, entry = or_property_info(p_size, v_size,prover_sizes,verifier_sizes,index, random_indices[i], count, message, m_prime, m_att, vals[i], other_vals[i], A1, A2, q1, q2, N, k_hat, t11, t21, B, [disclose_indices[i]], r1, bound, setup_start,protocol_start, num_attributes,G)
            count = count + 1

            if success_flag == False:
                data.append(entry)
                run_times.append(rt)
                run_times_setup.append(rt_with_setup)   
                break
        else:
            # if len(data) < index +1:
            rt_with_setup = time.perf_counter() - setup_start
            rt = time.perf_counter() - protocol_start
            data.append({"Index": index, "Prover size": p_size, "Verifier size": v_size, "Message size": np.array(message).nbytes,"Time":rt, "Time (with Setup)":rt_with_setup ,"Rej Count": 0, "Number Attributes": num_attributes,"Attribute Value":vals, "Not Attribute Value":other_vals, "Attribute Index":disclose_indices, "Success":True, "Error": -1, "message": message, "m_prime_1":m_prime_1, "m_prime_2": m_prime_2, "m_att_1": m_att_1, "m_att_2":m_att_2})

            run_times.append(rt) 
            run_times_setup.append(rt_with_setup)   



    avg_not = np.average(np.array(run_times))
    med_not = np.median(np.array(run_times))
    min_not = np.min(np.array(run_times))
    max_not = np.max(np.array(run_times))
    avg_not_setup = np.average(np.array(run_times_setup))
    med_not_setup = np.median(np.array(run_times_setup))
    min_not_setup = np.min(np.array(run_times_setup))
    max_not_setup = np.max(np.array(run_times_setup))
    p_avg = np.average(np.array(prover_sizes))
    p_med = np.median(np.array(prover_sizes))
    p_min = np.min(np.array(prover_sizes))
    p_max = np.max(np.array(prover_sizes))

    v_avg = np.average(np.array(verifier_sizes))
    v_med = np.median(np.array(verifier_sizes))
    v_min = np.min(np.array(verifier_sizes))
    v_max = np.max(np.array(verifier_sizes))

    m_avg = np.average(np.array(message_sizes))
    m_med = np.median(np.array(message_sizes))
    m_min = np.min(np.array(message_sizes))
    m_max = np.max(np.array(message_sizes))


    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("==========OR RESULTS with multi attributes: ", num_attributes)
    print("NO SETUP TIME: (avg, med, min, max)", avg_not, med_not, min_not, max_not)
    print("WITH SETUP TIME: (avg, med, min, max)", avg_not_setup, med_not_setup, min_not_setup, max_not_setup)
    print("PROVER SIZES avg, med, min, max: ",p_avg, p_med, p_min, p_max)
    print("VERIFIER SIZES avg, med, min, max: ", v_avg,v_med, v_min, v_max)
    print("MESSAGE SIZES avg, med, min, max: ", m_avg,m_med, m_min, m_max)
    return  run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes

def run_OR_false(iterations=100):
    data = []
    run_times = []
    run_times_setup = []
    prover_sizes = []
    verifier_sizes = []
    message_sizes = []
    for index in range(iterations):
        # print(index)
        p_size = 0
        v_size = 0
        setup_start = time.perf_counter()
        r1 = generate_random(k,l*tau,N,2) 
        random_vals = np.random.choice(100, 3, replace=False)
        random_indices = np.random.choice(2, 1)
        actual_val = random_vals[0]
        val = random_vals[1]
        other_val = random_vals[2]
        disclose_indices = np.random.choice(N, 1, replace=False)
        diff = val - other_val
        message = np.random.choice(2, (1,1,N))
        message[0][0][int(disclose_indices[0])] = actual_val
        message_sizes.append(np.array(message).nbytes)

        m_prime_1 = message.copy()
        m_prime_2 = message.copy()
        m_att_1 = [[[0]*N]]
        m_att_2 = [[[0]*N]]
        if random_indices[0] == 1: #False value OR True value
            m_prime_1[0][0][int(disclose_indices[0])] = diff
            m_prime_2[0][0][int(disclose_indices[0])] = 0
            m_att_1[0][0][int(disclose_indices[0])] = other_val
            m_att_2[0][0][int(disclose_indices[0])] = val
        else:
            m_prime_1[0][0][int(disclose_indices[0])] = 0
            m_prime_2[0][0][int(disclose_indices[0])] = diff
            m_att_1[0][0][int(disclose_indices[0])] = val
            m_att_2[0][0][int(disclose_indices[0])] = other_val 

        m_att_1 = poly_matmul(m_att_1, G, q2, poly_mod, N)
        m_att_2 = poly_matmul(m_att_2,G, q2, poly_mod, N)
        m_prime_1 = poly_matmul(m_prime_1, G, q2, poly_mod, N)
        m_prime_2 = poly_matmul(m_prime_2,G, q2, poly_mod, N)
        m_att = [m_att_1, m_att_2]
        m_prime = [m_prime_1, m_prime_2]
        B = gen_matrix_ring(k_hat, k, q2,N)

        message_sizes.append(np.array(message).nbytes)        

        A,D, q1_val, q2_val, N_deg, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=l, l_prime=l_prime,q1=q1,q2=q2,N=N)
        A1 = A[:n, :]
        A2 = A[n:,:]
        comm = cts_commit(np.array(A), np.array(message), np.array(r1), G,N, poly_mod, q1, q2, n, l,tau = tau)
        t11 = comm[:n]
        t21 = comm[n:]
        protocol_start = time.perf_counter()

        count = 1
        prover_sizes, verifier_sizes,p_size,v_size, success_flag, rt, rt_with_setup, entry = or_property_info(p_size, v_size, prover_sizes, verifier_sizes,index, random_indices[0], count, message, m_prime, m_att, val, other_val, A1, A2, q1, q2, N, k_hat, t11, t21, B, disclose_indices, r1, bound, setup_start,protocol_start, 1,G)
        entry['Actual Value'] = actual_val
        rt_with_setup = time.perf_counter() - setup_start
        rt = time.perf_counter() - protocol_start
        data.append(entry)
        run_times.append(rt) 
        run_times_setup.append(rt_with_setup)   

    avg = np.average(np.array(run_times))
    med = np.median(np.array(run_times))
    min = np.min(np.array(run_times))
    max = np.max(np.array(run_times))
    avg_setup = np.average(np.array(run_times_setup))
    med_setup = np.median(np.array(run_times_setup))
    min_setup = np.min(np.array(run_times_setup))
    max_setup = np.max(np.array(run_times_setup))

    p_avg = np.average(np.array(prover_sizes))
    p_med = np.median(np.array(prover_sizes))
    p_min = np.min(np.array(prover_sizes))
    p_max = np.max(np.array(prover_sizes))

    v_avg = np.average(np.array(verifier_sizes))
    v_med = np.median(np.array(verifier_sizes))
    v_min = np.min(np.array(verifier_sizes))
    v_max = np.max(np.array(verifier_sizes))

    m_avg = np.average(np.array(message_sizes))
    m_med = np.median(np.array(message_sizes))
    m_min = np.min(np.array(message_sizes))
    m_max = np.max(np.array(message_sizes))

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print("==========OR FALSE ATTRIBUTE===============================")
    print("NO SETUP TIME: (avg, med, min, max)", avg, med, min, max)
    print("WITH SETUP TIME: (avg, med, min, max)", avg_setup, med_setup, min_setup, max_setup)
    print("PROVER SIZES avg, med, min, max: ",p_avg, p_med, p_min, p_max)
    print("VERIFIER SIZES avg, med, min, max: ", v_avg,v_med, v_min, v_max)
    print("MESSAGE SIZES avg, med, min, max: ", m_avg,m_med, m_min, m_max)

    return  run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes

def run_multi_disclose(num_attributes, iterations=100):

    data = []
    run_times = []
    run_times_setup = []
    prover_sizes = []
    verifier_sizes = []
    message_sizes = []
    for index in range(iterations):
        # print(index)
        setup_start = time.perf_counter()
        p_size = 0
        v_size = 0
        B = gen_matrix_ring(k_hat, k, q2,N)
        r1 = generate_random(k,l*tau,N,2) 
        # to disclose 3 (ie first two indices):
        disclose_indices = np.random.choice(N, num_attributes, replace=False)
        vals = np.random.choice(100, num_attributes, replace=False)

        message = np.random.choice(2, (1,1,N))
        for i in range(num_attributes):
            message[0][0][int(disclose_indices[i])] = vals[i]
            # message[0][0][int(disclose_indices[i])] = 999
        message_sizes.append(np.array(message).nbytes)

        m_prime = message.copy()
        for i in range(num_attributes):
            m_prime[0][0][int(disclose_indices[i])] = 0
        m_att = [[[0]*N]]
        for i in range(num_attributes):
            m_att[0][0][int(disclose_indices[i])] = vals[i]

        m_prime = poly_matmul(np.array(m_prime), G, q2, poly_mod, N)
        m_att = poly_matmul(np.array(m_att),G, q2, poly_mod, N)
        A,D, q1_val, q2_val, N_deg, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=l, l_prime=l_prime,q1=q1,q2=q2,N=N)
        A1 = A[:n, :]
        A2 = A[n:,:]
        comm = cts_commit(np.array(A), np.array(message), np.array(r1),G, N, poly_mod, q1, q2, n, l,tau = tau)
        t11 = comm[:n]
        t21 = comm[n:]

        protocol_start = time.perf_counter()

        rejection_fail_count = 0
        rejection_abort = True

        while rejection_abort:  
            g= []
            for i in range(k_hat):
                f = gen_matrix_ring(1,1,q2,N)
                for j in disclose_indices:
                    f[0][0][j] = 0
                g.append(f[0])
            g = poly_matmul(g, np.array(G), q2, poly_mod,N)

            tmp = poly_matmul(B, r1,q2,poly_mod, N)
            t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)
            p_size += t_g.nbytes


            gammas = [random.randrange(0, q2) for i in range(k_hat)]
            v_size += np.array(gammas).nbytes

            y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**N, sigma=xi)
            y = [[y_sampler(),y_sampler()] for i in range(k)]

            h = []
            w = []
            for i in range(k_hat):
                h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
                w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
                h.append(h_i[0])
                w.append(w_i[0])
            w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N)) 
            p_size += w_bold.nbytes + np.array(h).nbytes + np.array(w).nbytes

            challenge = generate_random(l*tau,l*tau,N,2)
            v_size += challenge.nbytes

            z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))
            p_size += z.nbytes
            prover_sizes.append(p_size)
            verifier_sizes.append(v_size)
            if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
                rejection_fail_count = rejection_fail_count + 1

            else: 
                rejection_abort = False
                time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
                if status == True:
                    time_value, status, error = verifier_checks_2(k_hat, disclose_indices, True, h)
                if status == True:
                    # time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g)
                    time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g)

                rt_with_setup = time_value - setup_start
                rt = time_value - protocol_start
                data.append({"Index": index, "Prover size": p_size, "Verifier size": v_size, "Message size": np.array(message).nbytes,"Time":rt,  "Time(with Setup)": rt_with_setup, "Rej Count": rejection_fail_count, "Number Attributes": num_attributes, "Attribute Value":vals, "Attribute Index":disclose_indices, "Success":status, "Error": error, "message": message, "m_prime":m_prime, "m_att": m_att})
                run_times.append(rt)            
                run_times_setup.append(rt_with_setup)

    avg_not = np.average(np.array(run_times))
    med_not = np.median(np.array(run_times))
    min_not = np.min(np.array(run_times))
    max_not = np.max(np.array(run_times))
    avg_not_setup = np.average(np.array(run_times_setup))
    med_not_setup = np.median(np.array(run_times_setup))
    min_not_setup = np.min(np.array(run_times_setup))
    max_not_setup = np.max(np.array(run_times_setup))

    p_avg = np.average(np.array(prover_sizes))
    p_med = np.median(np.array(prover_sizes))
    p_min = np.min(np.array(prover_sizes))
    p_max = np.max(np.array(prover_sizes))

    v_avg = np.average(np.array(verifier_sizes))
    v_med = np.median(np.array(verifier_sizes))
    v_min = np.min(np.array(verifier_sizes))
    v_max = np.max(np.array(verifier_sizes))

    m_avg = np.average(np.array(message_sizes))
    m_med = np.median(np.array(message_sizes))
    m_min = np.min(np.array(message_sizes))
    m_max = np.max(np.array(message_sizes))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("==========DISCLOSE RESULTS with multi attributes: ", num_attributes)
    print("NO SETUP TIME: (avg, med, min, max)", avg_not, med_not, min_not, max_not)
    print("WITH SETUP TIME: (avg, med, min, max)", avg_not_setup, med_not_setup, min_not_setup, max_not_setup)
    print("PROVER SIZES avg, med, min, max: ",p_avg, p_med, p_min, p_max)
    print("VERIFIER SIZES avg, med, min, max: ", v_avg,v_med, v_min, v_max)
    print("MESSAGE SIZES avg, med, min, max: ", m_avg,m_med, m_min, m_max)
    return  run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes

def disclose_false(iterations=100):

    data = []
    run_times = []
    run_times_setup = []
    prover_sizes = []
    verifier_sizes = []
    message_sizes = []
    for index in range(iterations):
        setup_start = time.perf_counter()
        p_size = 0
        v_size = 0
        r1 = generate_random(k,l*tau,N,2) 

        disclose_index =  np.random.choice(N, 1, replace=False)[0]
        random_vals = np.random.choice(np.arange(1,100), 2, replace=False)

        actual_val = random_vals[0]
        val = random_vals[1]
        message = np.random.choice(2, (1,1,N))
        message[0][0][int(disclose_index)] = actual_val
        message_sizes.append(np.array(message).nbytes)
       
        m_prime = message.copy()
        m_prime[0][0][int(disclose_index)] = 0
        m_att = [[[0]*N]]
        m_att[0][0][int(disclose_index)] = val
 
        m_att = poly_matmul(np.array(m_att),G, q2, poly_mod, N)#.astype(int)
        m_prime = poly_matmul(np.array(m_prime), G, q2, poly_mod, N)#.astype(int)
        B = gen_matrix_ring(k_hat, k, q2,N)

        A,D, q1_val, q2_val, N_deg, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=l, l_prime=l_prime,q1=q1,q2=q2,N=N)
        A1 = A[:n, :]
        A2 = A[n:,:]
        comm = cts_commit(np.array(A), np.array(message), np.array(r1), G,N, poly_mod, q1, q2, n, l,tau = tau)
        t11 = comm[:n]
        t21 = comm[n:]
        protocol_start = time.perf_counter()

        rejection_fail_count = 0
        rejection_abort = True

        while rejection_abort:
            g = []        
            for i in range(k_hat):
                f = gen_matrix_ring(1,1,q2,N)
                f[0][0][int(disclose_index)] = 0
                f = mod_3D(len(f),f, q2)
                g.append(f[0])
            g = poly_matmul(g, np.array(G), q2, poly_mod,N)

            tmp = poly_matmul(B, r1,q2,poly_mod, N)
            t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)
            p_size += t_g.nbytes 


            gammas = [random.randrange(0, q2) for i in range(k_hat)]
            v_size += np.array(gammas).nbytes


            y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**N, sigma=xi)
            y = [[y_sampler(),y_sampler()] for i in range(k)]
            # print(y)
            h = []
            w = []
            for i in range(k_hat):
                h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime), q2, poly_mod, N)
                w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
                h.append(h_i[0])
                w.append(w_i[0])

            w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))
            p_size += w_bold.nbytes + np.array(h).nbytes + np.array(w).nbytes

            # challenge = generate_random(1,1,N,1)
            # z = np.array(poly_add_3D(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), q1, poly_mod, N))
            
            challenge = generate_random(l*tau,l*tau,N,2)
            v_size += np.array(challenge).nbytes

            z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))
            p_size += z.nbytes
            prover_sizes.append(p_size)
            verifier_sizes.append(v_size)
            
            if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
                rejection_fail_count = rejection_fail_count + 1

            else:
                rejection_abort = False

                time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
                if status == True:
                    time_value, status, error = verifier_checks_2(k_hat, [disclose_index], True, h)
                if status == True:
                    time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g)
                rt_with_setup = time_value - setup_start
                rt = time_value - protocol_start
                data.append({"Index": index, "Prover size": p_size, "Verifier size": v_size, "Message size": np.array(message).nbytes,"Time":rt, "Time(with Setup)": rt_with_setup, "Rej Count": rejection_fail_count,"Attribute Value":val,"Attribute Index":disclose_index, "Success":status, "Error": error, "message": message, "m_prime":m_prime, "m_att": m_att})
                run_times.append(rt)
                run_times_setup.append(rt_with_setup)        

    avg = np.average(np.array(run_times))
    med = np.median(np.array(run_times))
    min = np.min(np.array(run_times))
    max = np.max(np.array(run_times))
    avg_setup = np.average(np.array(run_times_setup))
    med_setup = np.median(np.array(run_times_setup))
    min_setup = np.min(np.array(run_times_setup))
    max_setup = np.max(np.array(run_times_setup))
    p_avg = np.average(np.array(prover_sizes))
    p_med = np.median(np.array(prover_sizes))
    p_min = np.min(np.array(prover_sizes))
    p_max = np.max(np.array(prover_sizes))

    v_avg = np.average(np.array(verifier_sizes))
    v_med = np.median(np.array(verifier_sizes))
    v_min = np.min(np.array(verifier_sizes))
    v_max = np.max(np.array(verifier_sizes))

    m_avg = np.average(np.array(message_sizes))
    m_med = np.median(np.array(message_sizes))
    m_min = np.min(np.array(message_sizes))
    m_max = np.max(np.array(message_sizes))

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print("==========DISCLOSE FALSE ATTRIBUTE===============================")
    print("NO SETUP TIME: (avg, med, min, max)", avg, med, min, max)
    print("WITH SETUP TIME: (avg, med, min, max)", avg_setup, med_setup, min_setup, max_setup)
    print("PROVER SIZES avg, med, min, max: ",p_avg, p_med, p_min, p_max)
    print("VERIFIER SIZES avg, med, min, max: ", v_avg,v_med, v_min, v_max)
    print("MESSAGE SIZES avg, med, min, max: ", m_avg,m_med, m_min, m_max)
    return  run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes

def run_multi_range(num_attributes, subset_choices = [1,2,3,4,5], max_bit_index = 5, iterations=100):
    data = []
    run_times = []
    run_times_setup = []
    prover_sizes = []
    verifier_sizes = []
    message_sizes = []
    ##### RANGE m0 is between 0 and 7, where m0 is 6
    for index in range(iterations):  
        p_size = 0
        v_size = 0
   
        setup_start = time.perf_counter()
        r1 = generate_random(k,l*tau,N,2) 
        # print(l*tau)
        # how many bits to show range (if 3 bits is chosen then 2**3 = 8)
        random_small_subset = np.random.choice(subset_choices, num_attributes, replace=True)
        # print(random_small_subset)
        # pick a number that's between 0 and 2**random_small_subset
        random_nums = [np.random.choice(2**exp) for exp in random_small_subset]
        # convert to binary
        # print(random_nums[0], random_small_subset)
        bin_attributes = np.array([np.binary_repr(int(random_nums[i]), width=int(random_small_subset[i])) for i in range(num_attributes)])
        starting_disclose_index = []
        for i in range(num_attributes):
            # since bit is leftmost weight 0 0 *1* for example, then we take the starting index at *1* instead of conventionally
            # ggg = math.ceil(N/num_attributes)*(i+1)
            start_interval = math.floor(N/num_attributes)*i 
            end_interval = math.floor(N/num_attributes)*(i+1) - max_bit_index 
            # print('ggg',start_interval, end_interval
            starting_disclose_index.append(np.random.choice(range(start_interval, end_interval)))
            # starting_disclose_index.append(np.random.choice(range(max_bit_index, math.ceil(N/num_attributes)*i-max_bit_index, 1))

        # print(starting_disclose_index)
        # exit()
        # print(starting_disclose_index, starting_disclose_index+random_small_subset+1)
        disclose_indices = [list(range(starting_disclose_index[i]+ max_bit_index - random_small_subset[i], starting_disclose_index[i]+max_bit_index)) for i in range(num_attributes)] 
        NOT_disclose_indices = [list(range(starting_disclose_index[i], starting_disclose_index[i] + max_bit_index - random_small_subset[i])) for i in range(num_attributes)]
        NOT_disclose_indices = list(chain.from_iterable(NOT_disclose_indices))

        message = np.random.choice(2, (1,1,N))

        for i in range(num_attributes):

            bin_att = [int(d) for d in bin_attributes[i]]

            message[0][0][list(disclose_indices[i])] = bin_att        

        for i in range(len(NOT_disclose_indices)):
            message[0][0][int(NOT_disclose_indices[i])] = 0
        
        message_sizes.append(np.array(message).nbytes)

        vals = message[0][0][NOT_disclose_indices]
        not_vals = [1]*len(NOT_disclose_indices)
        diffs = np.subtract(np.array(vals),np.array(not_vals)) 
        m_prime_NOT = message.copy()
        m_att_NOT = [[[0]*N]]
        for i in range(len(NOT_disclose_indices)):
            m_prime_NOT[0][0][int(NOT_disclose_indices[i])] = diffs[i]
            m_att_NOT[0][0][int(NOT_disclose_indices[i])] = not_vals[i]

        m_prime_NOT = poly_matmul(np.array(m_prime_NOT), G, q2, poly_mod, N)
        m_att_NOT = poly_matmul(np.array(m_att_NOT),G, q2, poly_mod, N)
        
        B = gen_matrix_ring(k_hat, k, q2,N)
        A,D, q1_val, q2_val, N_deg, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=l, l_prime=l_prime,q1=q1,q2=q2,N=N)
        A1 = A[:n, :]
        A2 = A[n:,:]
        comm = cts_commit(np.array(A), np.array(message), np.array(r1),G, N, poly_mod, q1, q2, n, l)
        t11 = comm[:n]
        t21 = comm[n:]

        count = 0

        for att in range(num_attributes):
            random_indices = np.random.choice(2, len(disclose_indices[att]))
            for bit in range(len(disclose_indices[att])):
                val = message[0][0][disclose_indices[att][bit]]
                other_val = (val + 1) % 2 
                diff = val - other_val

                m_prime_1 = message.copy()
                m_prime_2 = message.copy()
                if random_indices[bit] == 1:
                    m_prime_1[0][0][disclose_indices[att][bit]] = diff
                    m_prime_2[0][0][disclose_indices[att][bit]] = 0
                else:
                    m_prime_1[0][0][disclose_indices[att][bit]] = 0
                    m_prime_2[0][0][disclose_indices[att][bit]] = diff

                m_att_1 = [[[0]*N]]
                m_att_2 = [[[0]*N]]

                if random_indices[bit] == 1:
                    m_att_1[0][0][disclose_indices[att][bit]] = other_val
                    m_att_2[0][0][disclose_indices[att][bit]] = val  
                else:
                    m_att_1[0][0][disclose_indices[att][bit]] = val
                    m_att_2[0][0][disclose_indices[att][bit]] = other_val

                m_att_1 = poly_matmul(m_att_1, G, q2, poly_mod, N)
                m_att_2 = poly_matmul(m_att_2,G, q2, poly_mod, N)
                m_prime_1 = poly_matmul(m_prime_1, G, q2, poly_mod, N)
                m_prime_2 = poly_matmul(m_prime_2,G, q2, poly_mod, N)
                m_att = [m_att_1, m_att_2]
                m_prime = [m_prime_1, m_prime_2]

                protocol_start = time.perf_counter()
                count = att
                prover_sizes, verifier_sizes,p_size,v_size, success_flag, rt, rt_with_setup, entry = or_property_info(p_size, v_size, prover_sizes,verifier_sizes,index, random_indices[bit], count, message, m_prime, m_att, val, other_val, A1, A2, q1, q2, N, k_hat, t11, t21, B, [disclose_indices[att][bit]], r1, bound, setup_start,protocol_start, num_attributes, G)            
                if success_flag == False:
                    break

    
            if success_flag == False:
                break

        num_NOT_attributes = len(NOT_disclose_indices)

        if success_flag == True and num_NOT_attributes > 0:

            rejection_fail_count = 0
            rejection_abort = True
            h_coeffs_zero = True
            h_coeffs_zero_count = 0
            while rejection_abort and h_coeffs_zero:  
                g = []
                for i in range(k_hat):
                    f = gen_matrix_ring(1,1,q2,N)
                    for j in range(num_NOT_attributes):
                        while f[0][0][NOT_disclose_indices[j]] == 0:
                            f[0][0][NOT_disclose_indices[j]] = (diffs[j])*random.randint(-(q2-1)//2, (q2-1)//2)

                    g.append(f[0])
                g = poly_matmul(g, np.array(G), q2, poly_mod,N)

                tmp = poly_matmul(B, r1,q2,poly_mod, N)
                t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)
                p_size += t_g.nbytes

                gammas = [random.randint(-(q2-1)//2, (q2-1)//2) for i in range(k_hat)]
                v_size += np.array(gammas).nbytes

                y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**Integer(N), sigma=xi)
                y = [[y_sampler(),y_sampler()] for i in range(k)]

                
                h = []
                w = []
                for i in range(k_hat):
                    h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime_NOT), q2, poly_mod, N)
                    w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
                    h.append(h_i[0])
                    w.append(w_i[0])
                
                w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))
                p_size += w_bold.nbytes + np.array(h).nbytes + np.array(w).nbytes

                challenge = generate_random(2,2,N,2)
                v_size += np.array(challenge).nbytes

                z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))
                p_size += z.nbytes

                prover_sizes.append(p_size)
                verifier_sizes.append(v_size)
                if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
                    rejection_fail_count += 1
                else:
                    rejection_abort = False

                    time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
                    if status == True:
                        time_value, status, error = verifier_checks_2(k_hat, NOT_disclose_indices, False, h)
                        if status == False:
                            h_coeffs_zero_count += 1
                            continue
                    if status == True:
                        h_coeffs_zero = False
                        time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att_NOT, h,w, t_g)
            
                    rt_with_setup = time_value - setup_start
                    rt = time_value - protocol_start
                    entry = {"Index": index, "Prover size": p_size, "Verifier size": v_size, "Message size": np.array(message).nbytes,"Time":rt,"Time (with Setup)":rt_with_setup , "Rej Count":rejection_fail_count , "Number Attributes": num_NOT_attributes,"Attribute Value":vals, "Not Attribute Value": not_vals, "Attribute Index":NOT_disclose_indices , "Success":status, "Error": error, "message": message, "m_prime_1":"","m_prime_2": m_prime_NOT, "m_att_1": "", "m_att_2": m_att_NOT}

        data.append(entry)
        run_times.append(rt)
        run_times_setup.append(rt_with_setup)    
            
 
    avg_not = np.average(np.array(run_times))
    med_not = np.median(np.array(run_times))
    min_not = np.min(np.array(run_times))
    max_not = np.max(np.array(run_times))
    avg_not_setup = np.average(np.array(run_times_setup))
    med_not_setup = np.median(np.array(run_times_setup))
    min_not_setup = np.min(np.array(run_times_setup))
    max_not_setup = np.max(np.array(run_times_setup))
    p_avg = np.average(np.array(prover_sizes))
    p_med = np.median(np.array(prover_sizes))
    p_min = np.min(np.array(prover_sizes))
    p_max = np.max(np.array(prover_sizes))

    v_avg = np.average(np.array(verifier_sizes))
    v_med = np.median(np.array(verifier_sizes))
    v_min = np.min(np.array(verifier_sizes))
    v_max = np.max(np.array(verifier_sizes))

    m_avg = np.average(np.array(message_sizes))
    m_med = np.median(np.array(message_sizes))
    m_min = np.min(np.array(message_sizes))
    m_max = np.max(np.array(message_sizes))

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print("==========RANGE RESULTS with multi attributes: ", num_attributes)
    print("NO SETUP TIME: (avg, med, min, max)", avg_not, med_not, min_not, max_not)
    print("WITH SETUP TIME: (avg, med, min, max)", avg_not_setup, med_not_setup, min_not_setup, max_not_setup)
    print("PROVER SIZES avg, med, min, max: ",p_avg, p_med, p_min, p_max)
    print("VERIFIER SIZES avg, med, min, max: ", v_avg,v_med, v_min, v_max)
    print("MESSAGE SIZES avg, med, min, max: ", m_avg,m_med, m_min, m_max)
    
    return  run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes

num_iterations = 6
###################################################################
# RUNNING MULTI NOT TESTS
###################################################################
# field_names_NOT = ['Index', "Prover size", "Verifier size", "Message size",'Time', 'Time (with Setup)', 'Rej Count' ,"H Coeffs Rerun", 'Number Attributes', 'Attribute Value', 'Not Attribute Value', 'Attribute Index','Success', 'Error', "message", "m_prime", "m_att"]
# for i in range(1,6):
#     run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes = run_multi_NOT(i,num_iterations)
#     csv_file = foldername + '//multi_NOT_' + str(i) +'_att.csv'
#     write_file(csv_file,field_names_NOT, data)

# ####################################################################
# # RUNNING NOT TESTS (with False values)
# # ####################################################################
# run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes = run_NOT_false(num_iterations)
# csv_file = foldername +'//false_NOT.csv'
# field_names_NOT = ['Index', "Prover size", "Verifier size", "Message size",'Time', 'Time (with Setup)','Rej Count' , 'Attribute Value', 'Not Attribute Value', 'Attribute Index','Success', 'Error', "message", "m_prime", "m_att"]
# # write_file(csv_file,field_names_NOT, data)

# ##################################################################
# # RUNNING MULTI OR TESTS
# ##################################################################
# field_names_OR = ['Index', "Prover size", "Verifier size", "Message size",'Time','Time (with Setup)' ,'Rej Count' ,'Number Attributes','Attribute Value', 'Not Attribute Value', 'Attribute Index','Success', 'Error', "message", "m_prime_1", "m_prime_2", "m_att_1", "m_att_2"]

# for i in range(1,6):
#     run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes = run_multi_OR_random(i,num_iterations)
#     csv_file = foldername + '//multi_OR_' + str(i) +'_att.csv'
#     write_file(csv_file,field_names_OR, data)


# ####################################################################
# # RUNNING OR TESTS (with False values)
# ####################################################################
# csv_file = foldername +'//False_OR_SIZE.csv'
# run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes = run_OR_false(num_iterations)
# field_names_OR = ["Index","Prover size", "Verifier size", "Message size","Time","Time (with Setup)","Actual Value", 'Rej Count' ,'Number Attributes',"Attribute Value", "Not Attribute Value", "Attribute Index", "Success", "Error", "message", "m_prime_1", "m_prime_2", "m_att_1", "m_att_2"]
# write_file(csv_file,field_names_OR, data)


# ###################################################################
# # RUNNING DISCLOSE TESTS
# ###################################################################
# field_names_disclose = ['Index', "Prover size", "Verifier size", "Message size",'Time','Time(with Setup)', 'Rej Count' ,'Number Attributes','Attribute Value','Attribute Index','Success', 'Error', "message", "m_prime", "m_att"]

# for i in range(1,6):
#     run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes = run_multi_disclose(i,num_iterations)
#     csv_file = foldername + '//multi_DISCLOSE__SIZE_' + str(i) +'_att.csv'
#     write_file(csv_file,field_names_disclose, data) 

# # ####################################################################
# # # RUNNING DISCLOSE TESTS (with False values)
# # ####################################################################
# run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes = disclose_false(num_iterations)
# csv_file =  foldername +'//False_DISCLOSE_SIZE.csv'
# field_names_disclose = ["Index", "Prover size", "Verifier size", "Message size","Time", "Time(with Setup)", 'Rej Count' , "Attribute Value", "Attribute Index", "Success", "Error", "message", "m_prime", "m_att"]
# write_file(csv_file,field_names_disclose, data)

####################################################################
# RUNNING MULTI RANGE TESTS
####################################################################
field_names_range = ['Index', "Prover size", "Verifier size", "Message size",'Time', 'Time (with Setup)','Rej Count' , 'Number Attributes', 'Attribute Value', 'Not Attribute Value', 'Attribute Index','Success', 'Error', "message", "m_prime_1", "m_prime_2", "m_att_1", "m_att_2"]

for i in range(1,6):
    run_times_setup, run_times, data, prover_sizes, verifier_sizes,  message_sizes = run_multi_range(i,iterations=num_iterations)
    csv_file = foldername + '//NEW_multi_RANGE_SIZE_' + str(i) +'_att.csv'
    write_file(csv_file,field_names_range, data)