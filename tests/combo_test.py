
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ac import *
import csv
import time
from sage.all import *
import random
from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionLatticeSampler
from latticezk import rej1
from extra_functions import *
from itertools import chain

foldername = "COMBO RESULTS"
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
bound = xi*T*2
poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
n = 1  
l = 1
l_prime = 2
# G = 1
G = np.zeros((1,2,N))
G[0][0][-1] = 1
G[0][1][-1] = np.rint(np.sqrt(q2))

def get_random_vals_NOT_OR(num_attributes, N, m=100):
    random_vals = [np.random.choice(m, 2, replace=False) for att in range(num_attributes)]
    vals = [vals[0] for vals in random_vals]
    not_vals =  [vals[1] for vals in random_vals]
    diffs = []
    for i in range(num_attributes):
        diffs.append(vals[i] - not_vals[i])
    disclose_indices =  np.random.choice(N, num_attributes, replace=False)
    return vals, not_vals, diffs, disclose_indices

def run_combo(iterations = 100):
    data = []
    run_times = []
    run_times_setup = []
    
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
    bound = xi*T*2
    poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
    n = 1  
    l = 1
    l_prime = 2
    # G = 1
    G = np.zeros((1,2,N))
    G[0][0][-1] = 1
    G[0][1][-1] = np.rint(np.sqrt(q2))
    num_attributes = 1
    ##### RANGE m0 is between 0 and 7, where m0 is 6
    for index in range(iterations):  

        setup_start = time.perf_counter()
        
        message = np.random.choice(2, (1,1,N))
        
        # RANGE SETUP VALUES
        max_bit_index = 3
        random_small_subset = np.random.choice([1,2,3], num_attributes, replace=True)
        random_nums = [np.random.choice(2**exp) for exp in random_small_subset]
        bin_attributes = np.array([np.binary_repr(int(random_nums[i]), width=int(random_small_subset[i])) for i in range(num_attributes)])
        starting_disclose_index = [np.random.choice(range(N//2 - max_bit_index))]
        # print(starting_disclose_index)
        RANGE_disclose_indices = [list(range(starting_disclose_index[0]+ max_bit_index - random_small_subset[0], starting_disclose_index[0]+max_bit_index)) for i in range(num_attributes)] 
        RANGE_NOT_disclose_indices = [list(range(starting_disclose_index[0], starting_disclose_index[0] + max_bit_index - random_small_subset[0])) for i in range(num_attributes)]
        RANGE_NOT_disclose_indices = list(chain.from_iterable(RANGE_NOT_disclose_indices))

        for i in range(num_attributes):
            bin_att = [int(d) for d in bin_attributes[i]]
            message[0][0][list(RANGE_disclose_indices[i])] = bin_att        

        for i in range(len(RANGE_NOT_disclose_indices)):
            message[0][0][int(RANGE_NOT_disclose_indices[i])] = 0
        
        RANGE_vals = message[0][0][RANGE_NOT_disclose_indices]
        RANGE_not_vals = [1]*len(RANGE_NOT_disclose_indices)
        RANGE_diffs = np.subtract(np.array(RANGE_vals),np.array(RANGE_not_vals)) 
        # print("vals/not_vals/diffs", vals, not_vals, diffs)
        m_prime_NOT = message.copy()
        m_att_NOT = [[[0]*N]]
        for i in range(len(RANGE_NOT_disclose_indices)):
            m_prime_NOT[0][0][int(RANGE_NOT_disclose_indices[i])] = RANGE_diffs[i]
            m_att_NOT[0][0][int(RANGE_NOT_disclose_indices[i])] = RANGE_not_vals[i]

        m_prime_NOT = poly_matmul(np.array(m_prime_NOT), G, q2, poly_mod, N)
        m_att_NOT = poly_matmul(np.array(m_att_NOT),G, q2, poly_mod, N)
        
        # OR SETUP
        OTHER_disclose_indices =  np.random.choice(range(N//2, N), 3, replace=False)
        
        OR_random_vals = [np.random.choice(100, 2, replace=False) for att in range(num_attributes)]
        OR_vals = [vals[0] for vals in OR_random_vals]
        OR_other_vals =  [vals[1] for vals in OR_random_vals]
        OR_diff = OR_vals[0] - OR_other_vals[0]
        OR_random_indices = np.random.choice(2, num_attributes)

        message[0][0][int(OTHER_disclose_indices[0])] = OR_vals[0]
        OR_m_prime_1 = message.copy()
        OR_m_prime_2 = message.copy()
        OR_m_prime_1[0][0][int(OTHER_disclose_indices[0])] = OR_diff
        OR_m_prime_2[0][0][int(OTHER_disclose_indices[0])] = 0

        OR_m_att_1 = [[[0]*N]]
        OR_m_att_2 = [[[0]*N]]

        OR_m_att_1[0][0][int(OTHER_disclose_indices[0])] = OR_other_vals[0]
        OR_m_att_2[0][0][int(OTHER_disclose_indices[0])] = OR_vals[0]  
        OR_m_att_1 = poly_matmul(OR_m_att_1, G, q2, poly_mod, N)
        OR_m_att_2 = poly_matmul(OR_m_att_2,G, q2, poly_mod, N)
        OR_m_prime_1 = poly_matmul(OR_m_prime_1, G, q2, poly_mod, N)
        OR_m_prime_2 = poly_matmul(OR_m_prime_2,G, q2, poly_mod, N)
        OR_m_att = [OR_m_att_1, OR_m_att_2]
        OR_m_prime = [OR_m_prime_1, OR_m_prime_2]
        # NOT SETUP
        _NOT_random_vals = [np.random.choice(100, 2, replace=False) for att in range(num_attributes)]
        _NOT_vals = [vals[0] for vals in _NOT_random_vals]
        _NOT_not_vals =  [vals[1] for vals in _NOT_random_vals]
        _NOT_diffs = np.subtract(np.array(_NOT_vals), np.array(_NOT_not_vals))
        message[0][0][int(OTHER_disclose_indices[1])] = _NOT_not_vals[0]
        _NOT_m_prime = message.copy()
        _NOT_m_att = [[[0]*N]]
        _NOT_m_prime[0][0][int(OTHER_disclose_indices[1])] = _NOT_diffs[0]
        _NOT_m_att[0][0][int(OTHER_disclose_indices[1])] = _NOT_not_vals[0]

        # DISCLOSE SETUP
        DISC_vals = np.random.choice(100, 1, replace=False)
        message[0][0][int(OTHER_disclose_indices[2])] = DISC_vals[0]
        DISC_m_prime = message.copy()
        DISC_m_prime[0][0][int(OTHER_disclose_indices[2])] = 0
        DISC_m_att = [[[0]*N]]
        DISC_m_att[0][0][int(OTHER_disclose_indices[2])] = DISC_vals[0]
        
        # GENERAL SETUP
        r1 = generate_random(k,l*tau,N,2) 
        A,D, q1_val, q2_val, N_val, k_prime, gamma, gamma_prime, alpha = cts_setup(n=n, k=k ,l=l, l_prime=l_prime,q1=q1,q2=q2,N=N)
        B = gen_matrix_ring(k_hat, k, q2,N)
        A1 = A[:n, :]
        A2 = A[n:,:]
        comm = cts_commit(np.array(A), np.array(message), np.array(r1),G, N, poly_mod, q1, q2, n, l, tau = tau)
        t11 = comm[:n]
        t21 = comm[n:]
        count = 0

        protocol_start = time.perf_counter()
        ##### RANGE PROPERTY ON ATTRIBUTE 1
        for att in range(num_attributes):
            random_indices = np.random.choice(2, len(RANGE_disclose_indices[att]))
            for bit in range(len(RANGE_disclose_indices[att])):
                val = message[0][0][RANGE_disclose_indices[att][bit]]
                other_val = (val + 1) % 2 
                diff = val - other_val

                m_prime_1 = message.copy()
                m_prime_2 = message.copy()
                if random_indices[bit] == 1:
                    m_prime_1[0][0][RANGE_disclose_indices[att][bit]] = diff
                    m_prime_2[0][0][RANGE_disclose_indices[att][bit]] = 0
                else:
                    m_prime_1[0][0][RANGE_disclose_indices[att][bit]] = 0
                    m_prime_2[0][0][RANGE_disclose_indices[att][bit]] = diff

                m_att_1 = [[[0]*N]]
                m_att_2 = [[[0]*N]]

                if random_indices[bit] == 1:
                    m_att_1[0][0][RANGE_disclose_indices[att][bit]] = other_val
                    m_att_2[0][0][RANGE_disclose_indices[att][bit]] = val  
                else:
                    m_att_1[0][0][RANGE_disclose_indices[att][bit]] = val
                    m_att_2[0][0][RANGE_disclose_indices[att][bit]] = other_val

                m_att_1 = poly_matmul(m_att_1, G, q2, poly_mod, N)
                m_att_2 = poly_matmul(m_att_2,G, q2, poly_mod, N)
                m_prime_1 = poly_matmul(m_prime_1, G, q2, poly_mod, N)
                m_prime_2 = poly_matmul(m_prime_2,G, q2, poly_mod, N)
                RANGE_m_att = [m_att_1, m_att_2]
                RANGE_m_prime = [m_prime_1, m_prime_2]

                protocol_start = time.perf_counter()
                count = att
                
                success_flag, rt, rt_with_setup, entry = or_property(index, random_indices[bit], count, message, RANGE_m_prime, RANGE_m_att, val, other_val, A1, A2, q1, q2, N, k_hat, t11, t21, B, [RANGE_disclose_indices[att][bit]], r1, bound, setup_start,protocol_start, num_attributes, G)
                if success_flag == False:
                    break
            if success_flag == False:
                break

        num_NOT_attributes = len(RANGE_NOT_disclose_indices)
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
                        while f[0][0][RANGE_NOT_disclose_indices[j]] == 0:
                            f[0][0][RANGE_NOT_disclose_indices[j]] = (RANGE_diffs[j])*random.randint(-(q2-1)//2, (q2-1)//2)
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
                    h_i = poly_add_3D([g[i]], np.multiply(gammas[i],m_prime_NOT), q2, poly_mod, N)
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
                        time_value, status, error = verifier_checks_2(k_hat, RANGE_NOT_disclose_indices, False, h)
                        if status == False:
                            h_coeffs_zero_count += 1
                            continue
                    if status == True:
                        h_coeffs_zero = False
                        time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att_NOT, h,w, t_g)
            
                    rt_with_setup = time_value - setup_start
                    rt = time_value - protocol_start
                    # entry = {"Index": index, "Time":rt, "Time (with Setup)":rt_with_setup ,"Rej Count": rejection_fail_count, "Number Attributes": num_attributes,"Attribute Value":vals, "Not Attribute Value": not_vals, "Attribute Index":disclose_indices , "Success":status, "Error": error, "message": message, "m_prime":m_prime, "m_att": m_att}
                    entry = {"Index": index, "Time":rt,"Time (with Setup)":rt_with_setup , "Rej Count":rejection_fail_count , "Number Attributes": num_NOT_attributes,"Attribute Value":RANGE_vals, "Not Attribute Value": RANGE_not_vals, "Attribute Index":RANGE_NOT_disclose_indices , "Success":status, "Error": error, "message": message, "m_prime_1":"","m_prime_2": m_prime_NOT, "m_att_1": "", "m_att_2": m_att_NOT}

        if success_flag == False:
            data.append(entry)
            run_times.append(rt) 
            run_times_setup.append(rt_with_setup)   
            continue
        ##### OR PROPERTY ON ATTRIBUTE 2
        print("RANGE______", success_flag)

        count = 0
        success_flag, rt,rt_with_setup, entry = or_property(index, OR_random_indices[0], count, message, OR_m_prime, OR_m_att, OR_vals[0], OR_other_vals[0], A1, A2, q1, q2, N, k_hat, t11, t21, B, [OTHER_disclose_indices[0]], r1, bound, setup_start,protocol_start, num_attributes,G)
        if success_flag == False:
            data.append(entry)
            run_times.append(rt) 
            run_times_setup.append(rt_with_setup)   
            continue
        
        # NOT PROPERTY ON ATTRIBUTE 3
        print("OR______", success_flag)

        rejection_fail_count = 0
        rejection_abort = True
        h_coeffs_zero = True
        h_coeffs_zero_count = 0

        while rejection_abort and h_coeffs_zero:  
            g = []
            for i in range(k_hat):
                f = gen_matrix_ring(1,1,q2,N)
                while f[0][0][OTHER_disclose_indices[1]] == 0:
                    f[0][0][OTHER_disclose_indices[1]] = (_NOT_diffs[j])*random.randint(-(q2-1)//2, (q2-1)//2)
                print(f[0][0][OTHER_disclose_indices[1]])
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
                h_i = poly_add_3D([g[i]], np.multiply(gammas[i],_NOT_m_prime), q2, poly_mod, N)
                w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
                h.append(h_i[0])
                w.append(w_i[0])

            w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))

            challenge = generate_random(l*tau,l*tau,N,2)
            z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))

            if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
                rejection_fail_count += 1
            else:
                rejection_abort = False

                time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
                if status == True:
                    time_value, status, error = verifier_checks_2(k_hat, OTHER_disclose_indices[1], False, h)
                    if status == False:
                        h_coeffs_zero_count += 1
                        continue
                if status == True:
                    h_coeffs_zero = False
                    time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, _NOT_m_att, h,w, t_g)
        
                rt_with_setup = time_value - setup_start
                rt = time_value - protocol_start
                data.append({"Index": index, "Time":rt, "Time (with Setup)":rt_with_setup ,"Rej Count": rejection_fail_count, "H Coeffs Rerun": h_coeffs_zero_count, "Number Attributes": num_attributes,"Attribute Value":_NOT_vals, "Not Attribute Value": _NOT_not_vals, "Attribute Index":OTHER_disclose_indices[1] , "Success":status, "Error": error, "message": message, "m_prime":_NOT_m_prime, "m_att": _NOT_m_att})
                run_times.append(rt)
                run_times_setup.append(rt_with_setup)            

        if success_flag == False:
            data.append(entry)
            run_times.append(rt) 
            run_times_setup.append(rt_with_setup)   
            continue
        ####### DISCLOSE
        print("NOT______", success_flag)

        rejection_fail_count = 0
        rejection_abort = True

        while rejection_abort:  
            g= []
            for i in range(k_hat):
                f = gen_matrix_ring(1,1,q2,N)
                f[0][0][OTHER_disclose_indices[2]] = 0
                g.append(f[0])
            g = poly_matmul(g, np.array(G), q2, poly_mod,N)

            tmp = poly_matmul(B, r1,q2,poly_mod, N)
            t_g = poly_add_3D(tmp,np.array(g), q2, poly_mod, N)

            gammas = [random.randrange(0, q2) for i in range(k_hat)]

            y_sampler = DiscreteGaussianDistributionLatticeSampler(ZZ**N, sigma=xi)
            y = [[y_sampler(),y_sampler()] for i in range(k)]

            h = []
            w = []
            for i in range(k_hat):
                h_i = poly_add_3D([g[i]], np.multiply(gammas[i],DISC_m_prime), q2, poly_mod, N)
                w_i = poly_matmul(poly_add_3D(np.multiply(gammas[i],A2),B[[i],:], q2, poly_mod, N), y, q2, poly_mod, N)
                h.append(h_i[0])
                w.append(w_i[0])
            w_bold = np.array(poly_matmul(A1, y, q1, poly_mod, N))

            challenge = generate_random(l*tau,l*tau,N,2)
            z = np.array(poly_add_3D_no_mod(np.array(y),ring_transpose(poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N)), poly_mod, N))

            if rejection_sampling_1(z, poly_matmul_no_mod(np.array(challenge),ring_transpose(np.array(r1)), poly_mod, N), xi) == 0:
                rejection_fail_count = rejection_fail_count + 1
                # rt_with_setup = time.perf_counter() - setup_start
                # rt = time.perf_counter() - protocol_start
                # data.append({"Index": index, "Time":rt,  "Time(with Setup)": rt_with_setup, "Number Attributes": num_attributes, "Attribute Value":vals, "Attribute Index":disclose_indices, "Success":False, "Error": -1, "message": message, "m_prime":m_prime, "m_att": m_att})
                # run_times.append(rt)
                # run_times_setup.append(rt_with_setup)
                # continue
            else: 
                rejection_abort = False
                time_value, status, error = verifier_checks_1(z, bound, A1, q1, poly_mod, N, w_bold, challenge, t11)
                if status == True:
                    time_value, status, error = verifier_checks_2(k_hat, OTHER_disclose_indices[2], True, h)
                if status == True:
                    # time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, m_att, h,w, t_g)
                    time_value, status, error = verifier_checks_3(k_hat, gammas, A2, B, q2, poly_mod, N, z, challenge, t21, DISC_m_att, h,w, t_g)

        if success_flag == False:
            data.append(entry)
            run_times.append(rt) 
            run_times_setup.append(rt_with_setup)   
            continue
        print("DISC______", success_flag)
        
        rt_with_setup = time_value - setup_start
        rt = time_value - protocol_start
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
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("==========COMBO ALL RESULTS ")
    print("NO SETUP TIME: (avg, med, min, max)", avg_not, med_not, min_not, max_not)
    print("WITH SETUP TIME: (avg, med, min, max)", avg_not_setup, med_not_setup, min_not_setup, max_not_setup)
    return data, avg_not, med_not, min_not, max_not, avg_not_setup, med_not_setup, min_not_setup, max_not_setup

####################################################################
# RUNNING COMBO TESTS
####################################################################
iterations = 100
data, avg_not, med_not, min_not, max_not, avg_not_setup, med_not_setup, min_not_setup, max_not_setup = run_combo(iterations)

# # run_NOT()
csv_file_1 =   foldername +'//COMBO.csv'

field_names_NOT = ['Index', 'Time', 'Time (with Setup)','Rej Count' ,"Number Attributes",'Attribute Value', 'Not Attribute Value', 'Attribute Index','Success', 'Error', "message", "m_prime_1","m_prime_2", "m_att_1", "m_att_2"]

write_file(csv_file_1,field_names_NOT, data)

