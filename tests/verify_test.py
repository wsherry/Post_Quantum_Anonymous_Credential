sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))
from src.ac import *
from latticezk_signature import *
import time
from tests.helpers import *



def test_abdlop_mlwe__(F_comm, u, cred,start, bound):
    """Call abdlop_mlwe with example parameters"""
    # Size parameters
    deg = 512
    alpha = math.ceil(2*math.sqrt(q2 +1)*((2*math.sqrt(2))*math.sqrt(deg) + 1))
    eta = 59
    kappa = 2 
    # Size parameters
    m1 = 10     # size of s1
    m2 = 25    # size of s2
    n = 6      # dimension of the Module-SIS problem
    Z = 1      # number of equations for Euclidean norm
    lambd = 5  # number of random masking polynomials (boosting soundness to q1^-lambda)
    n_A = 1    # height of A matrix
    m_A = 10   # width of A matrix (must be same size as m1)
    # Algorithm parameters
    is_opti = False  # proof size optimized sigma_n1 version with reduced garbage commitments
    if is_opti:
        lambd = lambd//2  # reduced garbage commitments

    k = k_sigma_n1                      # degree of automorphism
    eta = eta_challenge_v2              # challenge norm upper bound
    get_challenge_u = get_challenge_v2  # challenge sampling algorithm

    rej_u_1 = rej1  # rejection sampling algorithm
    rej_u_2 = rej2
    rej_u_3 = rej_bimodal

    gamma1 = 19  # repetition rate parameter
    nu_s1 = 67108837
    # s1 max coefficient
    alpha_s1 = norm_Rql_bound(m1, nu_s1)  # s1 norm upper bound
    std1 = gamma1 *  alpha_s1 # standard deviation

    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 1
    nu_s2 = 59
    alpha_s2 = norm_Rql_bound(m2, nu_s2)
    std2 = gamma2 * alpha_s2 *eta
    rep_M2 = rej2_M(gamma2)

    # Private information
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector
    temp_s = []
    for i in range(cred.shape[0]):
        temp_s.append(ZZl_2_Rql(cred[i][0]))
    s1 = sg.vector(R_q,stack_vec_Rql(temp_s)) #.transpose()
    # Public information
    # ABDLOP commitment
    A1 = rand_Rq_mat(n, m1 + Z)
    A2 = rand_Rq_mat(n, m2)
    B_gamma = rand_Rq_mat(1, m2)
    B_beta = rand_Rq_mat(1, m2)
    B_ext = rand_Rq_mat(lambd, m2)
    b_ext = rand_Rql(m2)

    # MLWE

    temp = []

    for i in range(F_comm.shape[1]):
        temp.append(ZZl_2_Rql(F_comm[0][i]))
    A = sg.vector(R_q,stack_vec_Rql(temp))
    temp = []
    for i in range(u.shape[1]):
        temp.append(ZZl_2_Rql(u[0][i]))
    u = sg.vector(R_q,stack_vec_Rql(temp))

    # Shortness in the Euclidean norm
    n_is = [m_A+1]  # size of vectors for Euclidean norm equations
    E_s_is = [IM(m_A).stack(A)]
    v_is = [-stack_vec_Rql([zv(m_A), u])]
    norms = [norm_Rql(E_s_is[0]*s1 + v_is[0])]
    B_is = [int(bound)]  # integer Euclidean norm bound (sqrt(2048) is upper bound of norm for nu_s1 = 1)

    gamma3 = 6
    std3 = gamma3 * B_is[0]
    rep_M3 = rej_bimodal_M(gamma3)

    global rep_rate_abdlop_mlwe
    rep_rate_abdlop_mlwe = rep_M1 * 2*rep_M2 * rep_M3

    roh = 1.64  # z3 norm bound (for kappa=128)

    # ABDLOP tA
    squared_norm_diffs = [B_is[0]**2 - round(norms[0]**2)]  # squared norm is always an int (round float errors)
    theta = sg.vector(R_q, [N_2_binary_Rq(squared_norm_diffs[0])])
    tA = A1 * stack_vec_Rql([s1, theta]) + A2 * s2  # ABDLOP tA

    print(time.perf_counter()- start)
    result = abdlop_mlwe(m1, m2, n, k, Z, n_is, lambd,
                       is_opti, rej_u_1, rej_u_2, rej_u_3, get_challenge_u,
                       std1, rep_M1, std2, rep_M2, std3, rep_M3, roh,
                       s1, s2, A1, A2, B_gamma, B_beta, B_ext, b_ext, theta, tA,
                       E_s_is, v_is, B_is)

    iterations = 1
    while result == "Rejected":
        iterations = iterations + 1 
        result = abdlop_mlwe(m1, m2, n, k, Z, n_is, lambd,
                        is_opti, rej_u_1, rej_u_2, rej_u_3, get_challenge_u,
                        std1, rep_M1, std2, rep_M2, std3, rep_M3, roh,
                        s1, s2, A1, A2, B_gamma, B_beta, B_ext, b_ext, theta, tA,
                        E_s_is, v_is, B_is)
    
    if result == False:
        return False, iterations
    else: 
        return True, iterations


def verify_test(num_iterations = 100):
    data = []
    run_times = []
    d = 512
    N = 512
    k = 4
    q1 = 4294966997
    q2 = 67108837
    poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
    n = 1  
    l = 1
    l_hat = 2
    tau = 2
    norm_m = 22

    for index in range(num_iterations):  
        print("___NEW ITERATION___", index)

        start = time.perf_counter()   
        alph = math.ceil(2*math.sqrt(q2 +1)*((2*math.sqrt(2))*math.sqrt(N) + 1))
        
        G = np.zeros((1,2,N))
        G[0][0][-1] = 1
        G[0][1][-1] = np.rint(np.sqrt(q2))
        A, D, q1, q2, N, kappa, gamma, gamma_prime, alpha, opk_a0, opk_B, opk_u, osk,usk = AC_setup()
        A2 = A[:1, 1:]
        comm,rand = AC_registration(n=4,m=2,N=N,A=A,poly_mod=poly_mod,q1=q1,q2=q2,n1=1,l=l,G=G,tau=tau, usk=usk, beta=2)        
        miu = ZZl_2_Rql(rand.flatten())
        message = ZZl_2_Rql(usk[0][0])    
        c2 = comm[1:]
        u = []
        
        bound = (math.sqrt(3) + math.sqrt(2))*N*alph*2 + alph*math.sqrt(N*10)
        cred, u = AC_issue(D,A2, opk_a0,opk_B,u,comm,N,poly_mod,q2,bound)
        fcomm = generate_f_comm(opk_B,c2,D,opk_a0,A2,q2, poly_mod,N)
        result, num_runs = test_abdlop_mlwe__(fcomm, u, cred,start,bound)

        if result == True:
            rt = time.perf_counter() - start
            data.append({"Index": index, "Time":rt, "Success":True, "Iterations":num_runs, "Error": 0, "message": message, "Credential":cred})
            run_times.append(rt)
        else:
            rt = time.perf_counter() - start
            run_times.append(rt)
            data.append({"Index": index, "Time":rt, "Success":False, "Iterations":num_runs, "Error": 1, "message": message, "Credential":[]})
    
    avg = np.average(np.array(run_times))
    med = np.median(np.array(run_times))
    min = np.min(np.array(run_times))
    max = np.max(np.array(run_times))

    print("--------------------------------------------------------------------------")
    print("VERIFY TEST")
    print("TIME: (avg, med, min, max)", avg, med, min, max)
    return data, avg, med, min, max

# data, avg, med, min, max = verify_test()
# csv_file =  'VERIFY_test_5.csv'
# field_names_disclose = ["Index", "Time", "Success", "Iterations","Error", "message", "Credential"]
# write_file(csv_file,field_names_disclose, data)


def verify_transferred_credential_test(num_iterations = 100):
    data = []
    run_times = []
    N = 512
    k = 4
    q1 = 4294966997
    q2 = 67108837
    poly_mod = np.poly1d([1] + [0]*(N-1) + [1])
    n = 1  
    l = 1
    l_hat = 2
    tau = 2
    norm_m = 22
    eta = 59
    alph = math.ceil(2*math.sqrt(q2 +1)*((2*math.sqrt(2))*math.sqrt(N) + 1))
    n1 = 1

    G = np.zeros((1,2,N))
    G[0][0][-1] = 1
    G[0][1][-1] = np.rint(np.sqrt(q2))
    A, D, q1, q2, N, kappa, gamma, gamma_prime, alpha, opk_a0, opk_B, opk_u, osk,usk = AC_setup()
    A2 = A[:1, 1:]
    comm, rand = AC_registration(n=4,m=2,N=N,A=A,poly_mod=poly_mod,q1=q1,q2=q2,n1=1,l=l,G=G,tau=tau, usk=usk, beta=2)        
    message = ZZl_2_Rql(usk[0][0])    
    c2 = comm[1:]
    u = []
    bound = (math.sqrt(3) + math.sqrt(2))*N*alph*2 + alph*math.sqrt(N*10)
    fcomm, cred, u = AC_issue(D,A2, opk_a0,opk_B,u,comm,N,poly_mod,q2,bound)
    alph = math.ceil(2*math.sqrt(q2 +1)*((2*math.sqrt(2))*math.sqrt(N) + 1))

    for index in range(num_iterations):  
        print("___NEW ITERATION___", index)

        start = time.perf_counter()   
        transferred_f_comm, transferred_cred, transferred_comm, pi_prime,transferred_u = AC_prove(D,A,opk_a0, opk_B,n,k,l,l_hat,tau,usk,rand,N,poly_mod,q1,q2,n1,m,comm,cred,u,gamma_prime,G,beta=2)

        bound_transferred = (2*math.sqrt(N*20)*eta*kappa)*((math.sqrt(3)+math.sqrt(2))*N*alph*2 + alph*(math.sqrt(20*N))) 
        result, num_runs = test_abdlop_mlwe__(transferred_f_comm, transferred_u, transferred_cred,start, bound_transferred)

        if result == True:
            rt = time.perf_counter() - start
            data.append({"Index": index, "Time":rt, "Success":True, "Iterations":num_runs, "Error": 0, "message": message, "Credential":cred})
            run_times.append(rt)
        else:
            rt = time.perf_counter() - start
            run_times.append(rt)
            data.append({"Index": index, "Time":rt, "Success":False, "Iterations":num_runs, "Error": 1, "message": message, "Credential":[]})
    
    avg = np.average(np.array(run_times))
    med = np.median(np.array(run_times))
    min = np.min(np.array(run_times))
    max = np.max(np.array(run_times))

    print("--------------------------------------------------------------------------")
    print("VERIFY TRANSFER TEST")
    print("TIME: (avg, med, min, max)", avg, med, min, max)
    return data, avg, med, min, max

data, avg, med, min, max = verify_transferred_credential_test()
csv_file =  'VERIFY_test_transferred_100.csv'
field_names_disclose = ["Index", "Time", "Success", "Iterations","Error", "message", "Credential"]
write_file(csv_file,field_names_disclose, data)
# res, cred = verify_test()
# print(res, cred)
    
# csv_file = 'ISSUE_results.csv'

# with open(csv_file, mode='a', newline='') as file:
#     # Define the field names for the CSV header
#     fieldnames = ['Index', 'Time','Function', 'Status']
#     writer = csv.DictWriter(file, fieldnames=fieldnames)

#     # Write the header
#     writer.writeheader()
    
#     # Write the data rows
#     writer.writerows(data)

# print(f"Data has been written to {csv_file}")
#print("Run Toolbox Proof...")
#print(repeat_until_accept(test_abdlop_toolbox))

#print(measure_rep_rate(test_abdlop_toolbox, rep_rate_abdlop_toolbox, runs=10))

