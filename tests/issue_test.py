
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))
from src.ac import *
from latticezk_commitment import *
import time
from helpers import write_file

def R_to_S_vec(S,vec):
    new_vec = sg.vector(S, len(vec))
    for i in range(len(vec)):
        coeffs = Rql_2_ZZl(vec)
        poly_q2 = S(sum(c * t**k for k, c in enumerate(coeffs)))
        new_vec[i] = poly_q2
    return new_vec  

def R_to_S_mat(S,A):
    new_matrix = sg.matrix(S, A.nrows(), A.ncols())
    for i in range(A.nrows()):
        for j in range(A.ncols()):
            poly_q1 = A[i, j].lift()
            coeffs = poly_q1.coefficients(sparse=False)
            poly_q2 = S(sum(c * t**k for k, c in enumerate(coeffs)))
            new_matrix[i, j] = poly_q2
    return new_matrix         

def run_nizk_proof(miu):
    """Call abdlop_toolbox with example parameters"""
    print("-------------------------------")

    # Size parameters

    m1 = 74 #36    # size of s1 
    m2 = 32 #31    # size of s2
    
    ell = 0    # size of m
    n = 6      # dimension of the Module-SIS problem
    N = 16      # number of quadratic equations
    M = 1      # number of quadratic polynomials with constant coefficient zero
    n_bin = 4  # size of infinity norm vector
    Z = 3      # number of equations for Euclidean norm
    n_is = [38,16,16]  # size of vectors for Euclidean norm equations
    n_p = 37    # size of approximate shortness vector
    lambd = 6  # number of random masking polynomials (boosting soundness to q1^-lambda)
    lambd = 5
    m_pke = 38
    n_pke = 1
    k_pke = 36
    q_pke = 58373
    q = 4503586638659881
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
    rej_u_4 = rej_bimodal

    gamma1 = 17  # repetition rate parameter
    norm_m = 22
    alpha = sg.sqrt(4*d*m_pke + 4*4*d) + norm_m
    std1 = gamma1 * eta *alpha
    rep_M1 = rej1_M(gamma1)  # repetition rate

    gamma2 = 1.2
    nu_s2 = 1
    std2 = gamma2 * eta * nu_s2 * sg.sqrt(m2*d)
    rep_M2 = rej2_M(gamma2)

    # Private information
    s2 = rand_Rql_small(m2, nu_s2)  # ABDLOP randomness vector


    A_pke = rand_Rq_mat_pke(n_pke, m_pke)
    S_pke = rand_Rq_mat_pke(k_pke, n_pke)
    E_pke = rand_Rq_mat_pke_small(k_pke,m_pke, 2)
    B_pke = S_pke*A_pke + 3*E_pke

    # enc_encrypt in sagemath coded below:
    r_pke = rand_Rql_small(m_pke, 2)
    t0 = A_pke * R_to_S_vec(R_q_pke, r_pke)
    t1 = B_pke * R_to_S_vec(R_q_pke, r_pke) + R_to_S_vec(R_q_pke, miu)

    s1 = stack_vec_Rql([r_pke, miu])

    # Public information
    # ABDLOP commitment
    A1 = rand_Rq_mat(n, m1 + Z)
    A2 = rand_Rq_mat(n, m2)
    B_gamma = rand_Rq_mat(512//d, m2)
    B_beta = rand_Rq_mat(2, m2)
    B_ext = rand_Rq_mat(lambd, m2)
    b_ext = rand_Rql(m2)

    s = sigma_n1_struct_vec(s1)

    # # Quadratic relations over Rq
    R2_is = [rand_Rq_mat(k * (m1 + ell), k * (m1 + ell)) for i in range(N)]
    r1_is = [rand_Rql(k * (m1 + ell)) for i in range(N)]
    r0_is = [-(s.dot_product(R2_is[i] * s) + r1_is[i].dot_product(s)) for i in range(N)]

    # Quadratic relations over Zq
    R2_p_is = [rand_Rq_mat_first_zero(k * (m1 + ell), k * (m1 + ell)) for i in range(M)]
    r1_p_is = [rand_Rql_first_zero(k * (m1 + ell)) for i in range(M)]
    r0_p_is = [-(s.dot_product(R2_p_is[i] * s) + r1_p_is[i].dot_product(s)) for i in range(M)]

    # Shortness in the infinity norm
    P_s = ZM(n_bin,m1)
    P_s[-1,-1] = 1
    f = -(P_s*s1) + rand_Rql_bin(n_bin)  # P_s*s1 + P_m*m + f is polynomial vector with binary coefficients

    # Shortness in the Euclidean norm
    E_s_is = [sg.block_matrix([[IM(n_is[0]), ZM(n_is[0],n_is[1]+n_is[2]+n_bin)]]) , sg.block_matrix([[ZM(n_is[1], n_is[0]), IM(n_is[1]), ZM(n_is[1], n_is[2] +n_bin)]]),sg.block_matrix([[ZM(n_is[2], n_is[0]+n_is[1]), IM(n_is[2]), ZM(n_is[2],n_bin)]])]
    v_is = [zv(i) for i in n_is]
    norms = [int(norm_Rql(E_s_is[i]*s1))+2 for i in range(Z)]
    norms = [norm_Rql(E_s_is[i]*s1) for i in range(Z)]

    gamma3 = 2.5

    B_is = [2*sg.sqrt(d*m_pke), sg.sqrt(4*d*4),sg.sqrt(4*d*4)]

    std3 = gamma3 * sg.sqrt(337) * (B_is[0] + B_is[1] +B_is[2] + d)
    rep_M3 = rej_bimodal_M(gamma3)

    roh = 1.64  # z3 norm bound (for kappa=128)

    # Approximate Shortness
    D_s = IR_q(q_pke)**-1 *sg.block_matrix([[R_to_S_mat(R_q,A_pke),ZM(n_pke,k_pke)],[R_to_S_mat(R_q,B_pke),IM(k_pke)]])

    u = IR_q(q_pke)**-1*stack_vec_Rql([R_to_S_vec(R_q,t0), R_to_S_vec(R_q,t1)])

    gamma4 = 12
    B_p = (d*q*m_pke + 1) *(sg.sqrt((n_pke + 1)*d)) 
    std4 = gamma4 * sg.sqrt(337) * B_p

    rep_M4 = rej_bimodal_M(gamma4)
    print("repetitions from 1 to 4", RF(rep_M1), RF(rep_M2), RF(rep_M3), RF(rep_M4))
    global rep_rate_abdlop_toolbox
    rep_rate_abdlop_toolbox = rep_M1 * 2*rep_M2 * rep_M3 * rep_M4

    squared_norm_diffs = [int(B_is[i]**2 - round(norms[i]**2))  for i in range(Z)]

    squared_norm_diffs =[squared_norm_diffs[i]%q for i in range(Z)]
    theta = sg.vector(R_q, [N_2_binary_Rq(squared_norm_diffs[i]) for i in range(Z)])

    tA_tB = A1 * stack_vec_Rql([s1, theta]) + A2 * s2 + zv(n)
    tA = tA_tB[:n]  # ABDLOP tA
    tB = tA_tB[n:]  # ABDLOP tB

    result = abdlop_toolbox(m1, m2, ell, n, k, N, M, n_bin, Z, n_is, n_p, lambd,
                          is_opti, rej_u_1, rej_u_2, rej_u_3, rej_u_4, get_challenge_u,
                          std1, rep_M1, std2, rep_M2, std3, rep_M3, std4, rep_M4, roh,
                          s1, s2, A1, A2, B_gamma, B_beta, B_ext, b_ext, theta, tA, tB,
                          R2_is, r1_is, r0_is,
                          R2_p_is, r1_p_is, r0_p_is,
                          P_s, f,
                          E_s_is, v_is, B_is,
                          D_s, u)
    iterations = 0
    while result == "Rejected":
        iterations = iterations + 1 
        result = abdlop_toolbox(m1, m2, ell, n, k, N, M, n_bin, Z, n_is, n_p, lambd,
                          is_opti, rej_u_1, rej_u_2, rej_u_3, rej_u_4, get_challenge_u,
                          std1, rep_M1, std2, rep_M2, std3, rep_M3, std4, rep_M4, roh,
                          s1, s2, A1, A2, B_gamma, B_beta, B_ext, b_ext, theta, tA, tB,
                          R2_is, r1_is, r0_is,
                          R2_p_is, r1_p_is, r0_p_is,
                          P_s, f,
                          E_s_is, v_is, B_is,
                          D_s, u)
    if result == False:
        return False
    else: 
        return True

def issue_test(num_iterations = 10):
    data = []
    run_times = []
    d = 128
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
    poly_mod = np.poly1d([1] + [0]*(N-1) + [1])

    for index in range(num_iterations):  
        start = time.perf_counter()   
        alph = math.ceil(2*math.sqrt(q2 +1)*((2*math.sqrt(2))*math.sqrt(N) + 1))
        G = np.zeros((1,2,N))
        G[0][0][-1] = 1
        G[0][1][-1] = np.rint(np.sqrt(q2))
        A, D, q1, q2, N, kappa, gamma, gamma_prime, alpha, opk_a0, opk_B, opk_u, osk,usk = AC_setup()
        A2 = A[:1, 1:]
        comm,rand = AC_registration(n=4,m=2,N=N,A=A,poly_mod=poly_mod,q1=q1,q2=q2,n1=1,l=l,G=G,tau=tau, usk=usk, beta=2)        
        miu = ZZl_2_Rql(rand.flatten())
        message = ZZl_2_Rql(usk.flatten())      

        miu = stack_vec_Rql([miu, message])   
        result = run_nizk_proof(miu)
        if result == True:
            u = []
            bound = (math.sqrt(3) + math.sqrt(2))*N*alph*2 + alph*math.sqrt(N*10)
            fcomm, cred, u = AC_issue(D,A2, opk_a0,opk_B,u,comm,N,poly_mod,q2,bound)
            rt = time.perf_counter() - start
            data.append({"Index": index, "Time":rt, "Success":True, "Error": 0, "message": message, "Credential":cred})
            run_times.append(rt)
        else:
            rt = time.perf_counter() - start
            run_times.append(rt)            
            data.append({"Index": index, "Time":rt, "Success":False, "Error": 1, "message": message, "Credential":[]})
    
    avg = np.average(np.array(run_times))
    med = np.median(np.array(run_times))
    min = np.min(np.array(run_times))
    max = np.max(np.array(run_times))

    print("--------------------------------------------------------------------------")
    print("ISSUING TEST")
    print("TIME: (avg, med, min, max)", avg, med, min, max)
    return data, avg, med, min, max

data, avg, med, min, max = issue_test()
csv_file = 'ISSUE_test.csv'
field_names_disclose = ["Index", "Time", "Success", "Error", "message", "Credential"]
write_file(csv_file,field_names_disclose, data)

res, cred = issue_test()

    
