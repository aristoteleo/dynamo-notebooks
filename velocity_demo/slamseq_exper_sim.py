from numpy import *
from dynamo.tools.gillespie import *

def f_prop(C, a1, b1, a2, b2, a1_l, b1_l, a2_l, b2_l, K, n, be1, ga1, et1, de1, be2, ga2, et2, de2):
    # unlabeled mRNA
    u1 = C[0]
    s1 = C[1]
    u2 = C[2]
    s2 = C[3]

    # labeled mRNA
    w1 = C[4]
    l1 = C[5]
    w2 = C[6]
    l2 = C[7]

    # protein
    p1 = C[8]
    p2 = C[9]

    # propensities
    prop = np.zeros(18)
    # transcription
    prop[0] = a1 * p1**n / (K**n + p1**n) + b1 * K**n / (K**n + p2**n)         # 0 -> u1
    prop[1] = a2 * p2**n / (K**n + p2**n) + b2 * K**n / (K**n + p1**n)         # 0 -> u2
    prop[2] = a1_l * p1**n / (K**n + p1**n) + b1_l * K**n / (K**n + p2**n)     # 0 -> w1
    prop[3] = a2_l * p2**n / (K**n + p2**n) + b2_l * K**n / (K**n + p1**n)     # 0 -> w2
    # splicing
    prop[4] = be1 * u1      # u1 -> s1
    prop[5] = be2 * u2      # u2 -> s2
    prop[6] = be1 * w1      # w1 -> l1
    prop[7] = be2 * w2      # w2 -> l2
    # mRNA degradation
    prop[8] = ga1 * s1      # s1 -> 0
    prop[9] = ga2 * s2      # s2 -> 0
    prop[10] = ga1 * l1     # l1 -> 0
    prop[11] = ga2 * l2     # l2 -> 0
    # translation
    prop[12] = et1 * s1      # s1 --> p1
    prop[13] = et2 * s2      # s2 --> p2
    prop[14] = et1 * l1      # l1 --> p1
    prop[15] = et2 * l2      # l2 --> p2
    # protein degradation
    prop[16] = de1 * p1      # p1 -> 0
    prop[17] = de2 * p2      # p2 -> 0

    return prop

def f_stoich():
    # species
    u1 = 0
    s1 = 1
    u2 = 2
    s2 = 3
    w1 = 4
    l1 = 5
    w2 = 6
    l2 = 7
    p1 = 8
    p2 = 9

    # stoichiometry matrix
    # transcription
    stoich = np.zeros((18, 10))
    stoich[0, u1] = 1       # 0 -> u1
    stoich[1, u2] = 1       # 0 -> u2
    stoich[2, w1] = 1       # 0 -> w1
    stoich[3, w2] = 1       # 0 -> w2
    # splicing
    stoich[4, u1] = -1      # u1 -> s1
    stoich[4, s1] = 1
    stoich[5, u2] = -1      # u2 -> s2
    stoich[5, s2] = 1
    stoich[6, w1] = -1      # w1 -> l1
    stoich[6, l1] = 1
    stoich[7, w2] = -1      # w2 -> l2
    stoich[7, l2] = 1
    # mRNA degradation
    stoich[8, s1] = -1      # s1 -> 0
    stoich[9, s2] = -1      # s2 -> 0
    stoich[10, l1] = -1     # l1 -> 0
    stoich[11, l2] = -1     # l2 -> 0
    # translation
    stoich[12, p1] = 1      # s1 --> p1
    stoich[13, p2] = 1      # s2 --> p2
    stoich[14, p1] = 1      # l1 --> p1
    stoich[15, p2] = 1      # l2 --> p2
    # protein degradation
    stoich[16, p1] = -1      # p1 -> 0
    stoich[17, p2] = -1      # p2 -> 0

    return stoich

def simulate(a1, b1, a2, b2, a1_l, b1_l, a2_l, b2_l, K, n, be1, ga1, et1, de1, be2, ga2, et2, de2, C0, t_span, n_traj, report=False):
    stoich = f_stoich()
    update_func = lambda C, mu: C + stoich[mu, :]

    trajs_T = [[]] * n_traj
    trajs_C = [[]] * n_traj

    for i in range(n_traj):
        T, C = directMethod(lambda C: f_prop(C, a1, b1, a2, b2, a1_l, b1_l, a2_l, b2_l, K, n, be1, ga1, et1, de1, be2, ga2, et2, de2), update_func, t_span, C0[i])
        trajs_T[i] = T
        trajs_C[i] = C
        if report:
            print ('Iteration %d/%d finished.'%(i+1, n_traj), end='\r')
    return trajs_T, trajs_C