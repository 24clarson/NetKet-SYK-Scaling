from sys import argv
import os
from math import sqrt
from dynamite import config
from dynamite.operators import op_sum, op_product, Operator
from dynamite.tools import track_memory, get_max_memory_usage, get_cur_memory_usage
from dynamite.extras import majorana
from dynamite.subspaces import SpinConserve
from itertools import combinations
import numpy as np
from scipy.special import comb
import scipy
import argparse
from timeit import default_timer
from datetime import timedelta
parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, required=True, help='system size')
parser.add_argument('-J', type=float, required=False, default=1.0, help='coupling strength')
parser.add_argument('--seed', type=int, required=True, help='sample seed')
parser.add_argument('--dirc', type=str, required=False, default='./',  help='output directory')
parser.add_argument('--gpu', type=int, required=False, default=0, help='use GPU')
args = parser.parse_args()
if args.gpu == 1:
    config.initialize(gpu=True)  # for GPU
config.shell=True
L = args.L
config.L = L
J = args.J
seed = args.seed
msc_dir = args.dirc

config.subspace=SpinConserve(L, L//2)

if not os.path.isdir(msc_dir):
    os.mkdir(msc_dir)# cache Majoranas to save time
M = [(majorana(idx), majorana(idx+1)) for idx in range(0, 2*L, 2)]
RNG = np.random.default_rng(seed)
CPLS_J = RNG.normal(0, 1/sqrt(L), size=comb(L,2,exact=True))
CPLS_V = RNG.normal(0, 1/sqrt(L), size=comb(L,2,exact=True))

def build_H(save=False):
    # sum [ (psi_1+i*psi_2) * (psi_3-i*psi_4) + (psi_3+i*psi_4) * (psi_1-i*psi_2)]
    ops1 = [op_sum([
            op_product([op_sum([M[idx[0]][0],-1j*M[idx[0]][1]]),op_sum([M[idx[1]][0],1j*M[idx[1]][1]])]),
            op_product([op_sum([M[idx[1]][0],-1j*M[idx[1]][1]]),op_sum([M[idx[0]][0],1j*M[idx[0]][1]])]),
        ]) for idx in combinations(range(L), 2)]
    H1 = op_sum([cj*op/4 for cj,op in zip(CPLS_J, ops1)])
    # sum [ (psi_1-i*psi_2) * (psi_1+i*psi_2) * (psi_3-i*psi_4) * (psi_3+i*psi_4)]

    ops2 = [op_sum([
            op_product([
                op_product([op_sum([M[idx[0]][0],-1j*M[idx[0]][1]]),op_sum([M[idx[0]][0],1j*M[idx[0]][1]])]),
                op_product([op_sum([M[idx[1]][0],-1j*M[idx[1]][1]]),op_sum([M[idx[1]][0],1j*M[idx[1]][1]])]),
            ]),
            op_product([
                op_product([op_sum([M[idx[1]][0],-1j*M[idx[1]][1]]),op_sum([M[idx[1]][0],1j*M[idx[1]][1]])]),
                op_product([op_sum([M[idx[0]][0],-1j*M[idx[0]][1]]),op_sum([M[idx[0]][0],1j*M[idx[0]][1]])]),
            ]),
        ]) for idx in combinations(range(L), 2)]

    H2 = op_sum([cv*op2/16 for cv,op2 in zip(CPLS_V, ops2)])
    H = op_sum([H1, H2])
    # scipy.sparse.save_npz("ham-L12-seed0", H.to_numpy(sparse=True), compressed=True)
    if save == True:
        H.save(os.path.join(msc_dir,f'H_L={L}_seed={seed}.msc'))
    return H

def main():
    start = default_timer()
    H = build_H(save=True)
    eigvals, eigvecs = H.eigsolve(getvecs=True, nev=1)
    print(f'build Hamiltonian, L={L}, eigvals={eigvals}, seed={seed}; time elapsed: {timedelta(0,default_timer()-start)}')
    # print([eigvecs[1].entanglement_entropy([i for i in range(j)]) for j in range(L)])
    
if __name__ == '__main__':
    main()
