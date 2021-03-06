#!/usr/bin/env python3

import math
import random
import pykdtw

random.seed(7)  # set seed

P = 20  # Length of one sequence
Q = 20  # Length of the other sequence

# Random sequences
A = [random.random() for _ in range(P)]
B = [random.random() for _ in range(Q)]

# Make the P \times Q distance matrix
dist_mat = pykdtw.MatrixF64(P, Q)
for i, a in enumerate(A):
    for j, b in enumerate(B):
        dist_mat.set(i, j, abs(a - b))  # Lp norm

# If you want to make another matrix, you can use
# dist_mat.resize(P,Q) not to reconstruct the matrix

# Parameters for KDTW
C = 3  # non negative constant (typically 3)
NU = [0.01, 0.1, 1]  # stiffness parameter (ν')

for nu in NU:
    # Compute the kernel value (in log)
    log_K = pykdtw.log_KDTW(dist_mat, nu, C)
    print(f"K = {math.exp(log_K):g}\t(ν' = {nu})")
