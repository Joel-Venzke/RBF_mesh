from Node_Set import Node_Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import eigs
import sys


def hydrogen(node):
    return -1.0 / np.sqrt((node**2).sum())


for exp_order in [1.0, 0.0]:
    for dr in (5 * 10**np.arange(-2, -0.5, 0.2))[::-1]:

        print(exp_order, end=" ", flush=True)

        # attach exponential of n/Z (attaches exp(-Zr/n))
        # where n is the grounds state principle quantum number
        # and Z in the nuclear charge
        # (Z = c_0+z_c for SAE since they are both 1/r at small r)
        node_set = Node_Set(
            r_max=30,
            delta_r=dr,
            md_degree=8,
            rbf_order=7,
            poly_order=3,
            exp_order=exp_order,
            stencil_size=54,
            ecs_size=0,
            hyperviscosity_order=8,
            save=False,
            quiet=True,
            node_set_dir="/Users/joelvenzke/Repos/RBF_mesh/MD_node_sets")

        print(dr, end=" ", flush=True)
        k = 5
        ham = node_set.apply_potential(-node_set.laplace / 2.0, hydrogen)
        eig_val, eig_vec = eigs(ham, k=k, which="SR")

        for idx, current in enumerate(eig_val):
            if idx == 0:
                print((-0.5 - current), end=" ")
            else:
                print((-0.125 - current), end=" ")
        print()
