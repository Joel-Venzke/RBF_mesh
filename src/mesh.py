from Node_Set import Node_Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import eigs
import sys

# notes:
# random rotation cleans up the issue at large r (I think)
# the small r limit needs to be fixed. A quasi uniform node layout should do this
# find average distance to nearest neighbor on unit sphere and match this to delta_r
# scaling by the radius. use this distance for outer box of quasi uniform node set


def hydrogen(node):
    return -1.0 / np.sqrt((node**2).sum())


# attach exponential of n/Z (attaches exp(-Zr/n))
# where n is the grounds state principle quantum number
# and Z in the nuclear charge (Z = c_0+z_c for SAE since they are both 1/r at small r)
# float(sys.argv[1])
node_set = Node_Set(
    r_max=40,
    delta_r=0.3,
    md_degree=4,
    rbf_order=7,
    poly_order=2,
    exp_order=1.0,
    stencil_size=54,
    ecs_size=5,
    hyperviscosity_order=8,
    save=True,
    quiet=False,
    node_set_dir="/Users/jvenzke/Repos/RBF_mesh/MD_node_sets")

print "Calculating Target"
k = 15
ham = node_set.apply_potential(-node_set.laplace / 2.0, hydrogen)
eig_val, eig_vec = eigs(ham, k=k, which="SR")

for current in eig_val:
    print current
print