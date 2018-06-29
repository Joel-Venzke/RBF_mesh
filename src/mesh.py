from Node_Set import Node_Set
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
# notes:
# random rotation cleans up the issue at large r (I think)
# the small r limit needs to be fixed. A quasi uniform node layout should do this
# find average distance to nearest neighbor on unit sphere and match this to delta_r
# scaling by the radius. use this distance for outer box of quasi uniform node set

node_set = Node_Set(
    r_max=100,
    delta_r=0.4,
    stencil_size=50,
    md_degree=31,
    rbf_order=7,
    poly_order=5,
    exp_order=1, )

# node_set.check_quality()
node_set.calculate_opperator_weights()

function = 1. * node_set.node_set[:,
                                  0]**2 + 1. * node_set.node_set[:,
                                                                 1]**2 + 1. * node_set.node_set[:,
                                                                                                2]**2 + 4

plt.scatter(node_set.laplace.nonzero()[1], -node_set.laplace.nonzero()[0], s=1)
plt.axis('equal')
plt.show()
function = node_set.laplace.dot(function)

plt.semilogy(node_set.node_set[:, 0], function, 'o')
plt.xlabel("x-axis")
plt.show()
plt.semilogy(node_set.node_set[:, 1], function, 'o')
plt.xlabel("y-axis")
plt.show()
plt.semilogy(node_set.node_set[:, 2], function, 'o')
plt.xlabel("z-axis")
plt.show()

plt.semilogy(function, 'o')
plt.xlabel("index")
plt.show()