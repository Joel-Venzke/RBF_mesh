from Node_Set import Node_Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import eigs
# notes:
# random rotation cleans up the issue at large r (I think)
# the small r limit needs to be fixed. A quasi uniform node layout should do this
# find average distance to nearest neighbor on unit sphere and match this to delta_r
# scaling by the radius. use this distance for outer box of quasi uniform node set

node_set = Node_Set(
    r_max=500,
    delta_r=1.0,
    stencil_size=50,
    md_degree=10,
    rbf_order=7,
    poly_order=2,
    exp_order=2, )

# node_set.check_quality()
node_set.calculate_opperator_weights()

print "testing laplace"
function = 1. * node_set.node_set[:,
                                  0]**2 + 1. * node_set.node_set[:,
                                                                 1]**2 + 1. * node_set.node_set[:,
                                                                                                2]**2 + 4

plt.scatter(node_set.laplace.nonzero()[1], -node_set.laplace.nonzero()[0], s=1)
plt.axis('equal')
plt.show()
function = np.abs(node_set.laplace.dot(function) - 6)

plt.semilogy(node_set.node_set[:, 0], function, 'o')
plt.xlabel("x-axis")
plt.show()
plt.semilogy(node_set.node_set[:, 1], function, 'o')
plt.xlabel("y-axis")
plt.show()
plt.semilogy(node_set.node_set[:, 2], function, 'o')
plt.xlabel("z-axis")
plt.show()
plt.hist2d(
    np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
            node_set.node_set[:, 2]**2),
    np.log10(function + 1e-18),
    bins=100,
    norm=LogNorm())
plt.xlabel("r-axis")
plt.show()

plt.semilogy(function, 'o')
plt.xlabel("index")
plt.show()

print "first_deriviative test"
r = node_set.r_max + node_set.delta_r / 2.0
function_in = 1. / 2.0 * (node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]
                          **2 + node_set.node_set[:, 2]**2 - r * r)
function = node_set.first_deriviative[0].dot(
    function_in) - node_set.node_set[:, 0] + node_set.first_deriviative[1].dot(
        function_in
    ) - node_set.node_set[:, 1] + node_set.first_deriviative[2].dot(
        function_in) - node_set.node_set[:, 2]
plt.semilogy(node_set.node_set[:, 0], function, 'o')
plt.xlabel("x-axis")
plt.show()
plt.semilogy(node_set.node_set[:, 1], function, 'o')
plt.xlabel("y-axis")
plt.show()
plt.semilogy(node_set.node_set[:, 2], function, 'o')
plt.xlabel("z-axis")
plt.show()
plt.hist2d(
    np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
            node_set.node_set[:, 2]**2),
    np.log10(np.abs(function) + 1e-18),
    bins=100,
    norm=LogNorm())
plt.xlabel("r-axis")
plt.show()

plt.semilogy(function, 'o')
plt.xlabel("index")
plt.show()

print "exp test"
a = 2.0
function_in = np.exp(
    -a * np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
                 node_set.node_set[:, 2]**2))
function = node_set.first_deriviative[0].dot(
    function_in) - node_set.node_set[:, 0] * a * np.exp(
        -a * np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
                     node_set.node_set[:, 2]**2)) / np.sqrt(
                         node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**
                         2 + node_set.node_set[:, 2]**2)
plt.semilogy(node_set.node_set[:, 0], function, 'o')
plt.xlabel("x-axis")
plt.show()
plt.semilogy(node_set.node_set[:, 1], function, 'o')
plt.xlabel("y-axis")
plt.show()
plt.semilogy(node_set.node_set[:, 2], function, 'o')
plt.xlabel("z-axis")
plt.show()
plt.hist2d(
    np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
            node_set.node_set[:, 2]**2),
    np.log10(np.abs(function) + 1e-18),
    bins=100,
    norm=LogNorm())
plt.xlabel("r-axis")
plt.show()

plt.semilogy(function, 'o')
plt.xlabel("index")
plt.show()
