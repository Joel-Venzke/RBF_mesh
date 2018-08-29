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
    return -1.0 / np.sqrt((node**2).sum()) + 0.05 * node[0]


def SAE(node):
    r = np.sqrt((node**2).sum())
    # He: -0.918
    # c_0 = 1.0
    # z_c = 1.0
    # r_0 = 2.3526
    # a_list = [0.6628]
    # b_list = [4.0762]

    # Ne: -0.808
    # c_0 = 1.0
    # z_c = 9.0
    # r_0 = 2.0872
    # a_list = [-5.4072, 1.0374]
    # b_list = [4.1537, 67.1114]

    # Ar: -0.577
    c_0 = 1.0
    z_c = 17.0
    r_0 = 0.8472
    a_list = [-15.3755, -27.7431, 2.1705]
    b_list = [1.2484, 4.3924, 87.9345]

    # Kr: -0.524
    # c_0 = 1.0
    # z_c = 35.0
    # r_0 = 1.7906
    # a_list = [-61.9891, -1662.22, 1629.864, 3.5109]
    # b_list = [3.3015, 18.957, 19.1727, 272.3485]

    # leave me alone
    pot = -c_0 / r - z_c * np.exp(-r_0 * r) / r
    for idx in np.arange(len(a_list)):
        pot -= a_list[idx] * np.exp(-b_list[idx] * r)
    return pot


# attach exponential of n/Z (attaches exp(-Zr/n))
# where n is the grounds state principle quantum number
# and Z in the nuclear charge (Z = c_0+z_c for SAE since they are both 1/r at small r)
# float(sys.argv[1])
node_set = Node_Set(
    r_max=25,
    delta_r=0.3,
    md_degree=10,
    rbf_order=7,
    poly_order=2,
    exp_order=0.0,
    stencil_size=64,
    ecs_size=5,
    hyperviscosity_order=8,
    save=True,
    quiet=False)

y_floor = 1e-60
y_min = np.log10(y_floor)
y_max = 10
epslion = 1e-3

# print "Plotting matrix"
# plt.hist2d(
#     node_set.laplace.nonzero()[1],
#     -node_set.laplace.nonzero()[0],
#     bins=node_set.node_set.shape[0] / 100,
#     norm=LogNorm())
# plt.axis('equal')
# plt.show()

# print "testing laplace"
# function = 1. * node_set.node_set[:,
#                                   0]**2 + 1. * node_set.node_set[:,
#                                                                  1]**2 + 1. * node_set.node_set[:,
#                                                                                                 2]**2 + 4
# function = np.abs(node_set.laplace.dot(function) - 6)
# # plt.semilogy(node_set.node_set[:, 0], function, 'o')
# # plt.xlabel("x-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 1], function, 'o')
# # plt.xlabel("y-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 2], function, 'o')
# # plt.xlabel("z-axis")
# # plt.show()
# plt.hist2d(
#     np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#             node_set.node_set[:, 2]**2),
#     np.log10(function + y_floor),
#     bins=100,
#     norm=LogNorm())
# plt.xlabel("r-axis")
# plt.ylim([y_min, y_max])
# plt.xlim(xmin=0, xmax=node_set.r_max)
# plt.show()

# # plt.semilogy(function, 'o')
# # plt.xlabel("index")
# # plt.show()

# print "first_deriviative test"
# r = node_set.r_max + node_set.delta_r / 2.0
# function_in = 1. / 2.0 * (node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]
#                           **2 + node_set.node_set[:, 2]**2 - r * r)
# function = node_set.first_deriviative[0].dot(
#     function_in) - node_set.node_set[:, 0] + node_set.first_deriviative[1].dot(
#         function_in
#     ) - node_set.node_set[:, 1] + node_set.first_deriviative[2].dot(
#         function_in) - node_set.node_set[:, 2]
# # plt.semilogy(node_set.node_set[:, 0], function, 'o')
# # plt.xlabel("x-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 1], function, 'o')
# # plt.xlabel("y-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 2], function, 'o')
# # plt.xlabel("z-axis")
# # plt.show()
# plt.hist2d(
#     np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#             node_set.node_set[:, 2]**2),
#     np.log10(np.abs(function) + y_floor),
#     bins=100,
#     norm=LogNorm())
# plt.xlabel("r-axis")
# plt.ylim([y_min, y_max])
# plt.xlim(xmin=0, xmax=node_set.r_max)
# plt.show()

# print "dx test"
# function_in = 1. / 8.0 * (node_set.node_set[:, 0]**8)
# function = node_set.first_deriviative[0].dot(
#     function_in) - node_set.node_set[:, 0]**7
# # plt.semilogy(node_set.node_set[:, 0], function, 'o')
# # plt.xlabel("x-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 1], function, 'o')
# # plt.xlabel("y-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 2], function, 'o')
# # plt.xlabel("z-axis")
# # plt.show()
# plt.hist2d(
#     np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#             node_set.node_set[:, 2]**2),
#     np.log10(np.abs(function) + y_floor),
#     bins=100,
#     norm=LogNorm())
# plt.xlabel("r-axis")
# plt.ylim([y_min, y_max])
# plt.xlim(xmin=0, xmax=node_set.r_max)
# plt.show()

# print "dy test"
# function_in = 1. / 8.0 * (node_set.node_set[:, 1]**8)
# function = node_set.first_deriviative[1].dot(
#     function_in) - node_set.node_set[:, 1]**7
# # plt.semilogy(node_set.node_set[:, 0], function, 'o')
# # plt.xlabel("x-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 1], function, 'o')
# # plt.xlabel("y-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 2], function, 'o')
# # plt.xlabel("z-axis")
# # plt.show()
# plt.hist2d(
#     np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#             node_set.node_set[:, 2]**2),
#     np.log10(np.abs(function) + y_floor),
#     bins=100,
#     norm=LogNorm())
# plt.xlabel("r-axis")
# plt.ylim([y_min, y_max])
# plt.xlim(xmin=0, xmax=node_set.r_max)
# plt.show()

# print "dz test"
# function_in = 1. / 8.0 * (node_set.node_set[:, 2]**8)
# function = node_set.first_deriviative[2].dot(
#     function_in) - node_set.node_set[:, 2]**7
# # plt.semilogy(node_set.node_set[:, 0], function, 'o')
# # plt.xlabel("x-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 1], function, 'o')
# # plt.xlabel("y-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 2], function, 'o')
# # plt.xlabel("z-axis")
# # plt.show()
# plt.hist2d(
#     np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#             node_set.node_set[:, 2]**2),
#     np.log10(np.abs(function) + y_floor),
#     bins=100,
#     norm=LogNorm())
# plt.xlabel("r-axis")
# plt.ylim([y_min, y_max])
# plt.xlim(xmin=0, xmax=node_set.r_max)
# plt.show()

# # plt.semilogy(function, 'o')
# # plt.xlabel("index")
# # plt.show()

# print "exp test"
# a = 0.5
# function_in = np.exp(
#     -a * np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#                  node_set.node_set[:, 2]**2))
# function = node_set.first_deriviative[0].dot(
#     function_in) - node_set.node_set[:, 0] * a * np.exp(
#         -a * np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#                      node_set.node_set[:, 2]**2)) / np.sqrt(
#                          node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**
#                          2 + node_set.node_set[:, 2]**2)
# # plt.semilogy(node_set.node_set[:, 0], function, 'o')
# # plt.xlabel("x-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 1], function, 'o')
# # plt.xlabel("y-axis")
# # plt.show()
# # plt.semilogy(node_set.node_set[:, 2], function, 'o')
# # plt.xlabel("z-axis")
# # plt.show()
# plt.hist2d(
#     np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#             node_set.node_set[:, 2]**2),
#     np.log10(np.abs(function) + y_floor),
#     bins=100,
#     norm=LogNorm())
# plt.xlabel("r-axis")
# plt.ylim([y_min, y_max])
# plt.xlim(xmin=0, xmax=node_set.r_max)
# plt.show()

# plt.semilogy(function, 'o')
# plt.xlabel("index")
# plt.show()
# print "Calculating Target"
# k = 1035
# target = -0.5
# ham = node_set.apply_potential(
#     1.0j * node_set.laplace * epslion - node_set.laplace / 2.0, hydrogen)
# eig_val, eig_vec = eigs(ham, k=k, which="SR")
# ground_state_idx = np.argmin(np.abs(eig_val.real - target))
# if ground_state_idx != 0:
#     eig_val = eig_val[ground_state_idx:]
#     eig_vec = eig_vec[:, ground_state_idx:]
#     k -= ground_state_idx
# idx = 0
# for i in np.arange(1, k + 1):
#     n = i**2
#     if idx + n <= k:
#         percent_error = 100 * np.abs(
#             -0.5 / i**2 - eig_val[idx:idx + n].real) / (0.5 / i**2)
#         print "Error for n=" + str(i) + " (mean, min, max):", np.mean(
#             percent_error), np.min(percent_error), np.max(percent_error)
#     idx = idx + n
# print
# idx = 0
# for i in np.arange(1, k + 1):
#     n = i**2
#     if idx + n <= k:
#         print "Energies n=" + str(i) + ":", -0.5 / i**2, eig_val[idx:idx + n]
#     idx = idx + n
# print

# ground_state = np.exp(-np.sqrt(node_set.node_set[:, 0]**2 + node_set.
#                                node_set[:, 1]**2 + node_set.node_set[:, 2]**2))
# ground_state /= np.sqrt(
#     np.sum(np.conjugate(ground_state) * node_set.weights * ground_state))
# print "norm", np.sum(
#     np.conjugate(ground_state) * node_set.weights * ground_state)
# print "<r>k error:", 1 - np.sum(
#     np.conjugate(ground_state) * node_set.weights *
#     np.sqrt(node_set.node_set[:, 0]**2 + node_set.node_set[:, 1]**2 +
#             node_set.node_set[:, 2]**2) * ground_state).real

# print
# print
k = 50
ham = node_set.apply_potential(-node_set.laplace / 2.0, hydrogen)
eig_val, eig_vec = eigs(ham, k=k, which="SR")
print np.max(eig_val.imag)

print eig_vec.shape

for e_val in eig_val:
    print e_val.real, "\t", e_val.imag

plt.scatter(eig_val.real, eig_val.imag)
# plt.ylim([-10, 10])
plt.show()
print
print

k = 50
ham = node_set.apply_potential(
    1.0j * node_set.laplace * epslion - node_set.laplace / 2.0, hydrogen)
eig_val, eig_vec = eigs(ham, k=k, which="SR")
print np.max(eig_val.imag)

print eig_vec.shape

for e_val in eig_val:
    print e_val.real, "\t", e_val.imag

plt.scatter(eig_val.real, eig_val.imag)
# plt.ylim([-10, 10])
plt.show()
print
print

# ham = node_set.apply_potential(-node_set.laplace_ecs / 2.0, hydrogen)
# eig_val, eig_vec = eigs(ham, k=k, which="SR")
# ground_state_idx = np.argmin(np.abs(eig_val - target))
# print ground_state_idx
# if ground_state_idx != 0:
#     eig_val = eig_val[ground_state_idx:]
#     eig_vec = eig_vec[:, ground_state_idx:]
#     k -= ground_state_idx
# idx = 0
# for i in np.arange(1, k + 1):
#     n = i**2
#     if idx + n <= k:
#         percent_error = 100 * np.abs(
#             -0.5 / i**2 - eig_val[idx:idx + n].real) / (0.5 / i**2)
#         print "Error for n=" + str(i) + " (mean, min, max):", np.mean(
#             percent_error), np.min(percent_error), np.max(percent_error)
#     idx = idx + n
# print
# idx = 0
# for i in np.arange(1, k + 1):
#     n = i**2
#     if idx + n <= k:
#         print "Energies n=" + str(i) + ":", -0.5 / i**2, eig_val[idx:
#                                                                  idx + n].real
#     idx = idx + n
# print