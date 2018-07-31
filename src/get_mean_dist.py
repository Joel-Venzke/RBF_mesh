import numpy as np
from sklearn.neighbors import KDTree


def get_nearest_neighbors_from_set(node_set, k):
    tree = KDTree(node_set)
    return tree.query(node_set, k=k)


def get_average_spacing(node_set):
    nearest_dist, nearest_idx = get_nearest_neighbors_from_set(node_set, 2)
    return nearest_dist[:, 1].mean()


node_set_dir = "/Users/jvenzke/Repos/RBF_mesh/MD_node_sets"
delta_r = 0.5
max_md_order = 165
unit_spacing = np.zeros([max_md_order + 1])
unit_spacing[0] = float("inf")
for order in range(1, max_md_order + 1):
    num_points_per_shell = (order + 1)**2
    shell_nodes = np.loadtxt(node_set_dir + "/md" + str(order).zfill(3) + "." +
                             str(num_points_per_shell).zfill(5) + ".txt")
    unit_spacing[order] = get_average_spacing(shell_nodes)
print unit_spacing
for r in np.arange(delta_r / 2.0, 5, delta_r):
    print r, np.argmin(np.abs(delta_r - r * unit_spacing))