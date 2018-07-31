import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import comb
from scipy.sparse import csr_matrix
import sobol_seq
import sys
from scipy.linalg import toeplitz, pascal
from numpy.matlib import repmat
import h5py


class Node_Set:
    def __init__(self,
                 r_max=100,
                 delta_r=0.5,
                 stencil_size=56,
                 md_degree=8,
                 rbf_order=7,
                 poly_order=3,
                 exp_order=1,
                 ecs_size=5,
                 save=True,
                 quiet=False,
                 node_set_dir="/Users/jvenzke/Repos/RBF_mesh/MD_node_sets"):
        self.quiet = quiet
        if not self.quiet:
            print "Creating Node Set"
        self.r_max = r_max
        self.delta_r = delta_r
        self.md_degree = md_degree
        self.stencil_size = stencil_size
        self.rbf_order = rbf_order
        self.poly_order = poly_order
        self.exp_order = exp_order
        self.node_set_dir = node_set_dir
        self.ecs_size = ecs_size

        if not self.quiet:
            print
            print "r_max:\t\t" + str(self.r_max)
            print "delta_r:\t" + str(self.delta_r)
            print "ecs_size:\t" + str(self.ecs_size)
            print "md_degree:\t" + str(self.md_degree)
            print "rbf_order:\t" + str(self.rbf_order)
            print "poly_order:\t" + str(self.poly_order)
            print "exp_order:\t" + str(self.exp_order)
            print "stencil_size:\t" + str(self.stencil_size)
            print "node_set_dir:\t" + str(self.node_set_dir)
            print

        if self.rbf_order < 3 or self.rbf_order % 2 != 1:
            exit("RBF order must be >3 and odd")
        self.create_node_set()
        self.full_node_set = np.concatenate(
            [self.node_set, self.boundary_nodes])
        self.full_node_set_ecs = np.concatenate(
            [self.node_set_ecs, self.boundary_nodes_ecs])
        # used for testing only. do not uncomment
        # self.full_node_set = self.node_set
        self.radius = np.sqrt((self.node_set**2).sum(axis=1))
        self.theta = np.angle(self.node_set[:, 2] + 1.0j * np.sqrt(
            (self.node_set[:, :2]**2).sum(axis=1)))
        self.phi = np.angle(self.node_set[:, 0] + 1.0j * self.node_set[:, 1])
        self.num_nodes = self.node_set.shape[0]
        if not self.quiet:
            print "Number of nodes:", self.num_nodes
            print "Getting Nearest Neighbors"
        self.nearest_dist, self.nearest_idx = self.get_nearest_neighbors()
        self.calculate_operator_weights()
        if save:
            self.save_node_set()
        if not self.quiet:
            print "Node set complete"

    # based on http://slideflix.net/doc/4183369/gregory-s-quadrature-method
    # pages 4-5 (accessed: 2018)
    def gregory_weights(self, n_nodes, order):
        if order < 2 or order > n_nodes:
            # use midpoint rule
            return np.ones(n_nodes) * self.delta_r
        # do the gregory thin
        r = 1.0 / np.arange(1, order + 1)
        array = np.zeros([order - 1])
        array[0] = r[0]
        matrix = toeplitz(r[0:order - 1], array)
        # create the sequence of Gregory coefficients
        gc = np.linalg.solve(matrix, r[1:])
        weights = np.ones(n_nodes) * self.delta_r
        # create matrix
        gc_repmat = repmat(gc, 1, order - 1).reshape(
            [order - 1, order - 1]).transpose() * self.delta_r
        # create pascal matrix
        pascal_lower = pascal(order - 1, kind='lower', exact=False)
        # match matlab
        pascal_lower[:, 1::2] *= -1.0
        weights_updates = np.sum(gc_repmat * pascal_lower, axis=0)[1:]
        weights[:order - 2] -= weights_updates
        weights[n_nodes - order + 2:] -= weights_updates[::-1]
        return weights

    def rand_rotation_matrix(self, deflection=1.0, randnums=None):
        """
        Creates a random rotation matrix.

        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
        """
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

        if randnums is None:
            randnums = np.random.uniform(size=(3, ))

        theta, phi, z = randnums

        theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0 * deflection  # For magnitude of pole deflection.

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.

        r = np.sqrt(z)
        Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.

        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M

    def get_poly_terms(self, nodes, poly_order, dtype='float'):
        current_dim = nodes.shape[1]
        nodes = np.array(nodes, dtype=dtype)

        # for 1D return array that is [1, x, x^2, ..., x^(poly_order)]
        # note it becomes 2D to account for all node locations
        if current_dim == 1:
            # create array with coordinates
            poly_set = np.tile(nodes[:, -1], (poly_order + 1, 1)).transpose()
            # deal with constant term
            poly_set[:, 0] = 1.0
            # multiply across rows
            poly_set = np.cumprod(poly_set, axis=1)
            return poly_set
        else:
            # get the array [1, x, x^2, ..., x^(poly_order)] for this dim
            current_dim_poly = self.get_poly_terms(
                nodes[:, -1].reshape((nodes.shape[0], 1)),
                poly_order,
                dtype=dtype)
            # get the array [1, x, y, ..., x^2, xy, y^2, ..., z^2, ..., x^(poly_order)]
            lower_dim_poly = self.get_poly_terms(
                nodes[:, :-1], poly_order, dtype=dtype)
            # number of poly terms in this dimension
            num_poly_terms = comb(
                poly_order + current_dim, current_dim, exact=True)
            # allocate array
            poly_set = np.zeros((nodes.shape[0], num_poly_terms), dtype=dtype)
            # populate new array
            col = 0
            for poly_deg in range(
                    0, poly_order + 2):  # loop over all poly degrees
                for cur_degree in range(0, poly_deg):  # loop over poly degrees
                    # find upper and lower bound for dim-1 poly list
                    lower_dim_poly_upper_idx = comb(
                        (poly_deg - cur_degree - 1) + current_dim - 1,
                        current_dim - 1,
                        exact=True)
                    lower_dim_poly_lower_idx = comb(
                        (poly_deg - cur_degree - 2) + current_dim - 1,
                        current_dim - 1,
                        exact=True)
                    # find number of poly terms to multiply together
                    col_offset = lower_dim_poly_upper_idx - lower_dim_poly_lower_idx
                    # calculate terms
                    poly_set[:, col:col +
                             col_offset] = lower_dim_poly[:,
                                                          lower_dim_poly_lower_idx:
                                                          lower_dim_poly_upper_idx] * current_dim_poly[:, cur_degree].reshape(
                                                              (current_dim_poly.
                                                               shape[0], 1))
                    # update index location
                    col += col_offset
            return poly_set

    def apply_potential(self, matrix, potential_function):
        ret_mat = matrix.copy()
        for idx in np.arange(ret_mat.shape[0]):
            potential = potential_function(self.node_set[idx])
            ret_mat[idx, idx] += potential
        return ret_mat

    def rotation_matrix_z(self, phi):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        return np.array([[cos_phi, -sin_phi, 0.0], [sin_phi, cos_phi, 0.0],
                         [0.0, 0.0, 1.0]])

    def rotation_matrix_y(self, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([[cos_theta, 0.0, sin_theta], [0.0, 1.0, 0.0],
                         [-sin_theta, 0.0, cos_theta]])

    def rotation_matrix_aline_z(self, theta, phi):
        return np.matmul(
            self.rotation_matrix_y(-theta), self.rotation_matrix_z(-phi))

    def get_average_spacing(self, node_set):
        nearest_dist, nearest_idx = self.get_nearest_neighbors_from_set(
            node_set, 2)
        return nearest_dist[:, 1].mean()

    def get_md_degree(self, radius):
        # the average spacing on the unit sphere given some order node set
        # unit_spacing[md_order] returns spacing
        # note: md_order = 0 returns infinity since the nodes start at order 1
        unit_spacing = np.array([
            float("inf"), 1.63299316, 1.14222871, 0.87743913, 0.69598147,
            0.58057049, 0.49911459, 0.4360181, 0.38830468, 0.35175229,
            0.31839752, 0.29193107, 0.27046536, 0.25208047, 0.23474402,
            0.21998443, 0.20693719, 0.19572402, 0.18541912, 0.17584254,
            0.16744181, 0.15989746, 0.15256163, 0.14673463, 0.14026741,
            0.13504948, 0.13029142, 0.1256364, 0.12131351, 0.11747782,
            0.11365009, 0.10996617, 0.10672006, 0.10354628, 0.10059035,
            0.09775551, 0.09518732, 0.09280582, 0.09043152, 0.08811574,
            0.08597759, 0.08380385, 0.08186108, 0.08002145, 0.0782486,
            0.07653958, 0.07491495, 0.07333657, 0.07181651, 0.0704031,
            0.06907399, 0.06767164, 0.06640887, 0.06519383, 0.06402736,
            0.06284692, 0.0617011, 0.0606756, 0.05968297, 0.05865884,
            0.05766154, 0.05679024, 0.05585916, 0.05498633, 0.05412859,
            0.05327777, 0.05249474, 0.05173961, 0.05097561, 0.0502528,
            0.04958018, 0.04880479, 0.04817526, 0.04752193, 0.04690221,
            0.04625873, 0.04573772, 0.04510057, 0.04450767, 0.04396571,
            0.04344229, 0.04286121, 0.04238041, 0.04184752, 0.04137183,
            0.04089132, 0.04039299, 0.03993977, 0.03948541, 0.03905235,
            0.03864758, 0.03821014, 0.03778766, 0.03740032, 0.03700535,
            0.03662915, 0.03621143, 0.03587692, 0.03551261, 0.03515779,
            0.03480729, 0.0344525, 0.03412606, 0.03378796, 0.03347909,
            0.03316675, 0.03283485, 0.03254751, 0.03223662, 0.03194602,
            0.03166242, 0.03135705, 0.03110392, 0.03082337, 0.0305658,
            0.03030111, 0.03003478, 0.02977189, 0.02952609, 0.0292679,
            0.0290245, 0.02879608, 0.02856074, 0.0283379, 0.02809466,
            0.02789209, 0.02765184, 0.02744155, 0.02724763, 0.02702829,
            0.02681467, 0.02660705, 0.02642264, 0.02621097, 0.02599498,
            0.02583466, 0.02563592, 0.02545643, 0.0252755, 0.0250976,
            0.02490806, 0.0247351, 0.02457273, 0.02438707, 0.02422045,
            0.02406481, 0.02388955, 0.02374186, 0.02357876, 0.02341766,
            0.02326615, 0.02311393, 0.02295128, 0.02281399, 0.02266768,
            0.0225129, 0.02237694, 0.02223218, 0.02209592, 0.0219511,
            0.02182045, 0.02169247, 0.02154614, 0.02141682, 0.02128498,
            0.02116473
        ])
        return np.argmin(np.abs(self.delta_r - radius * unit_spacing))

    # it appears that midpoint rule works better than Gregory
    def create_node_set(self, quadrature_order=1):
        # read in md node set on shell
        seed = 1
        num_points_per_shell = (self.md_degree + 1)**2
        shell_nodes = np.loadtxt(self.node_set_dir + "/md" +
                                 str(self.md_degree).zfill(3) + "." +
                                 str(num_points_per_shell).zfill(5) + ".txt")

        average_spacing = self.get_average_spacing(shell_nodes)
        match_radius = self.delta_r / average_spacing
        r_ecs = self.r_max - self.ecs_size
        box_radius = self.delta_r * np.ceil(
            match_radius / self.delta_r) + self.delta_r / 2.0
        if not self.quiet:
            print "Average spacing on unit sphere:", average_spacing
            print "Match radius:", match_radius

        if r_ecs < match_radius:
            exit(
                "ERROR: the ECS is to close to the center. r_ecs < match_radius"
            )
        self.node_set = []
        self.node_set_ecs = []

        # this it for the boundary nodes later
        # and the shell nodes
        rotation_matrix = self.rand_rotation_matrix()
        current_shell = None
        num_r_points = np.arange(self.delta_r / 2.0, self.r_max,
                                 self.delta_r).shape[0]

        quadrature_weights = self.gregory_weights(num_r_points,
                                                  quadrature_order)
        for idx, r in enumerate(
                np.arange(self.delta_r / 2.0, self.r_max, self.delta_r)):
            if r >= r_ecs:
                r_outer = r - r_ecs
                current_shell = shell_nodes * r
                current_shell_ecs = shell_nodes * (
                    r_ecs + r_outer * np.exp(1.0j * np.pi / 4.0))
            if r >= match_radius:  # we have reach max md order value
                current_shell = shell_nodes * r
                current_shell_ecs = shell_nodes * r
            else:  # we are growing in md order
                current_md_degree = self.get_md_degree(r)
                num_points_per_shell = (current_md_degree + 1)**2
                current_shell = r * np.loadtxt(
                    self.node_set_dir + "/md" + str(current_md_degree).zfill(3)
                    + "." + str(num_points_per_shell).zfill(5) + ".txt")
                current_shell_ecs = r * np.loadtxt(
                    self.node_set_dir + "/md" + str(current_md_degree).zfill(3)
                    + "." + str(num_points_per_shell).zfill(5) + ".txt")
            # get random numbers
            random_numbers, seed = sobol_seq.i4_sobol(3, seed)
            # rotate shells to sample theta and phi evenly
            rotation_matrix = self.rand_rotation_matrix(
                randnums=random_numbers)
            current_shell[:, :3] = np.transpose(
                rotation_matrix.dot(np.transpose(current_shell[:, :3])))
            current_shell_ecs[:, :3] = np.transpose(
                rotation_matrix.dot(np.transpose(current_shell_ecs[:, :3])))
            # update integration weights with distance between shells
            # (we are using Gregory rule in r since r_min = delta_r/2)
            current_shell[:, 3] *= quadrature_weights[idx]
            self.node_set.append(current_shell)
            self.node_set_ecs.append(
                np.array(current_shell_ecs, dtype='complex'))
        self.node_set = np.concatenate(self.node_set)
        self.node_set_ecs = np.concatenate(self.node_set_ecs)

        boundary_degree = min(165, int(self.md_degree * 1.5))
        num_points_boundary_shell = (boundary_degree + 1)**2
        boundary_shell_nodes = np.loadtxt(
            self.node_set_dir + "/md" + str(boundary_degree).zfill(3) + "." +
            str(num_points_boundary_shell).zfill(5) + ".txt")
        r = self.r_max + self.delta_r / 2.0
        r_outer = r - r_ecs
        current_shell = boundary_shell_nodes * r
        current_shell_ecs = boundary_shell_nodes * (
            r_ecs + r_outer * np.exp(1.0j * np.pi / 4.0))
        # rotate boundary nodes the same as previous shell
        current_shell[:, :3] = np.transpose(
            rotation_matrix.dot(np.transpose(current_shell[:, :3])))
        current_shell_ecs[:, :3] = np.transpose(
            rotation_matrix.dot(np.transpose(current_shell_ecs[:, :3])))
        # trim weights as they are not needed for the boundary
        self.boundary_nodes = current_shell[:, :3]
        self.boundary_nodes_ecs = current_shell_ecs[:, :3]
        # clean up arrays for use in the rest of the code
        self.weights = self.node_set[:, 3]
        self.node_set = self.node_set[:, :3]
        self.node_set_ecs = self.node_set_ecs[:, :3]

    def get_nearest_neighbors(self):
        tree = KDTree(self.full_node_set)
        nearest_dist, nearest_idx = tree.query(
            self.full_node_set, k=self.stencil_size)
        return nearest_dist[:self.node_set.shape[
            0]], nearest_idx[:self.node_set.shape[0]]

    def get_nearest_neighbors_from_set(self, node_set, k):
        tree = KDTree(node_set)
        return tree.query(node_set, k=k)

    def rbf_phi(self, radius):
        return radius**self.rbf_order

    def get_row_and_col(self):
        # allocate space
        row_col_set = np.zeros(
            (self.laplace_weights.shape[0] * self.laplace_weights.shape[1], 2))
        current_idx = 0
        for row_idx in self.nearest_idx:
            # fill row index (first entry returned by kdTree)
            row_col_set[current_idx:current_idx + row_idx.shape[0],
                        0] = row_idx[0]
            # fill col index (the array returned by kdTree)
            row_col_set[current_idx:current_idx + row_idx.shape[0],
                        1] = row_idx
            # update index
            current_idx += row_idx.shape[0]
        return row_col_set.transpose()

    def create_operator_matrix(self, weights):
        weights = weights.reshape((weights.shape[0] * weights.shape[1], ))
        row_col = self.get_row_and_col()
        row_less_than = row_col[0] < self.node_set.shape[0]
        weights = weights[row_less_than]
        row_col = row_col[:, row_less_than]
        col_less_than = row_col[1] < self.node_set.shape[0]
        weights = weights[col_less_than]
        row_col = row_col[:, col_less_than]
        matrix = csr_matrix(
            (weights, row_col),
            shape=(self.node_set.shape[0], self.node_set.shape[0]))
        return matrix

    def calculate_operator_weights(self):
        if not self.quiet:
            print "Calculating operator weights"
        self.laplace_weights = np.zeros(self.nearest_idx.shape)
        self.laplace_weights_ecs = np.zeros(
            self.nearest_idx.shape, dtype='complex')
        self.first_derivative_weights = np.zeros(
            [self.nearest_idx.shape[0], self.nearest_idx.shape[1], 3])

        num_poly_terms = comb(self.poly_order + 3, 3, exact=True)

        for idx, node_list_idx in enumerate(self.nearest_idx):
            if (idx + 1) % int(self.nearest_idx.shape[0] / 10) == 0:
                if not self.quiet:
                    print ".",
                    sys.stdout.flush()
            # shift nodes to origin
            node_list = self.full_node_set[node_list_idx]
            node_list_shifted = node_list - node_list[0]

            node_list_ecs = self.full_node_set_ecs[node_list_idx]
            node_list_shifted_ecs = node_list_ecs - node_list_ecs[0]

            cur_radius = np.sqrt((node_list[0]**2).sum())
            # makes exp(-r) = 1e-10
            exp_cut_radius = 23.0258509299
            # makes exp(-r) = 1e-7
            # exp_cut_radius = 16.118095651

            if cur_radius < exp_cut_radius and self.exp_order != 0:
                matrix_size = self.stencil_size + num_poly_terms + 1
            else:
                matrix_size = self.stencil_size + num_poly_terms

            # create A matrix and right hand side
            A_matrix = np.zeros((matrix_size, matrix_size))
            laplace_rhs = np.zeros((matrix_size))
            first_derivative_rhs = np.zeros((matrix_size, 3))

            # fill normal A matrix
            for node_idx, row_node in enumerate(node_list_shifted):
                # A_matrix
                A_matrix[node_idx, :self.stencil_size] = self.rbf_phi(
                    np.sqrt(((node_list_shifted - row_node)**2).sum(axis=1)))

                # save computational time
                derivative_term_rbf = np.sqrt(
                    ((row_node)**2).sum())**(self.rbf_order - 2)
                # laplacian right hand side
                laplace_rhs[node_idx] = self.rbf_order * (
                    self.rbf_order + 3.0 - 2.0) * derivative_term_rbf
                # first_deriviative right hand side
                first_derivative_rhs[node_idx, 0] = self.rbf_order * (
                    -row_node[0]) * derivative_term_rbf
                first_derivative_rhs[node_idx, 1] = self.rbf_order * (
                    -row_node[1]) * derivative_term_rbf
                first_derivative_rhs[node_idx, 2] = self.rbf_order * (
                    -row_node[2]) * derivative_term_rbf

            # add poly
            for poly_idx in np.arange(num_poly_terms):
                if poly_idx == 4 or poly_idx == 6 or poly_idx == 9:
                    laplace_rhs[self.stencil_size + poly_idx] = 2.0
                if poly_idx == 1:
                    first_derivative_rhs[self.stencil_size + poly_idx, 0] = 1.0
                if poly_idx == 2:
                    first_derivative_rhs[self.stencil_size + poly_idx, 1] = 1.0
                if poly_idx == 3:
                    first_derivative_rhs[self.stencil_size + poly_idx, 2] = 1.0
            A_matrix[:self.stencil_size, self.stencil_size:
                     self.stencil_size + num_poly_terms] = self.get_poly_terms(
                         node_list_shifted, self.poly_order)
            A_matrix[self.stencil_size:self.stencil_size + num_poly_terms, :
                     self.stencil_size] = A_matrix[:self.stencil_size,
                                                   self.stencil_size:
                                                   self.stencil_size +
                                                   num_poly_terms].transpose()

            # add exp (use unshifted distance)
            # only add one exp since our nodes are at very few values of r
            # this leads to exp(-r/a) and exp(-r/(a+1)) being nearly identical
            # for many node sets
            if cur_radius < exp_cut_radius and self.exp_order != 0:
                a = 1.0 / float(self.exp_order)
                node_list_radius = np.sqrt(((node_list)**2).sum(axis=1))
                laplace_rhs[matrix_size - 1] = a * a * np.exp(
                    -a * node_list_radius[0]) - 2.0 * a * np.exp(
                        -a * node_list_radius[0]) / node_list_radius[0]

                first_derivative_rhs[
                    matrix_size - 1, 0] = a * node_list[0, 0] * np.exp(
                        -a * node_list_radius[0]) / node_list_radius[0]
                first_derivative_rhs[
                    matrix_size - 1, 1] = a * node_list[0, 1] * np.exp(
                        -a * node_list_radius[0]) / node_list_radius[0]
                first_derivative_rhs[
                    matrix_size - 1, 2] = a * node_list[0, 2] * np.exp(
                        -a * node_list_radius[0]) / node_list_radius[0]

                exp_row = np.exp(-a * node_list_radius)
                A_matrix[:self.stencil_size, matrix_size - 1] = exp_row
                A_matrix[matrix_size - 1, :self.stencil_size] = exp_row

            try:
                weights = np.linalg.solve(A_matrix,
                                          np.concatenate(
                                              (laplace_rhs.reshape(
                                                  (laplace_rhs.shape[0], 1)),
                                               first_derivative_rhs),
                                              axis=1))
            except:
                print "Error in calculating weights"
                print cur_radius
                print matrix_size
                print self.stencil_size + num_poly_terms + self.exp_order
                print self.stencil_size + num_poly_terms
                print A_matrix
                exit()
            self.laplace_weights[idx, :] = weights[:self.stencil_size, 0]
            self.first_derivative_weights[idx, :,
                                          0] = weights[:self.stencil_size, 1]
            self.first_derivative_weights[idx, :,
                                          1] = weights[:self.stencil_size, 2]
            self.first_derivative_weights[idx, :,
                                          2] = weights[:self.stencil_size, 3]

            # now for the ECS version
            A_matrix_ecs = np.zeros(
                (matrix_size, matrix_size), dtype='complex')
            laplace_ecs_rhs = np.zeros((matrix_size), dtype='complex')

            # fill normal A matrix
            for node_idx, row_node in enumerate(node_list_shifted_ecs):
                # A_matrix
                A_matrix_ecs[node_idx, :self.stencil_size] = self.rbf_phi(
                    np.sqrt(((node_list_shifted_ecs - row_node)**2).sum(
                        axis=1)))

                # save computational time
                derivative_term_rbf = np.sqrt(
                    ((row_node)**2).sum())**(self.rbf_order - 2)
                # laplacian right hand side
                laplace_ecs_rhs[node_idx] = self.rbf_order * (
                    self.rbf_order + 3.0 - 2.0) * derivative_term_rbf

            # add poly
            for poly_idx in np.arange(num_poly_terms):
                if poly_idx == 4 or poly_idx == 6 or poly_idx == 9:
                    laplace_ecs_rhs[self.stencil_size + poly_idx] = 2.0
            A_matrix_ecs[:self.stencil_size, self.stencil_size:self.
                         stencil_size + num_poly_terms] = self.get_poly_terms(
                             node_list_shifted_ecs,
                             self.poly_order,
                             dtype='complex')
            A_matrix_ecs[
                self.stencil_size:self.stencil_size + num_poly_terms, :self.
                stencil_size] = A_matrix_ecs[:self.stencil_size, self.
                                             stencil_size:self.stencil_size +
                                             num_poly_terms].transpose()

            # add exp (use unshifted distance)
            # only add one exp since our nodes are at very few values of r
            # this leads to exp(-r/a) and exp(-r/(a+1)) being nearly identical
            # for many node sets
            if cur_radius < exp_cut_radius and self.exp_order != 0:
                a = 1.0 / float(self.exp_order)
                node_list_radius = np.sqrt(((node_list_ecs)**2).sum(axis=1))
                laplace_ecs_rhs[matrix_size - 1] = a * a * np.exp(
                    -a * node_list_radius[0]) - 2.0 * a * np.exp(
                        -a * node_list_radius[0]) / node_list_radius[0]

                exp_row = np.exp(-a * node_list_radius)
                A_matrix_ecs[:self.stencil_size, matrix_size - 1] = exp_row
                A_matrix_ecs[matrix_size - 1, :self.stencil_size] = exp_row

            try:
                weights = np.linalg.solve(A_matrix_ecs, laplace_ecs_rhs)
            except:
                print "Error in calculating weights ECS"
                print cur_radius
                print matrix_size
                print self.stencil_size + num_poly_terms + self.exp_order
                print self.stencil_size + num_poly_terms
                print A_matrix
                exit()
            self.laplace_weights_ecs[idx, :] = weights[:self.stencil_size]
        if not self.quiet:
            print
        self.laplace = self.create_operator_matrix(self.laplace_weights)
        self.laplace_ecs = self.create_operator_matrix(
            self.laplace_weights_ecs)
        self.first_deriviative = []
        self.first_deriviative.append(
            self.create_operator_matrix(self.first_derivative_weights[:, :,
                                                                      0]))
        self.first_deriviative.append(
            self.create_operator_matrix(self.first_derivative_weights[:, :,
                                                                      1]))
        self.first_deriviative.append(
            self.create_operator_matrix(self.first_derivative_weights[:, :,
                                                                      2]))

    def save_node_set(self):
        row_col = self.laplace.nonzero()
        sparserows = row_col[0]
        sparsecols = row_col[1]

        if not self.quiet:
            print "Saving operators"

        file = h5py.File("nodes.h5", "w")
        # write node set
        grp_node_set = file.create_group("node_set")
        dset = file.create_dataset(
            "node_set/node_idx", (self.node_set.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = np.arange(self.node_set.shape[0])
        dset = file.create_dataset(
            "node_set/x", (self.node_set.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.node_set[:, 0]
        dset = file.create_dataset(
            "node_set/y", (self.node_set.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.node_set[:, 1]
        dset = file.create_dataset(
            "node_set/z", (self.node_set.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.node_set[:, 2]
        dset = file.create_dataset(
            "node_set/weights", (self.node_set.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.weights

        grp_boundary_nodes = file.create_group("boundary_nodes")
        dset = file.create_dataset(
            "boundary_nodes/node_idx", (self.boundary_nodes.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = np.arange(self.boundary_nodes.shape[0])
        dset = file.create_dataset(
            "boundary_nodes/x", (self.boundary_nodes.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.boundary_nodes[:, 0]
        dset = file.create_dataset(
            "boundary_nodes/y", (self.boundary_nodes.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.boundary_nodes[:, 1]
        dset = file.create_dataset(
            "boundary_nodes/z", (self.boundary_nodes.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.boundary_nodes[:, 2]

        grp_operators = file.create_group("operators")
        dset = file.create_dataset(
            "operators/row_idx", (sparserows.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = sparserows
        dset = file.create_dataset(
            "operators/col_idx", (sparserows.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = sparsecols
        dset = file.create_dataset(
            "operators/laplace", (sparserows.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.laplace[sparserows, sparsecols]
        dset = file.create_dataset(
            "operators/laplace_ecs", (sparserows.shape[0], 2),
            dtype='float64',
            chunks=True)
        real_data = np.array(
            self.laplace_ecs[sparserows, sparsecols].real).reshape(
                (sparserows.shape[0], ))
        imag_data = np.array(
            self.laplace_ecs[sparserows, sparsecols].imag).reshape(
                (sparserows.shape[0], ))
        dset[:] = np.array(zip(real_data, imag_data))
        dset = file.create_dataset(
            "operators/dx", (sparserows.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.first_deriviative[0][sparserows, sparsecols]
        dset = file.create_dataset(
            "operators/dy", (sparserows.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.first_deriviative[1][sparserows, sparsecols]
        dset = file.create_dataset(
            "operators/dz", (sparserows.shape[0], ),
            dtype='float64',
            chunks=True)
        dset[:] = self.first_deriviative[2][sparserows, sparsecols]

        grp_parameters = file.create_group("parameters")
        dset = file.create_dataset(
            "parameters/delta_r", (1, ), dtype='float64', chunks=True)
        dset[:] = self.delta_r
        dset = file.create_dataset(
            "parameters/r_max", (1, ), dtype='float64', chunks=True)
        dset[:] = self.r_max
        dset = file.create_dataset(
            "parameters/ecs_size", (1, ), dtype='float64', chunks=True)
        dset[:] = self.ecs_size
        dset = file.create_dataset(
            "parameters/md_degree", (1, ), dtype='float64', chunks=True)
        dset[:] = self.md_degree
        dset = file.create_dataset(
            "parameters/rbf_order", (1, ), dtype='float64', chunks=True)
        dset[:] = self.rbf_order
        dset = file.create_dataset(
            "parameters/poly_order", (1, ), dtype='float64', chunks=True)
        dset[:] = self.poly_order
        dset = file.create_dataset(
            "parameters/exp_order", (1, ), dtype='float64', chunks=True)
        dset[:] = self.exp_order
        dset = file.create_dataset(
            "parameters/stencil_size", (1, ), dtype='float64', chunks=True)
        dset[:] = self.stencil_size

    def check_quality(self):
        print "Calculating Quality of Node Set"
        max_x = np.zeros(self.num_nodes)
        max_y = np.zeros(self.num_nodes)
        max_z = np.zeros(self.num_nodes)
        min_x = np.zeros(self.num_nodes)
        min_y = np.zeros(self.num_nodes)
        min_z = np.zeros(self.num_nodes)
        z_offset = np.zeros(self.num_nodes)
        for idx, node_list_idx in enumerate(self.nearest_idx):
            node_list = np.transpose(
                self.rotation_matrix_aline_z(self.theta[idx], self.phi[idx])
                .dot(np.transpose(self.full_node_set[node_list_idx])))
            max_x[idx] = np.max(node_list[:, 0])
            min_x[idx] = np.min(node_list[:, 0])
            max_y[idx] = np.max(node_list[:, 1])
            min_y[idx] = np.min(node_list[:, 1])
            max_z[idx] = np.max(node_list[:, 2])
            min_z[idx] = np.min(node_list[:, 2])
            z_offset[idx] = node_list[0, 2]

        x_span = max_x - min_x
        y_span = max_y - min_y
        z_span = max_z - min_z

        x_asm = np.abs(max_x + min_x)
        y_asm = np.abs(max_y + min_y)
        z_asm = np.abs(max_z + min_z) - z_offset  # make z centered about 0

        x_asm_ratio = x_asm / x_span
        y_asm_ratio = y_asm / y_span
        z_asm_ratio = z_asm / z_span

        xy_ratio = x_span / y_span
        xz_ratio = x_span / z_span
        yz_ratio = y_span / z_span

        average_span = (x_span + y_span + z_span) / 3.0

        x_avg_ratio = x_span / average_span
        y_avg_ratio = y_span / average_span
        z_avg_ratio = z_span / average_span

        print
        print "max ratio (xy, xz, yz):", xy_ratio.max(), xz_ratio.max(
        ), yz_ratio.max()
        print "min ratio (xy, xz, yz):", xy_ratio.min(), xz_ratio.min(
        ), yz_ratio.min()
        print "mean ratio (xy, xz, yz):", xy_ratio.mean(), xz_ratio.mean(
        ), yz_ratio.mean()
        print "radius of max ratio (xy, xz, yz):", np.sqrt(
            (self.node_set[xy_ratio.argmax()]**2).sum()), np.sqrt(
                (self.node_set[xz_ratio.argmax()]**2).sum()), np.sqrt(
                    (self.node_set[yz_ratio.argmax()]**2).sum())
        print "radius of min ratio (xy, xz, yz):", np.sqrt(
            (self.node_set[xy_ratio.argmin()]**2).sum()), np.sqrt(
                (self.node_set[xz_ratio.argmin()]**2).sum()), np.sqrt(
                    (self.node_set[yz_ratio.argmin()]**2).sum())

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # # ax = fig.add_subplot(111, projection='3d')
        # xy_ratio_idx = xy_ratio.argmax()
        # ax.scatter(self.node_set[self.nearest_idx[xy_ratio_idx]][:, 0],
        #            self.node_set[self.nearest_idx[xy_ratio_idx]][:, 1])
        # plt.axis('equal')
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(self.node_set[self.nearest_idx[xy_ratio_idx]][:, 0],
        #            self.node_set[self.nearest_idx[xy_ratio_idx]][:, 2])
        # plt.axis('equal')
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(self.node_set[self.nearest_idx[xy_ratio_idx]][:, 1],
        #            self.node_set[self.nearest_idx[xy_ratio_idx]][:, 2])
        # plt.axis('equal')
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # xz_ratio_idx = xz_ratio.argmax()
        # ax.scatter(self.node_set[self.nearest_idx[xz_ratio_idx]][:, 0],
        #            self.node_set[self.nearest_idx[xz_ratio_idx]][:, 1])
        # plt.axis('equal')
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(self.node_set[self.nearest_idx[xz_ratio_idx]][:, 0],
        #            self.node_set[self.nearest_idx[xz_ratio_idx]][:, 2])
        # plt.axis('equal')
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(self.node_set[self.nearest_idx[xz_ratio_idx]][:, 1],
        #            self.node_set[self.nearest_idx[xz_ratio_idx]][:, 2])
        # plt.axis('equal')
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # yz_ratio_idx = yz_ratio.argmax()
        # ax.scatter(self.node_set[self.nearest_idx[yz_ratio_idx]][:, 1],
        #            self.node_set[self.nearest_idx[yz_ratio_idx]][:, 2])
        # plt.axis('equal')
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        # xy_ratio_idx = xy_ratio.argmin()
        # ax.scatter(self.node_set[self.nearest_idx[xy_ratio_idx]][:, 0],
        #            self.node_set[self.nearest_idx[xy_ratio_idx]][:, 1])
        # plt.axis('equal')
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # xz_ratio_idx = xz_ratio.argmin()
        # ax.scatter(self.node_set[self.nearest_idx[xz_ratio_idx]][:, 0],
        #            self.node_set[self.nearest_idx[xz_ratio_idx]][:, 2])
        # plt.axis('equal')
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # yz_ratio_idx = yz_ratio.argmin()
        # ax.scatter(self.node_set[self.nearest_idx[yz_ratio_idx]][:, 1],
        #            self.node_set[self.nearest_idx[yz_ratio_idx]][:, 2])
        # plt.axis('equal')
        # plt.show()

        print
        print "max ratio from average span (x, y, z):", x_avg_ratio.max(
        ), y_avg_ratio.max(), z_avg_ratio.max()
        print "min ratio from average span (x, y, z):", x_avg_ratio.min(
        ), y_avg_ratio.min(), z_avg_ratio.min()
        print "mean ratio from average span (x, y, z):", x_avg_ratio.mean(
        ), y_avg_ratio.mean(), z_avg_ratio.mean()
        print "radius of max ratio from average span (x, y, z):", np.sqrt(
            (self.node_set[x_avg_ratio.argmax()]**2).sum()), np.sqrt(
                (self.node_set[z_avg_ratio.argmax()]**2).sum()), np.sqrt(
                    (self.node_set[z_avg_ratio.argmax()]**2).sum())
        print "radius of min ratio from average span (x, y, z):", np.sqrt(
            (self.node_set[x_avg_ratio.argmin()]**2).sum()), np.sqrt(
                (self.node_set[y_avg_ratio.argmin()]**2).sum()), np.sqrt(
                    (self.node_set[z_avg_ratio.argmin()]**2).sum())

        print
        print "max span (x, y, z):", x_span.max(), y_span.max(), z_span.max()
        print "min span (x, y, z):", x_span.min(), y_span.min(), z_span.min()
        print "mean span (x, y, z):", x_span.mean(), y_span.mean(
        ), z_span.mean()
        print "radius of max span (x, y, z)", np.sqrt(
            (self.node_set[x_span.argmax()]**2).sum()), np.sqrt(
                (self.node_set[y_span.argmax()]**2).sum()), np.sqrt(
                    (self.node_set[z_span.argmax()]**2).sum())
        print "radius of min span (x, y, z)", np.sqrt(
            (self.node_set[x_span.argmin()]**2).sum()), np.sqrt(
                (self.node_set[y_span.argmin()]**2).sum()), np.sqrt(
                    (self.node_set[z_span.argmin()]**2).sum())

        print
        print "max asymmetry (x, y, z):", x_asm_ratio.max(), y_asm_ratio.max(
        ), z_asm_ratio.max()
        print "min asymmetry (x, y, z):", x_asm_ratio.min(), y_asm_ratio.min(
        ), z_asm_ratio.min()
        print "mean asymmetry (x, y, z):", x_asm_ratio.mean(
        ), y_asm_ratio.mean(), z_asm_ratio.mean()
        print "radius of max asymmetry ratio (x, y, z)", np.sqrt(
            (self.node_set[x_asm_ratio.argmax()]**2).sum()), np.sqrt(
                (self.node_set[y_asm_ratio.argmax()]**2).sum()), np.sqrt(
                    (self.node_set[z_asm_ratio.argmax()]**2).sum())
        print "radius of min asymmetry ratio (x, y, z)", np.sqrt(
            (self.node_set[x_asm_ratio.argmin()]**2).sum()), np.sqrt(
                (self.node_set[y_asm_ratio.argmin()]**2).sum()), np.sqrt(
                    (self.node_set[z_asm_ratio.argmin()]**2).sum())
