import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import comb
from scipy.sparse import csr_matrix


class Node_Set:
    def __init__(self,
                 r_max=100,
                 delta_r=0.1,
                 stencil_size=60,
                 md_degree=6,
                 rbf_order=7,
                 poly_order=5,
                 exp_order=3,
                 node_set_dir="/Users/jvenzke/Repos/RBF_mesh/MD_node_sets"):
        self.r_max = r_max
        self.delta_r = delta_r
        self.md_degree = md_degree
        self.stencil_size = stencil_size
        self.rbf_order = rbf_order
        self.poly_order = poly_order
        self.exp_order = exp_order
        self.node_set_dir = node_set_dir

        if self.rbf_order < 3 or self.rbf_order % 2 != 1:
            exit("RBF order must be >3 and odd")
        print "Creating Node Set"
        self.node_set, self.weights = self.create_node_set()
        self.radius = np.sqrt((self.node_set**2).sum(axis=1))
        self.theta = np.angle(self.node_set[:, 2] + 1.0j * np.sqrt(
            (self.node_set[:, :2]**2).sum(axis=1)))
        self.phi = np.angle(self.node_set[:, 0] + 1.0j * self.node_set[:, 1])
        self.num_nodes = self.node_set.shape[0]
        print "Number of nodes:", self.num_nodes
        print "Getting Nearest Neighbors"
        self.nearest_dist, self.nearest_idx = self.get_nearest_neighbors()

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

    def get_poly_terms(self, nodes, poly_order):
        current_dim = nodes.shape[1]

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
            current_dim_poly = self.get_poly_terms(nodes[:, -1].reshape(
                (nodes.shape[0], 1)), poly_order)
            # get the array [1, x, y, ..., x^2, xy, y^2, ..., z^2, ..., x^(poly_order)]
            lower_dim_poly = self.get_poly_terms(nodes[:, :-1], poly_order)
            # number of poly terms in this dimension
            num_poly_terms = comb(
                poly_order + current_dim, current_dim, exact=True)
            # allocate array
            poly_set = np.zeros((nodes.shape[0], num_poly_terms))
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

    def create_node_set(self):
        # read in md node set on shell
        num_points_per_shell = (self.md_degree + 1)**2
        shell_nodes = np.loadtxt(self.node_set_dir + "/md" +
                                 str(self.md_degree).zfill(3) + "." +
                                 str(num_points_per_shell).zfill(5) + ".txt")

        average_spacing = self.get_average_spacing(shell_nodes)
        match_radius = self.delta_r / average_spacing
        print "Average spacing on unit sphere:", average_spacing
        print "Match radius:", match_radius, self.get_average_spacing(
            shell_nodes * match_radius)
        self.node_set = []
        for r in np.arange(self.delta_r / 2.0, self.r_max + self.delta_r / 2.0,
                           self.delta_r):
            if r > match_radius:
                current_shell = shell_nodes * r
                rotation_matrix = self.rand_rotation_matrix()
                current_shell[:, :3] = np.transpose(
                    rotation_matrix.dot(np.transpose(current_shell[:, :3])))
                #TODO: handle weights (account for radius (r integration) and expanding surface area)
                self.node_set.append(current_shell)
        self.node_set = np.concatenate(self.node_set)
        return self.node_set[:, :3], self.node_set[:, 3]

    def get_nearest_neighbors(self):
        tree = KDTree(self.node_set)
        return tree.query(self.node_set, k=self.stencil_size)

    def get_nearest_neighbors_from_set(self, node_set, k):
        tree = KDTree(node_set)
        return tree.query(node_set, k=k)

    def rbf_phi(self, radius):
        return radius**self.rbf_order

    def get_row_and_col(self):
        row_col_set = np.zeros(
            (self.laplace_weights.shape[0] * self.laplace_weights.shape[1], 2))
        current_idx = 0
        for row_idx in self.nearest_idx:
            row_col_set[current_idx:current_idx + row_idx.shape[0],
                        0] = row_idx[0]
            row_col_set[current_idx:current_idx + row_idx.shape[0],
                        1] = row_idx
            current_idx += row_idx.shape[0]
        return row_col_set.transpose()

    def calculate_opperator_weights(self):
        print "Calculating operator weights"
        self.laplace_weights = np.zeros(self.nearest_idx.shape)

        num_poly_terms = comb(self.poly_order + 3, 3, exact=True)

        for idx, node_list_idx in enumerate(self.nearest_idx):
            if idx % int(self.nearest_idx.shape[0] / 10) == 0:
                print ".",
            # shift nodes to origin
            node_list = self.node_set[node_list_idx]
            node_list_shifted = node_list - node_list[0]

            matrix_size = self.stencil_size + num_poly_terms + self.exp_order

            # create A matrix and right hand side
            A_matrix = np.zeros((matrix_size, matrix_size))
            right_hand_side = np.zeros((matrix_size))

            # fill normal A matrix
            for node_idx, row_node in enumerate(node_list_shifted):
                A_matrix[node_idx, :self.stencil_size] = self.rbf_phi(
                    np.sqrt(((node_list_shifted - row_node)**2).sum(axis=1)))
                right_hand_side[node_idx] = self.rbf_order * (
                    self.rbf_order + 3.0 - 2.0) * np.sqrt(
                        ((row_node)**2).sum())**(self.rbf_order - 2)

            # add poly
            for poly_idx in np.arange(num_poly_terms):
                if poly_idx == 4 or poly_idx == 6 or poly_idx == 9:
                    right_hand_side[self.stencil_size + poly_idx] = 2.0
            A_matrix[:self.stencil_size, self.stencil_size:
                     self.stencil_size + num_poly_terms] = self.get_poly_terms(
                         node_list_shifted, self.poly_order)
            A_matrix[self.stencil_size:self.stencil_size + num_poly_terms, :
                     self.stencil_size] = A_matrix[:self.stencil_size,
                                                   self.stencil_size:
                                                   self.stencil_size +
                                                   num_poly_terms].transpose()

            # add exp (use unshifted distance)
            for a in np.arange(1, self.exp_order + 1):
                node_list_radius = np.sqrt(((node_list)**2).sum(axis=1))
                right_hand_side[matrix_size
                                - self.exp_order - 1 + a] = a * a * np.exp(
                                    -a * node_list_radius[0]
                                ) - 2.0 * a / node_list_radius[0]
                exp_row = np.exp(-a * node_list_radius)
                A_matrix[:self.stencil_size, matrix_size - self.exp_order - 1 +
                         a] = exp_row
                A_matrix[matrix_size - self.exp_order - 1 + a, :
                         self.stencil_size] = exp_row

            # solve and store node weights (drop poly and exp terms)
            weights = np.linalg.solve(A_matrix, right_hand_side)
            self.laplace_weights[idx, :] = weights[:self.stencil_size]
        print
        self.laplace = csr_matrix(
            (self.laplace_weights.reshape((self.laplace_weights.shape[0] *
                                           self.laplace_weights.shape[1], )),
             self.get_row_and_col()),
            shape=(self.node_set.shape[0], self.node_set.shape[0]))
        print self.laplace

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
                .dot(np.transpose(self.node_set[node_list_idx])))
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
