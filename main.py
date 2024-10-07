import numpy as np
import scipy.sparse as sps
import time

import pygeon as pg
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.linear_system import create_restriction
from pygeon.numerics.innerproducts import mass_matrix
from pygeon.numerics.stiffness import stiff_matrix


class Poincare:
    def __init__(self, mdg: pg.MixedDimensionalGrid, create_ops=True):
        """
        Class for generating Poincaré operators p
        that satisfy pd + dp = I
        with d the exterior derivative

        Args:
            mdg (pg.MixedDimensionalGrid): a (mixed-dimensional) grid
        """
        self.mdg = mdg
        self.dim = mdg.dim_max()
        self.define_bar_spaces()

        if create_ops:
            self.create_operators()

    def euler_characteristic(self):
        sd = self.mdg.subdomains()[0]

        match self.dim:
            case 1:
                return sd.num_nodes - sd.num_cells
            case 2:
                return sd.num_nodes - sd.num_faces + sd.num_cells
            case 3:
                return sd.num_nodes - sd.num_ridges + sd.num_faces - sd.num_cells

    def define_bar_spaces(self):
        """
        Flag the mesh entities that will be used to generate the Poincaré operators
        """
        # Preallocation
        self.bar_spaces = [None] * (self.dim + 1)

        # Cells
        self.bar_spaces[self.dim] = np.zeros(self.mdg.num_subdomain_cells(), dtype=bool)

        # Faces
        self.bar_spaces[self.dim - 1] = pg.SpanningTree(
            self.mdg, "all_bdry"
        ).flagged_faces

        # Edges in 3D
        if self.dim == 3:
            self.bar_spaces[1] = self.flag_edges()

        # Nodes
        self.bar_spaces[0] = self.flag_nodes()

    def flag_edges(self):
        """
        Flag the edges of the grid that form a spanning tree of the nodes

        Returns:
            np.ndarray: boolean array with flagged edges
        """

        grad = pg.grad(self.mdg)
        incidence = grad.T @ grad

        root = self.find_central_node()

        tree = sps.csgraph.breadth_first_tree(incidence, root, directed=False)
        c_start, c_end, _ = sps.find(tree)

        rows = np.hstack((c_start, c_end))
        cols = np.hstack([np.arange(c_start.size)] * 2)
        vals = np.ones_like(rows)

        edge_finder = sps.csc_array(
            (vals, (rows, cols)), shape=(grad.shape[1], tree.nnz)
        )
        edge_finder = np.abs(grad) @ edge_finder
        I, _, V = sps.find(edge_finder)
        tree_edges = I[V == 2]

        flagged_edges = np.ones(grad.shape[0], dtype=bool)
        flagged_edges[tree_edges] = False

        return flagged_edges

    def find_central_node(self):

        sd = self.mdg.subdomains()[0]
        center = np.mean(sd.nodes, axis=1, keepdims=True)
        dists = np.linalg.norm(sd.nodes - center, axis=0)

        return np.argmin(dists)

    def flag_nodes(self):
        """
        Flag all the nodes except for the first one

        Returns:
            np.ndarray: boolean array with flagged nodes
        """
        num_nodes = self.mdg.subdomains()[0].num_nodes
        flagged_nodes = np.ones(num_nodes, dtype=bool)
        flagged_nodes[0] = False

        return flagged_nodes

    def create_operators(self):
        """
        Saves the poincare operators in self.op
        """
        # Preallocation
        n = self.dim
        self.op = [None] * (n + 1)

        # Cells to faces, similar to the SpanningTree
        if n > 1:
            self.op[n] = self.create_op(n)

        # Faces to edges (only for 3D)
        if n > 2:
            self.op[2] = self.create_op(2)

        # Edges to nodes

        op_nodal = self.create_op(1)
        subtract_mean = lambda x: x - np.mean(x)
        self.op[1] = lambda f: subtract_mean(op_nodal(f))

        # Nodes to the constants
        self.op[0] = lambda f: np.full_like(f, np.mean(f))

    def create_op(self, k):
        """
        Create the Poincaré operator for k-forms

        Args:
            k (int): order of the form

        Returns:
            function: the Poincaré operator
        """
        n_minus_k = self.dim - k
        _diff = diff(self.mdg, n_minus_k + 1)

        R_0 = create_restriction(~self.bar_spaces[k])
        R_bar = create_restriction(self.bar_spaces[k - 1])

        pi_0_d_bar = R_0 @ _diff @ R_bar.T

        return lambda f: R_bar.T @ sps.linalg.spsolve(pi_0_d_bar, R_0 @ f)

    def apply(self, k, f):
        """Apply the Poincare operator

        Args:
            k (int): order of the differential form
            f (np.ndarray): the differential form

        Returns:
            np.ndarray: the image of the Poincaré operator under f
        """

        return self.op[k](f)

    def decompose(self, k, f):
        """use the Poincaré operators to decompose f = pd(f) + dp(f)

        Args:
            k (int): order of the k-form f
            f (np.ndarray): the function to be decomposed

        Returns:
            tuple(np.ndarray): the decomposition of f as (dp(f), pd(f))
        """
        n_minus_k = self.dim - k

        if k == self.dim:  # then df = 0
            pdf = np.zeros_like(f)
        else:
            df = diff(self.mdg, n_minus_k) @ f
            pdf = self.apply(k + 1, df)

        if k == 0:  # then dpf = mean(f)
            dpf = self.apply(k, f)
        else:
            pf = self.apply(k, f)
            dpf = diff(self.mdg, n_minus_k + 1) @ pf

        return pdf, dpf

    def check_chain_property(self, k, f):

        if k <= 0:
            ppf = 0
        else:
            pf = self.apply(k, f)
            ppf = self.apply(k - 1, pf)

        assert np.allclose(ppf, 0)


def generate_random_source(sd):
    np.random.seed(0)
    f_0 = np.random.rand(sd.num_nodes)
    f_1 = np.random.rand(sd.num_ridges)
    f_2 = np.random.rand(sd.num_faces)
    f_3 = np.random.rand(sd.num_cells)

    f = [f_0, f_1, f_2, f_3]

    if sd.dim == 2:
        f = f[1:]

    return f


def test_solver():
    # Check the four-step solver.

    N, dim = 16, 3

    sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
    mdg = pg.as_mdg(sd)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    print("h = {:.2e}".format(np.mean(sd.cell_diameters())))

    f = generate_random_source(sd)

    poin = Poincare(mdg, False)

    mdg = poin.mdg

    # assemble matrices
    M = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
    S = [stiff_matrix(mdg, dim - k, None) for k in range(dim + 1)]
    D = [diff(mdg, dim - k) for k in range(dim)]
    MD = [M[k + 1] @ D[k] for k in range(dim)]

    f[0] -= np.sum(M[0] @ f[0])

    def timed_solve(A, b):
        t = time.time()
        sol = sps.linalg.spsolve(A.tocsc(), b)
        print("ndof: {}, Time: {:1.2f}s".format(len(b), time.time() - t))

        return sol

    def solve_subproblem(poin, k, rhs):
        LS = pg.LinearSystem(S[k], rhs)
        LS.flag_ess_bc(~poin.bar_spaces[k], np.zeros_like(poin.bar_spaces[k]))

        return LS.solve(solver=timed_solve)

    for k in range(dim, 0, -1):
        print("k = {}".format(k))

        A = sps.bmat([[M[k - 1], -MD[k - 1].T], [MD[k - 1], S[k]]])
        LS = pg.LinearSystem(A, np.hstack((M[k - 1] @ f[k - 1], M[k] @ f[k])))

        vu = LS.solve(solver=timed_solve)

        v_true = vu[: M[k - 1].shape[0]]
        u_true = vu[M[k - 1].shape[0] :]

        # Step 1
        v_b = solve_subproblem(poin, k - 1, MD[k - 1].T @ f[k])

        # Step 2
        if k >= 2:
            w_v = solve_subproblem(poin, k - 2, MD[k - 2].T @ (f[k - 1] - v_b))
            v = v_b + D[k - 2] @ w_v

        else:  # k - 2 < 0
            print("ndof: 0, Time: -s")
            v = v_b - np.sum(M[0] @ v_b)

        assert np.allclose(v_true, v)

        # Step 3
        u_b = solve_subproblem(poin, k, M[k] @ f[k] - MD[k - 1] @ v)

        # Step 4
        v_u = solve_subproblem(
            poin, k - 1, M[k - 1] @ (v - f[k - 1]) - MD[k - 1].T @ u_b
        )

        u = u_b + D[k - 1] @ v_u
        assert np.allclose(u_true, u)


def test_Poincare_constants(dim=2):
    if dim == 2:
        N_list = 2 ** np.arange(3, 8)
    else:
        N_list = 2 ** np.arange(5)

    table = np.zeros((len(N_list), dim + 1))

    for N_i, N in enumerate(N_list):
        sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
        mdg = pg.as_mdg(sd)

        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        poin = Poincare(mdg)

        table[N_i, 0] = np.mean(sd.cell_diameters())

        for k in range(dim):

            R = create_restriction(poin.bar_spaces[k])
            M = R @ mass_matrix(mdg, dim - k, None) @ R.T
            S = R @ stiff_matrix(mdg, dim - k, None) @ R.T

            if k == 0:  # Remove the constants
                proj = mass_matrix(mdg, dim - k, None)
                proj /= np.sum(proj)
                M_op = lambda v: M @ (v - np.ones(len(v) + 1) @ proj @ R.T @ v)
                M2 = sps.linalg.LinearOperator(matvec=M_op, shape=M.shape)
                labda = sps.linalg.eigsh(
                    M2, 1, M=S, which="LM", tol=1e-6, return_eigenvectors=False
                )

            else:
                labda = sps.linalg.eigsh(
                    M, 1, M=S, which="LM", tol=1e-6, return_eigenvectors=False
                )

            table[N_i, k + 1] = np.sqrt(labda[0])

    with np.printoptions(formatter={"float": "{: 0.2e}".format}):
        print(table)


def test_aux_precond(dim=2, k=1):

    print("n = {}, k = {}".format(dim, k))
    if dim == 2:
        N_list = 2 ** np.arange(3, 8)
    else:
        N_list = 2 ** np.arange(5)

    alpha_list = np.power(10.0, np.arange(-4, 1))

    cond_table = np.zeros((len(N_list), len(alpha_list) + 1))
    iter_table = np.zeros((len(N_list), len(alpha_list)), dtype=int)

    for N_i, N in enumerate(N_list):
        if dim == 2 and N >= 2**7:
            calc_cond = False
        elif dim == 3 and N >= 2**4:
            calc_cond = False
        else:
            calc_cond = True

        sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
        mdg = pg.as_mdg(sd)

        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        poin = Poincare(mdg)

        cond_table[N_i, 0] = np.mean(sd.cell_diameters())

        f = generate_random_source(sd)

        # assemble matrices
        M = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
        S = [stiff_matrix(mdg, dim - k, None) for k in range(dim + 1)]
        D = [diff(mdg, dim - k) for k in range(dim)]

        def nonlocal_iterate(arr):
            nonlocal iters
            iters += 1

        for alpha_i, alpha in enumerate(alpha_list):
            print("N = {}, alpha = {:.2e}".format(N, alpha))
            a_squared = alpha**2
            A = a_squared * M[k] + S[k]

            iters = 0

            def precond(f):
                LS1 = pg.LinearSystem(S[k], f)
                LS1.flag_ess_bc(~poin.bar_spaces[k], np.zeros_like(poin.bar_spaces[k]))
                first_term = LS1.solve()

                LS2 = pg.LinearSystem(a_squared * S[k - 1], D[k - 1].T @ f)
                LS2.flag_ess_bc(
                    ~poin.bar_spaces[k - 1], np.zeros_like(poin.bar_spaces[k - 1])
                )
                second_term = D[k - 1] @ LS2.solve()

                return first_term + second_term

            P = sps.linalg.LinearOperator(matvec=precond, shape=A.shape)

            v, _ = sps.linalg.minres(A, f[k], M=P, callback=nonlocal_iterate, rtol=1e-8)
            if calc_cond:
                lambda_max = sps.linalg.eigs(
                    precond(A), 1, which="LM", tol=1e-4, return_eigenvectors=False
                )
                lambda_min = sps.linalg.eigs(
                    precond(A), 1, which="SM", tol=1e-4, return_eigenvectors=False
                )

                cond_table[N_i, alpha_i + 1] = np.abs(lambda_max[0] / lambda_min[0])
            iter_table[N_i, alpha_i] = iters

    with np.printoptions(formatter={"float": "{:1.2f}".format}):
        print(cond_table)
    print(iter_table)


def test_properties():
    N, dim = 5, 3
    sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
    mdg = pg.as_mdg(sd)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    f = generate_random_source(sd)
    poin = Poincare(mdg)

    # Check the decomposition and chain property
    for k, f_ in enumerate(f):
        pdf, dpf = poin.decompose(k, f_)
        assert np.allclose(f_, pdf + dpf)

        poin.check_chain_property(k, f_)


def plot_trees():
    mdg = pg.unit_grid(2, 1 / 5, as_mdg=True)
    mdg.compute_geometry()

    tree = pg.SpanningTree(mdg, "all_bdry")
    tree.visualize_2d(
        mdg, "tree-cotree.pdf", draw_grid=False, draw_tree=True, draw_cotree=True
    )
    tree.visualize_2d(
        mdg, "grid-tree.pdf", draw_grid=True, draw_tree=True, draw_cotree=False
    )


def plot_trees_mdg():
    import porepy as pp

    mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [0, 1]
    )
    pg.convert_from_pp(mdg)

    mdg.compute_geometry()

    tree = pg.SpanningTree(mdg, "all_bdry")
    tree.visualize_2d(
        mdg, "mdg_cotree.pdf", draw_grid=True, draw_tree=True, draw_cotree=False
    )
    tree.visualize_2d(
        mdg, "mdg_tree.pdf", draw_grid=False, draw_tree=True, draw_cotree=True
    )


if __name__ == "__main__":

    print("Solving the Hodge-Laplace problem")
    test_solver()

    print("Computing the Poincaré constants")
    for dim in [2, 3]:
        test_Poincare_constants(dim)

    print("Preconditioning the projection problem")
    for dim in [2, 3]:
        for k in range(1, dim):
            test_aux_precond(dim, k)
