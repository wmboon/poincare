import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.linear_system import create_restriction
from pygeon.numerics.innerproducts import mass_matrix
from pygeon.numerics.stiffness import stiff_matrix


class Poincare:
    def __init__(self, mdg):
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
        self.bar_spaces[self.dim - 1] = pg.SpanningTree(mdg, "all_bdry").flagged_faces

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

        return np.argmax(dists == np.min(dists))

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


if __name__ == "__main__":

    N, dim = 10, 2

    sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
    mdg = pg.as_mdg(sd)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    # tree = pg.SpanningTree(mdg, "all_bdry")
    # tree.visualize_2d(mdg, "tree.pdf", False, True, True)

    f_0 = np.random.rand(sd.num_nodes)
    f_1 = np.random.rand(sd.num_ridges)
    f_2 = np.random.rand(sd.num_faces)
    f_3 = np.random.rand(sd.num_cells)

    f = [f_0, f_1, f_2, f_3]

    if dim == 2:
        f = f[1:]

    poin = Poincare(mdg)
    pg.cell_stiff

    for k, f_ in enumerate(f):
        pdf, dpf = poin.decompose(k, f_)
        assert np.allclose(f_, pdf + dpf)

        poin.check_chain_property(k, f_)

    # assemble matrices
    M = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
    S = [stiff_matrix(mdg, dim - k, None) for k in range(dim + 1)]
    D = [diff(mdg, dim - k) for k in range(dim)]
    MD = [M[k + 1] @ D[k] for k in range(dim)]

    for k in range(dim, 0, -1):
        print("k = {}".format(k))

        A = sps.bmat([[M[k - 1], -MD[k - 1].T], [MD[k - 1], S[k]]])
        LS = pg.LinearSystem(A, np.hstack((M[k - 1] @ f[k - 1], M[k] @ f[k])))

        vu = LS.solve()
        print(len(vu))

        v_true = vu[: M[k - 1].shape[0]]
        u_true = vu[M[k - 1].shape[0] :]

        # Step 1
        LS = pg.LinearSystem(S[k - 1], MD[k - 1].T @ f[k])
        LS.flag_ess_bc(~poin.bar_spaces[k - 1], np.zeros_like(poin.bar_spaces[k - 1]))
        print(np.sum(poin.bar_spaces[k - 1]))

        v0 = LS.solve()

        # Step 2
        if k >= 2:
            LS = pg.LinearSystem(S[k - 2], MD[k - 2].T @ (f[k - 1] - v0))
            LS.flag_ess_bc(
                ~poin.bar_spaces[k - 2], np.zeros_like(poin.bar_spaces[k - 2])
            )
            print(np.sum(poin.bar_spaces[k - 2]))

            v1 = LS.solve()
            v = v0 + D[k - 2] @ v1
        else:
            print(0)
            v = v0 - np.mean(v0 - v_true)

        assert np.allclose(v_true, v)

        # Step 3
        LS = pg.LinearSystem(S[k], M[k] @ f[k] - MD[k - 1] @ v0)
        LS.flag_ess_bc(~poin.bar_spaces[k], np.zeros_like(poin.bar_spaces[k]))
        print(np.sum(poin.bar_spaces[k]))

        v3 = LS.solve()

        # Step 4
        LS = pg.LinearSystem(S[k - 1], M[k - 1] @ (v - f[k - 1]) - MD[k - 1].T @ v3)
        LS.flag_ess_bc(~poin.bar_spaces[k - 1], np.zeros_like(poin.bar_spaces[k - 1]))
        print(np.sum(poin.bar_spaces[k - 1]))

        v4 = LS.solve()

        u = v3 + D[k - 1] @ v4
        assert np.allclose(u_true, u)

    pass
