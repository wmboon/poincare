import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.linear_system import create_restriction


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
        self.flag_entities()
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

    def flag_entities(self):
        """
        Flag the mesh entities that will be used to generate the Poincaré operators
        """
        self.flagged_cells = np.ones(self.mdg.num_subdomain_cells(), dtype=bool)
        self.flagged_faces = pg.SpanningTree(mdg, "all_bdry").flagged_faces
        self.flagged_edges = self.flag_edges()
        self.flagged_nodes = self.flag_nodes()

    def flag_edges(self):
        """
        Flag the edges of the grid that form a spanning tree of the nodes

        Returns:
            np.ndarray: boolean array with flagged edges
        """
        if self.dim == 2:
            return ~self.flagged_faces
        elif self.dim == 3:
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

            flagged_edges = np.zeros(grad.shape[0], dtype=bool)
            flagged_edges[tree_edges] = True

            return flagged_edges

    def find_central_node(self):

        sd = mdg.subdomains()[0]
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

        # Cells to faces, similar to the spanning tree
        if n > 1:
            self.op[n] = self.create_op(n, self.flagged_cells, self.flagged_faces)

        # Faces to edges (only for 3D)
        if n > 2:
            self.op[2] = self.create_op(2, ~self.flagged_faces, ~self.flagged_edges)

        # Edges to nodes
        op_nodal = self.create_op(1, self.flagged_edges, self.flagged_nodes)
        subtract_mean = lambda x: x - np.mean(x)
        self.op[1] = lambda f: subtract_mean(op_nodal(f))

        # Nodes to the constants
        self.op[0] = lambda f: np.full_like(f, np.mean(f))

    def create_op(self, k, flag_dom, flag_ran):
        """
        Create the Poincaré operator for k-forms

        Args:
            k (int): order of the form
            flag_dom (np.ndarray): flagged k-entities in the domain
            flag_ran (np.ndarray): flagged (k-1)-entities in the range

        Returns:
            function: the Poincaré operator
        """
        n_minus_k = self.dim - k
        _diff = diff(self.mdg, n_minus_k + 1)

        R_dom = create_restriction(flag_dom)
        R_ran = create_restriction(flag_ran)

        pi_bar_diff = R_dom @ _diff @ R_ran.T

        return lambda f: R_ran.T @ sps.linalg.spsolve(pi_bar_diff, R_dom @ f)

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

    N, dim = 5, 3
    sd = pg.unit_grid(2, 0.15, as_mdg=False)
    mdg = pg.as_mdg(sd)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    tree = pg.SpanningTree(mdg, "all_bdry")
    tree.visualize_2d(mdg, "tree.pdf", False, True, True)

    f_0 = np.random.rand(sd.num_nodes)
    f_1 = np.random.rand(sd.num_ridges)
    f_2 = np.random.rand(sd.num_faces)
    f_3 = np.random.rand(sd.num_cells)

    poin = Poincare(mdg)

    for k, f in enumerate([f_0, f_1, f_2, f_3]):
        pdf, dpf = poin.decompose(k, f)
        assert np.allclose(f, pdf + dpf)

        poin.check_chain_property(k, f)

    # Solve a Hodge Laplace problem
    S_c = pg.cell_stiff(mdg)
    S_f = pg.face_stiff(mdg)
    S_r = pg.ridge_stiff(mdg)
    S_p = pg.peak_stiff(mdg)

    # d_c = pg.peak_mass(mdg) @ diff(mdg, 0)
    d_f = pg.cell_mass(mdg) @ pg.div(mdg)
    d_r = pg.face_mass(mdg) @ pg.curl(mdg)
    d_p = pg.ridge_mass(mdg) @ pg.grad(mdg)

    print("k = n")

    A = sps.bmat([[pg.face_mass(mdg), -d_f.T], [d_f, S_c]])
    LS = pg.LinearSystem(
        A, np.hstack((np.zeros(sd.num_faces), pg.cell_mass(mdg) @ f_3))
    )
    vu = LS.solve()
    print(len(vu))

    v_true = vu[: S_f.shape[0]]
    u_true = vu[S_f.shape[0] :]

    # step 1
    LS = pg.LinearSystem(S_f, d_f.T @ f_3)
    LS.flag_ess_bc(~poin.flagged_faces, np.zeros_like(poin.flagged_faces))
    print(np.sum(poin.flagged_faces))

    v0 = LS.solve()

    # step 2
    LS = pg.LinearSystem(S_r, d_r.T @ (f_2 - v0))
    LS.flag_ess_bc(poin.flagged_edges, np.zeros_like(poin.flagged_edges))
    print(np.sum(~poin.flagged_edges))

    v1 = LS.solve()

    v = v0 + pg.curl(mdg) @ v1
    assert np.allclose(v_true, v)

    # step 4
    LS = pg.LinearSystem(S_f, pg.face_mass(mdg) @ (v - f_2))
    LS.flag_ess_bc(~poin.flagged_faces, np.zeros_like(poin.flagged_faces))
    print(np.sum(poin.flagged_faces))

    v4 = LS.solve()

    u = pg.div(mdg) @ v4
    assert np.allclose(u_true, u)

    ## vector laplacian (k = n - 1)
    print("k = n - 1")

    A = sps.bmat([[pg.ridge_mass(mdg), -d_r.T], [d_r, S_f]])
    LS = pg.LinearSystem(
        A, np.hstack((pg.ridge_mass(mdg) @ f_1, pg.face_mass(mdg) @ f_2))
    )
    vu = LS.solve()
    print(len(vu))

    v_true = vu[: S_r.shape[0]]
    u_true = vu[S_r.shape[0] :]

    # step 1
    LS = pg.LinearSystem(S_r, d_r.T @ f_2)
    LS.flag_ess_bc(poin.flagged_edges, np.zeros_like(poin.flagged_edges))
    print(np.sum(~poin.flagged_edges))

    v0 = LS.solve()

    # step 2
    LS = pg.LinearSystem(S_p, d_p.T @ (f_1 - v0))
    LS.flag_ess_bc(~poin.flagged_nodes, np.zeros_like(poin.flagged_nodes))
    print(np.sum(poin.flagged_nodes))

    v1 = LS.solve()

    v = v0 + pg.grad(mdg) @ v1
    assert np.allclose(v_true, v)

    # step 3
    LS = pg.LinearSystem(S_f, pg.face_mass(mdg) @ f_2 - d_r @ v0)
    LS.flag_ess_bc(~poin.flagged_faces, np.zeros_like(poin.flagged_faces))
    print(np.sum(poin.flagged_faces))

    v3 = LS.solve()

    # step 4
    LS = pg.LinearSystem(S_r, pg.ridge_mass(mdg) @ (v - f_1) - d_r.T @ v3)
    LS.flag_ess_bc(poin.flagged_edges, np.zeros_like(poin.flagged_edges))
    print(np.sum(~poin.flagged_edges))

    v4 = LS.solve()

    u = v3 + pg.curl(mdg) @ v4
    assert np.allclose(u_true, u)

    ## vector laplacian (k = n - 2)
    print("k = n - 2")

    A = sps.bmat([[pg.peak_mass(mdg), -d_p.T], [d_p, S_r]])
    LS = pg.LinearSystem(
        A, np.hstack((pg.peak_mass(mdg) @ f_0, pg.ridge_mass(mdg) @ f_1))
    )
    vu = LS.solve()
    print(len(vu))

    v_true = vu[: S_p.shape[0]]
    u_true = vu[S_p.shape[0] :]

    # step 1
    LS = pg.LinearSystem(S_p, d_p.T @ f_1)
    LS.flag_ess_bc(~poin.flagged_nodes, np.zeros_like(poin.flagged_nodes))
    print(np.sum(poin.flagged_nodes))

    v0 = LS.solve()

    # step 2

    v = v0 - np.mean(v0) + np.mean(v_true)
    assert np.allclose(v_true, v)

    # step 3
    LS = pg.LinearSystem(S_r, pg.ridge_mass(mdg) @ f_1 - d_p @ v0)
    LS.flag_ess_bc(poin.flagged_edges, np.zeros_like(poin.flagged_edges))
    print(np.sum(~poin.flagged_edges))

    v3 = LS.solve()

    # step 4
    LS = pg.LinearSystem(S_p, pg.peak_mass(mdg) @ (v - f_0) - d_p.T @ v3)
    LS.flag_ess_bc(~poin.flagged_nodes, np.zeros_like(poin.flagged_nodes))
    print(np.sum(poin.flagged_nodes))

    v4 = LS.solve()

    u = v3 + pg.grad(mdg) @ v4
    assert np.allclose(u_true, u)

    pass
