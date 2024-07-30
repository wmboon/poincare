import numpy as np
import scipy.sparse as sps

import pygeon as pg
from pygeon.numerics.differentials import exterior_derivative as diff


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
        self.flagged_faces = pg.SpanningTree(mdg).flagged_faces
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

            tree = sps.csgraph.breadth_first_tree(incidence, 0, directed=False)
            c_start, c_end, _ = sps.find(tree)

            rows = np.hstack((c_start, c_end))
            cols = np.hstack([np.arange(c_start.size)] * 2)
            vals = np.ones_like(rows)

            edge_finder = sps.csc_array(
                (vals, (rows, cols)), shape=(grad.shape[1], tree.nnz)
            )
            edge_finder = np.abs(grad) @ edge_finder
            I, J, V = sps.find(edge_finder)

            _, index = np.unique(J[V == 2], return_index=True)
            tree_edges = I[V == 2][index]

            flagged_edges = np.zeros(grad.shape[0], dtype=bool)
            flagged_edges[tree_edges] = True

            return flagged_edges

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

        R_dom = pg.numerics.linear_system.create_restriction(flag_dom)
        R_ran = pg.numerics.linear_system.create_restriction(flag_ran)

        splu = sps.linalg.splu(R_dom @ _diff @ R_ran.T)

        return lambda f: R_ran.T @ splu.solve(R_dom @ f)

    def apply(self, n_minus_k, f):
        """Apply the Poincare operator

        Args:
            k (int): order of the differential form
            f (np.ndarray): the differential form

        Returns:
            np.ndarray: the image of the Poincaré operator under f
        """

        return self.op[n_minus_k](f)

    def decompose(self, k, f):
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


if __name__ == "__main__":

    N, dim = 5, 3
    # sd = pp.StructuredTetrahedralGrid([N] * dim, [1] * dim)
    # sd = pp.StructuredTriangleGrid([N] * dim, [1] * dim)
    sd = pg.unit_grid(3, 0.1, as_mdg=False)
    mdg = pg.as_mdg(sd)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    f_0 = np.random.rand(sd.num_nodes)
    f_1 = np.random.rand(sd.num_ridges)
    f_2 = np.random.rand(sd.num_faces)
    f_3 = np.random.rand(sd.num_cells)

    poin = Poincare(mdg)

    for k, f in enumerate([f_0, f_1, f_2, f_3]):
        pdf, dpf = poin.decompose(k, f)
        assert np.allclose(f, pdf + dpf)

    pass
