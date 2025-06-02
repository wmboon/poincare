import os
import sys
import time

import numpy as np
import porepy as pp
import pyamg
import pygeon as pg
import scipy.sparse as sps
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.innerproducts import mass_matrix
from pygeon.numerics.linear_system import create_restriction
from pygeon.numerics.stiffness import stiff_matrix

sys.path.append(os.path.dirname(__file__))
import mdg_setup


def generate_random_source(mdg: pg.MixedDimensionalGrid):
    sd = mdg.subdomains(dim=mdg.dim_max())[0]

    np.random.seed(0)
    f_0 = np.random.rand(sd.num_nodes)
    f_1 = np.random.rand(mdg.num_subdomain_ridges())
    f_2 = np.random.rand(mdg.num_subdomain_faces())
    f_3 = np.random.rand(mdg.num_subdomain_cells())

    f = [f_0, f_1, f_2, f_3][3 - sd.dim :]

    return f


def generate_N_list(dim, n_grids=5):
    if dim == 2:
        N_list = 2 ** np.arange(3, 3 + n_grids)
    else:
        N_list = 2 ** np.arange(n_grids)
    return N_list


def check_chain_property(poin, k, f):
    if k <= 0:
        ppf = 0
    else:
        pf = poin.apply(k, f)
        ppf = poin.apply(k - 1, pf)

    assert np.allclose(ppf, 0)


def test_properties_cube(N=2, dim=3):
    mdg = pg.unit_grid(dim, 1 / N)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    test_properties(mdg)


def test_properties_mdg(N=15):
    mesh_kwargs = {"mesh_size_frac": 1 / N, "mesh_size_min": 1 / (2 * N)}
    mdg = mdg_setup.fracture_grid(mesh_kwargs)

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()
    pp.Exporter(mdg, "grid").write_vtu()

    test_properties(mdg)
    test_solver(mdg)


def test_properties_seven(N=1):
    mesh_kwargs = {"cell_size": 1 / N, "cell_size_fracture": 1 / N}
    mdg, _ = pp.mdg_library.seven_fractures_one_L_intersection(mesh_kwargs)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    test_properties(mdg)
    test_solver(mdg)


def test_properties(mdg):
    poin = pg.Poincare(mdg)
    dim = mdg.dim_max()

    f = generate_random_source(mdg)
    tip_ridges, tip_faces = get_tips(mdg)
    f[dim - 1][tip_faces] = 0
    f[dim - 2][tip_ridges] = 0

    # Check the decomposition and chain property
    for k, f_ in enumerate(f):
        pdf, dpf, qf = poin.decompose(k, f_, True)
        assert np.allclose(f_, pdf + dpf + qf)

        check_chain_property(poin, k, f_)

    print("All properties passed, dim = {}".format(mdg.dim_max()))

    new_basis = poin.orthogonalize_cohomology_basis(1)


def get_tips(mdg):
    tip_ridges = np.concatenate([sd.tags["tip_ridges"] for sd in mdg.subdomains()])
    tip_faces = np.concatenate([sd.tags["tip_faces"] for sd in mdg.subdomains()])

    return tip_ridges, tip_faces


def test_solver_cube(N=16, dim=3):
    # Check the four-step solver.
    mdg = pg.unit_grid(dim, 1 / N)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    test_solver(mdg)


def test_solver(mdg: pg.MixedDimensionalGrid):
    dim = mdg.dim_max()
    sd = mdg.subdomains(dim=dim)[0]
    print("h = {:.2e}".format(np.mean(sd.cell_diameters())))

    f = generate_random_source(mdg)

    poin = pg.Poincare(mdg)

    # Assemble mass and stiffness matrices
    M = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
    D = [diff(mdg, dim - k) for k in range(dim)]
    MD = [M[k + 1] @ D[k] for k in range(dim)]
    S = [D[k].T @ MD[k] for k in range(dim)]
    S.append(stiff_matrix(mdg, 0, None))

    f[0] -= np.sum(M[0] @ f[0]) / np.sum(M[0])

    tip_ridges, tip_faces = get_tips(mdg)
    ess_vals = [np.zeros_like(f_i, dtype=bool) for f_i in f]
    ess_vals[dim - 1] = tip_faces
    ess_vals[dim - 2] = tip_ridges

    def timed_solve(A, b):
        t = time.time()
        sol = sps.linalg.spsolve(A.tocsc(), b)
        print("ndof: {}, Time: {:1.2f}s".format(len(b), time.time() - t))

        return sol

    for k in range(1, dim + 1):
        print("k = {}".format(k))

        A = sps.bmat([[M[k - 1], -MD[k - 1].T], [MD[k - 1], S[k]]])
        b = np.hstack((M[k - 1] @ f[k - 1], M[k] @ f[k]))
        LS = pg.LinearSystem(A, b)

        ess_bc = np.concatenate((ess_vals[k - 1], ess_vals[k]))
        LS.flag_ess_bc(ess_bc, np.zeros_like(ess_bc))

        vu = LS.solve(solver=timed_solve)

        v_true = vu[: M[k - 1].shape[0]]
        u_true = vu[M[k - 1].shape[0] :]

        # Step 1
        v_b = poin.solve_subproblem(k - 1, S[k - 1], MD[k - 1].T @ f[k], timed_solve)

        # Step 2
        if k >= 2:
            w_v = poin.solve_subproblem(
                k - 2, S[k - 2], MD[k - 2].T @ (f[k - 1] - v_b), timed_solve
            )
            v = v_b + D[k - 2] @ w_v

        else:  # k - 2 < 0
            print("ndof: 0, Time: -s")
            v = v_b - np.sum(M[0] @ v_b) / np.sum(M[0])

        assert np.allclose(v_true, v)

        # Step 3
        u_b = poin.solve_subproblem(k, S[k], M[k] @ f[k] - MD[k - 1] @ v, timed_solve)

        # Step 4
        v_u = poin.solve_subproblem(
            k - 1, S[k - 1], M[k - 1] @ (v - f[k - 1]) - MD[k - 1].T @ u_b, timed_solve
        )

        u = u_b + D[k - 1] @ v_u
        assert np.allclose(u_true, u)


def test_Poincare_constants(dim=2):
    N_list = generate_N_list(dim)

    table = np.zeros((len(N_list), dim + 1))

    for N_i, N in enumerate(N_list):
        mdg = pg.unit_grid(dim, 1 / N)

        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        poin = pg.Poincare(mdg)

        table[N_i, 0] = np.mean(poin.top_sd.cell_diameters())

        for k in range(dim):
            R = create_restriction(poin.bar_spaces[k])
            M = R @ mass_matrix(mdg, dim - k, None) @ R.T
            S = R @ stiff_matrix(mdg, dim - k, None) @ R.T

            if k == 0:  # Remove the constants
                proj = mass_matrix(mdg, dim - k, None)
                proj /= np.sum(proj)

                def M_op(v):
                    return M @ (v - np.ones(len(v) + 1) @ proj @ R.T @ v)

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


def test_aux_precond(dim=2, k=1, use_direct=True):
    print("n = {}, k = {}".format(dim, k))
    N_list = generate_N_list(dim)

    alpha_list = np.power(10.0, np.arange(-4, 1))

    cond_table = np.zeros((len(N_list), len(alpha_list) + 1))
    iter_table = np.zeros((len(N_list), len(alpha_list)), dtype=int)

    for N_i, N in enumerate(N_list):
        if use_direct:
            if dim == 2 and N >= 2**7:
                calc_cond = False
            elif dim == 3 and N >= 2**4:
                calc_cond = False
            else:
                calc_cond = True
        else:
            calc_cond = False

        sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
        mdg = pg.as_mdg(sd)

        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        poin = pg.Poincare(mdg)

        cond_table[N_i, 0] = np.mean(sd.cell_diameters())

        f = generate_random_source(mdg)

        # assemble matrices
        M = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
        S = [stiff_matrix(mdg, dim - k, None) for k in range(dim + 1)]
        D = [diff(mdg, dim - k) for k in range(dim)]

        def nonlocal_iterate(arr):
            nonlocal iters
            iters += 1

        # Define preconditioners
        if use_direct:

            def precond_base(f, a_squared):
                LS1 = pg.LinearSystem(S[k], f)
                LS1.flag_ess_bc(~poin.bar_spaces[k], np.zeros_like(poin.bar_spaces[k]))
                first_term = LS1.solve()

                LS2 = pg.LinearSystem(a_squared * S[k - 1], D[k - 1].T @ f)
                LS2.flag_ess_bc(
                    ~poin.bar_spaces[k - 1], np.zeros_like(poin.bar_spaces[k - 1])
                )
                second_term = D[k - 1] @ LS2.solve()

                return first_term + second_term

        if not use_direct:
            AMG1 = compute_AMG_solver(S[k], poin.bar_spaces[k])
            AMG2 = compute_AMG_solver(S[k - 1], poin.bar_spaces[k - 1])

            def precond_base(f, a_squared):
                LS1 = pg.LinearSystem(S[k], f)
                LS1.flag_ess_bc(~poin.bar_spaces[k], np.zeros_like(poin.bar_spaces[k]))
                if k == dim - 1:
                    first_term = LS1.solve()
                else:
                    first_term = LS1.solve(AMG1)

                LS2 = pg.LinearSystem(S[k - 1], D[k - 1].T @ f)
                LS2.flag_ess_bc(
                    ~poin.bar_spaces[k - 1], np.zeros_like(poin.bar_spaces[k - 1])
                )
                second_term = D[k - 1] @ LS2.solve(AMG2) / a_squared

                return first_term + second_term

        for alpha_i, alpha in enumerate(alpha_list):
            print("N = {}, alpha = {:.2e}".format(N, alpha))
            a_squared = alpha**2
            A = a_squared * M[k] + S[k]

            iters = 0

            def precond(f):
                return precond_base(f, a_squared)

            P = sps.linalg.LinearOperator(matvec=precond, shape=A.shape)

            sps.linalg.minres(A, f[k], M=P, callback=nonlocal_iterate, rtol=1e-8)
            if calc_cond:
                lambda_max = sps.linalg.eigs(
                    precond(A),
                    1,
                    which="LM",
                    tol=1e-4,
                    return_eigenvectors=False,
                )
                lambda_min = sps.linalg.eigs(
                    precond(A),
                    1,
                    which="SM",
                    tol=1e-4,
                    return_eigenvectors=False,
                )

                cond_table[N_i, alpha_i + 1] = np.abs(lambda_max[0] / lambda_min[0])
            iter_table[N_i, alpha_i] = iters

    if calc_cond:
        with np.printoptions(formatter={"float": "{:1.2f}".format}):
            print(cond_table)
    print(iter_table)


def compute_AMG_solver(A, bar_space):
    LS = pg.LinearSystem(A)
    LS.flag_ess_bc(~bar_space, np.zeros_like(bar_space))
    A_red = sps.csr_matrix(LS.reduce_system()[0])

    A_red.indices = A_red.indices.astype(np.int32)
    A_red.indptr = A_red.indptr.astype(np.int32)

    AMG = pyamg.smoothed_aggregation_solver(A_red)
    AMG = AMG.aspreconditioner()

    def solver(_, f):
        return AMG(f)

    return solver


def plot_trees():
    mdg = pg.unit_grid(2, 1 / 5, as_mdg=True)
    mdg.compute_geometry()

    tree = pg.SpanningTree(mdg, "first_bdry")
    tree.visualize_2d(
        mdg,
        "first_cotree.svg",
        draw_grid=True,
        draw_tree=True,
        draw_cotree=False,
        start_color="blue",
    )
    tree.visualize_2d(
        mdg,
        "first_tree.svg",
        draw_grid=False,
        draw_tree=True,
        draw_cotree=True,
        start_color="blue",
    )

    tree = pg.SpanningTree(mdg, "all_bdry")
    tree.visualize_2d(
        mdg,
        "all_cotree.svg",
        draw_grid=True,
        draw_tree=True,
        draw_cotree=False,
        start_color="blue",
    )
    tree.visualize_2d(
        mdg,
        "all_tree.svg",
        draw_grid=False,
        draw_tree=True,
        draw_cotree=True,
        start_color="blue",
    )


def plot_trees_mdg():
    mesh_args = {"cell_size": 0.25, "cell_size_fracture": 0.125}
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "simplex", mesh_args, [0, 1]
    )
    pg.convert_from_pp(mdg)

    mdg.compute_geometry()

    # Top-dimensional domain
    top_sd = mdg.subdomains()[0]
    cn = top_sd.cell_nodes()

    def drift(sd, cn, indices, vector):
        nodes = cn @ indices
        sd.nodes[:, nodes] += vector[:, None]

    eps = 0.05

    ne = np.logical_and(
        top_sd.cell_centers[0, :] > 0.5, top_sd.cell_centers[1, :] > 0.5
    )
    drift(top_sd, cn, ne, np.array([eps, eps, 0]))

    nw = np.logical_and(
        top_sd.cell_centers[0, :] < 0.5, top_sd.cell_centers[1, :] > 0.5
    )
    drift(top_sd, cn, nw, np.array([-eps, eps, 0]))

    se = np.logical_and(
        top_sd.cell_centers[0, :] > 0.5, top_sd.cell_centers[1, :] < 0.5
    )
    drift(top_sd, cn, se, np.array([eps, -eps, 0]))

    sw = np.logical_and(
        top_sd.cell_centers[0, :] < 0.5, top_sd.cell_centers[1, :] < 0.5
    )
    drift(top_sd, cn, sw, np.array([-eps, -eps, 0]))

    top_sd.compute_geometry()

    # One-dimensional fractures
    for top_sd in mdg.subdomains(dim=1):
        cn = top_sd.cell_nodes()

        vec = top_sd.cell_centers - np.array([0.5, 0.5, 0])[:, None]
        mean_vec = vec.mean(axis=1)
        mean_vec /= np.linalg.norm(mean_vec)

        pos = mean_vec @ vec > 0
        drift(top_sd, cn, pos, eps * mean_vec)

        neg = mean_vec @ vec < 0
        drift(top_sd, cn, neg, -eps * mean_vec)

        top_sd.compute_geometry()

    tree = pg.SpanningTree(mdg, "all_bdry")

    tree.visualize_2d(
        mdg,
        "mdg_grid.svg",
        draw_grid=True,
        draw_tree=False,
        draw_cotree=False,
    )

    tree.visualize_2d(
        mdg,
        "mdg_cotree.svg",
        draw_grid=True,
        draw_tree=True,
        draw_cotree=False,
        start_color="blue",
    )
    tree.visualize_2d(
        mdg,
        "mdg_tree.svg",
        draw_grid=False,
        draw_tree=True,
        draw_cotree=True,
        start_color="blue",
    )


def test_properties_holes_2D():
    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(
        "/home/AD.NORCERESEARCH.NO/wibo/pygeon/two_holes_2D.geo", 2
    )

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    test_properties(mdg)

    # test_solver(mdg)


def test_properties_holes_3D():
    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(
        "/home/AD.NORCERESEARCH.NO/wibo/pygeon/two_holes_3D.geo", 3
    )

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    test_properties(mdg)
    # test_solver(mdg)


def visualize_tree_with_holes():
    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(
        "/home/AD.NORCERESEARCH.NO/wibo/pygeon/two_holes_2D.geo", 2
    )

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    tree = pg.SpanningTree(mdg, "all_bdry")

    tree.visualize_2d(
        mdg,
        "holes_cotree.svg",
        draw_grid=True,
        draw_tree=True,
        draw_cotree=False,
        start_color="blue",
    )
    tree.visualize_2d(
        mdg,
        "holes_trees.svg",
        draw_grid=False,
        draw_tree=True,
        draw_cotree=True,
        start_color="blue",
    )


if __name__ == "__main__":
    # plot_trees_mdg()
    test_properties_holes_2D()
    test_properties_holes_3D()
    # test_properties_mdg(5)
    # test_properties_seven()
    print("Testing decomposition and co-chain properties")
    for dim in [2, 3]:
        test_properties_cube(dim=dim)

    print("Solving the Hodge-Laplace problem")
    test_solver_cube(5)

    # print("Computing the Poincaré constants")
    # for dim in [2, 3]:
    #     test_Poincare_constants(dim)

    # print("Preconditioning the projection problem")
    # for dim in [2, 3]:
    #     for k in range(1, dim):
    #         test_aux_precond(dim, k)

    # print("Preconditioning the projection problem with AMG")
    # test_aux_precond(3, 1, False)
