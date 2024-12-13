import numpy as np
import scipy.sparse as sps
import time

import pygeon as pg
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.linear_system import create_restriction
from pygeon.numerics.innerproducts import mass_matrix
from pygeon.numerics.stiffness import stiff_matrix


def generate_random_source(mdg: pg.MixedDimensionalGrid):
    sd = mdg.subdomains(dim=mdg.dim_max())[0]

    np.random.seed(0)
    f_0 = np.random.rand(sd.num_nodes)
    f_1 = np.random.rand(mdg.num_subdomain_ridges())
    f_2 = np.random.rand(mdg.num_subdomain_faces())
    f_3 = np.random.rand(mdg.num_subdomain_cells())

    f = [f_0, f_1, f_2, f_3]

    if sd.dim == 2:
        f = f[1:]

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


def test_properties(N=2, dim=3):
    mdg = pg.unit_grid(dim, 1 / N)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    f = generate_random_source(mdg)
    poin = pg.Poincare(mdg)

    # Check the decomposition and chain property
    for k, f_ in enumerate(f):
        pdf, dpf = poin.decompose(k, f_)
        assert np.allclose(f_, pdf + dpf)

        check_chain_property(poin, k, f_)

    print("All properties passed, dim = {}".format(dim))


def test_solver(N=10, dim=3):

    # Check the four-step solver.
    mdg = pg.unit_grid(dim, 1 / N)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    sd = mdg.subdomains(dim=dim)[0]
    print("h = {:.2e}".format(np.mean(sd.cell_diameters())))

    f = generate_random_source(mdg)

    poin = pg.Poincare(mdg)

    mdg = poin.mdg

    # Assemble mass and stiffness matrices
    M = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
    D = [diff(mdg, dim - k) for k in range(dim)]
    MD = [M[k + 1] @ D[k] for k in range(dim)]
    S = [D[k].T @ MD[k] for k in range(dim)]
    S.append(stiff_matrix(mdg, 0, None))

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

    for k in range(1, dim + 1):
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
    N_list = generate_N_list(dim)

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

            sps.linalg.minres(A, f[k], M=P, callback=nonlocal_iterate, rtol=1e-8)
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


def plot_trees():
    mdg = pg.unit_grid(2, 1 / 5, as_mdg=True)
    mdg.compute_geometry()

    tree = pg.SpanningTree(mdg, "first_bdry")
    tree.visualize_2d(
        mdg, "first_cotree.pdf", draw_grid=True, draw_tree=True, draw_cotree=False
    )
    tree.visualize_2d(
        mdg, "first_tree.pdf", draw_grid=False, draw_tree=True, draw_cotree=True
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

    print("Testing decomposition and co-chain properties")
    for dim in [2, 3]:
        test_properties(dim=dim)

    print("Solving the Hodge-Laplace problem")
    test_solver()

    print("Computing the Poincaré constants")
    for dim in [2, 3]:
        test_Poincare_constants(dim)

    print("Preconditioning the projection problem")
    for dim in [2, 3]:
        for k in range(1, dim):
            test_aux_precond(dim, k)
