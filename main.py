import os
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import porepy as pp
import pygeon as pg
import scipy.sparse as sps
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.innerproducts import mass_matrix
from pygeon.numerics.stiffness import stiff_matrix

import mdg_setup
import plotter

dirname = os.path.dirname(__file__)


def generate_random_source(mdg: pg.MixedDimensionalGrid):
    sd = mdg.subdomains(dim=mdg.dim_max())[0]

    np.random.seed(0)
    f_0 = np.random.rand(sd.num_nodes)
    f_1 = np.random.rand(mdg.num_subdomain_ridges())
    f_2 = np.random.rand(mdg.num_subdomain_faces())
    f_3 = np.random.rand(mdg.num_subdomain_cells())

    f = [f_0, f_1, f_2, f_3][3 - sd.dim :]

    return f


def generate_sin_source(mdg: pg.MixedDimensionalGrid):
    sd = mdg.subdomains(dim=mdg.dim_max())[0]

    def source(x):
        return np.prod(np.sin(2 * np.pi * x[: sd.dim]))

    def source_vec(x):
        return source(x) * np.array([1, -1, 2])

    f_0 = pg.Lagrange1().interpolate(sd, source)

    f_2 = pg.RT0().interpolate(sd, source_vec)
    f_3 = pg.PwConstants().interpolate(sd, source)

    if sd.dim == 3:
        f_1 = pg.NedelecR0().interpolate(sd, source_vec)

        return [f_0, f_1, f_2, f_3]
    else:
        return [f_0, f_2, f_3]


def generate_N_list(dim, n_grids=5):
    if dim == 2:
        N_list = 2 ** np.arange(3, 3 + n_grids)
    else:
        N_list = 2 ** np.arange(n_grids)
    return N_list


def generate_N_list_Poincare(dim, n_grids=5):
    if dim == 2:
        N_list = 2 ** np.arange(3, 3 + n_grids)
    else:
        N_list = 2 * 1.5 ** np.arange(2, 2 + n_grids)
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
        pdf, dpf, qf = poin.decompose(k, f_)
        assert np.allclose(f_, pdf + dpf + qf)

        check_chain_property(poin, k, f_)

    print(f"All properties passed, dim = {mdg.dim_max()}")


def visualize_cohomology(mdg, k):
    data_list = []
    poin = pg.Poincare(mdg)

    A = poin.compute_basis_harmonic_forms(k)

    if k == 1:
        space = pg.RT0() if poin.dim == 2 else pg.NedelecR0()
    elif k == 2:
        space = pg.RT0()

    eval = space.eval_at_cell_centers(poin.top_sd)

    for i, col in enumerate(A.T):
        cell_u = (eval @ col).reshape((3, -1))
        data_list.append((f"cohom_{k}_{i}", cell_u))

    return data_list


def get_tips(mdg):
    tip_ridges = np.concatenate([sd.tags["tip_ridges"] for sd in mdg.subdomains()])
    tip_faces = np.concatenate([sd.tags["tip_faces"] for sd in mdg.subdomains()])

    return tip_ridges, tip_faces


def test_scaling_solver(dim=3):

    N_list = generate_N_list_Poincare(dim) / 1.5

    t_direct, t_poinc = [], []

    for N in N_list:
        times = test_solver_cube(N, dim, "5.2e").astype(float)
        t_poinc.append(times[0])
        t_direct.append(times[1])

    t_poinc = np.array(t_poinc)
    t_direct = np.array(t_direct)

    np.savetxt("direct_scaling.csv", t_direct, delimiter=",")
    np.savetxt("poinca_scaling.csv", t_poinc, delimiter=",")


def plot_scaling(dim=3):
    t_direct = np.loadtxt("direct_scaling.csv", delimiter=",")
    t_poinc = np.loadtxt("poinca_scaling.csv", delimiter=",")
    import matplotlib.pyplot as plt

    width = 0.6 * 700 / 72.7
    _, ax = plt.subplots(figsize=(width, width * 0.75), constrained_layout=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:dim]
    markers = ["^", "o", "s"]
    for i in range(dim):
        plt.loglog(
            t_poinc[:, 2 * i],
            t_poinc[:, 2 * i + 1],
            "--",
            color=colors[i],
            marker=markers[i],
        )
        plt.loglog(
            t_direct[:, 2 * i],
            t_direct[:, 2 * i + 1],
            color=colors[i],
            marker=markers[i],
        )

    # Legend 1: line style
    from matplotlib.lines import Line2D

    style_handles = [
        Line2D([], [], color="gray", linestyle="-", label="Original system"),
        Line2D([], [], color="gray", linestyle="--", label="Sequential solver"),
    ]

    # Legend 2: markers (no line, just the marker)
    style_handles.extend(
        [
            Line2D([], [], color=colors[0], linestyle="", marker="^", label="$k$ = 1"),
            Line2D([], [], color=colors[1], linestyle="", marker="o", label="$k$ = 2"),
            Line2D([], [], color=colors[2], linestyle="", marker="s", label="$k$ = 3"),
        ]
    )

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Degrees of freedom")

    ax.legend(handles=style_handles)

    plt.savefig("scaling.pdf")

    rates = np.log(t_direct[-1, 1::2] / t_direct[-2, 1::2]) / np.log(
        t_direct[-1, ::2] / t_direct[-2, ::2]
    )

    print(rates)


def test_solver_cube(N=16, dim=3, fmt="5.2f"):
    # Check the four-step solver.
    mdg = pg.unit_grid(dim, 1 / N)
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    return test_solver(mdg, fmt)


def test_solver(mdg: pg.MixedDimensionalGrid, fmt="5.2f"):
    buffer = StringIO()
    sys.stdout = buffer

    dim = mdg.dim_max()
    sd = mdg.subdomains(dim=dim)[0]
    print(f"h = {np.mean(sd.cell_diameters()):.2e}")

    poin = pg.Poincare(mdg)
    f = generate_sin_source(mdg)

    cohom_basis = [poin.compute_basis_harmonic_forms(k) for k in range(dim + 1)]
    e = [np.random.rand(hom_basis.shape[1]) for hom_basis in cohom_basis]

    # Assemble mass and stiffness matrices
    M = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
    D = [diff(mdg, dim - k) for k in range(dim)]
    MD = [M[k + 1] @ D[k] for k in range(dim)]
    S = [D[k].T @ MD[k] for k in range(dim)]
    S.append(stiff_matrix(mdg, 0, None))

    MC = [M[k] @ cohom_basis[k] for k in range(dim + 1)]
    CMC = [cohom_basis[k].T @ MC[k] for k in range(dim + 1)]

    # Multiply right-hand side
    f = [M[k] @ f[k] for k in range(dim + 1)]

    # Zerfo out tips
    tip_ridges, tip_faces = get_tips(mdg)
    ess_vals = [np.zeros_like(f_i, dtype=bool) for f_i in f]
    ess_vals[dim - 1] = tip_faces
    ess_vals[dim - 2] = tip_ridges

    def timed_solve(A, b):
        t = time.time()
        sol = sps.linalg.spsolve(A.tocsc(), b)
        print(f"ndof: {len(b):6}, Time: {time.time() - t:{fmt}}s")

        return sol

    def timed_solve_cohom(A, b):
        t = time.time()
        sol = np.linalg.solve(A, b)
        print(f"ndof: {len(b):6}, Time: {time.time() - t:{fmt}}s")

        return sol

    for k in range(1, dim + 1):
        print(f"k = {k}")

        A = sps.bmat(
            [
                [M[k - 1], -MD[k - 1].T, None],
                [MD[k - 1], S[k], -MC[k]],
                [None, MC[k].T, None],
            ]
        )
        b = np.hstack((M[k - 1] @ f[k - 1], M[k] @ f[k], CMC[k] @ e[k]))
        LS = pg.LinearSystem(A, b)

        ess_bc = np.concatenate(
            (ess_vals[k - 1], ess_vals[k], np.zeros_like(e[k], dtype=bool))
        )
        LS.flag_ess_bc(ess_bc, np.zeros_like(ess_bc))

        vu = LS.solve(solver=timed_solve)

        v_true, u_true, _ = np.split(
            vu, [M[k - 1].shape[0], M[k - 1].shape[0] + M[k].shape[0]]
        )

        # Step 1
        v = poin.solve_subproblem(k - 1, S[k - 1], MD[k - 1].T @ f[k], timed_solve)

        # Step 2
        if k >= 2:
            w_v = poin.solve_subproblem(
                k - 2, S[k - 2], MD[k - 2].T @ (f[k - 1] - v), timed_solve
            )
            v += D[k - 2] @ w_v

        else:  # k = 1, so v is a 0-form
            print(f"ndof: {0:5}, Time: {0:{fmt}}s")

        # Step 3
        varphi_coeff = timed_solve_cohom(CMC[k - 1], MC[k - 1].T @ (f[k - 1] - v))
        v += cohom_basis[k - 1] @ varphi_coeff

        assert np.allclose(v_true, v)

        # Step 4
        phi_coeff = timed_solve_cohom(CMC[k], MC[k].T @ (-f[k]))
        phi_f = cohom_basis[k] @ phi_coeff

        # Step 5
        u_hat = poin.solve_subproblem(
            k, S[k], M[k] @ (f[k] + phi_f) - MD[k - 1] @ (v), timed_solve
        )
        if k == dim:
            print(f"ndof: {0:5}, Time: {0:{fmt}}s")

        # Step 6
        v_u = poin.solve_subproblem(
            k - 1,
            S[k - 1],
            M[k - 1] @ (v - f[k - 1]) - MD[k - 1].T @ u_hat,
            timed_solve,
        )

        # Step 7
        phi_coeff = timed_solve_cohom(CMC[k], CMC[k] @ e[k] - MC[k].T @ u_hat)
        phi_u = cohom_basis[k] @ phi_coeff

        u = u_hat + D[k - 1] @ v_u + phi_u
        assert np.allclose(u_true, u)

    sys.stdout = sys.__stdout__  # restore stdout

    output = buffer.getvalue()
    print(output)

    return post_process_table(output, dim)


def post_process_table(output, dim):

    # post-processing as a table
    lines = output.splitlines()
    table = [line.split() for line in lines]

    indices = [0, 1, 10]
    if dim == 3:
        indices.append(19)
    keep = np.ones(len(table), dtype=bool)
    keep[indices] = False

    table = [table[k] for k in np.where(keep)[0]]

    table = np.array(table).astype(object)

    # Remove commas and s
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            table[i, j] = table[i, j][:-1]

    table = np.hstack(np.split(table, dim))
    table = np.roll(table, -1, axis=0)

    table = table[:, 1::2]

    # sums
    sums = table[:-1].copy()
    sums = np.sum(sums.astype(float), axis=0)
    sums[::2] = [f"{i:5}" for i in sums[::2]]
    sums[1::2] = [f"{i:5.2e}" for i in sums[1::2]]

    table = np.vstack((table, table[-1]))
    table[-2] = sums

    times = table[-2:].copy()

    for i in range(table.shape[0]):
        for j in range(0, table.shape[1], 2):
            ndof = int(float(table[i, j]))
            table[i, j] = "\\num{ " + f"{ndof:5}" + "}"
            if ndof:
                table[i, j + 1] = f"{float(table[i, j + 1]):5.2e}s"
            else:
                table[i, j + 1] = "     -"

    str_table = []
    for row in table:
        str_table.append(" & ".join(row))

    output = " \\\\ \n".join(str_table)
    print("Table:")
    print(output)

    return times


def test_Poincare_constants(dim=2, donut=False):
    N_list = generate_N_list_Poincare(dim)

    # if donut:
    #     N_list = N_list * 2

    table = np.zeros((len(N_list), dim + 1))

    for N_i, N in enumerate(N_list):
        print(f"N = {N}")
        if not donut:
            mdg = pg.unit_grid(dim, 1 / N)
        else:
            GEO = """//+
            SetFactory("OpenCASCADE");
            Torus(1) = {{0.5, 0.5, 0.5, 0.3, 0.1, 2 * Pi}};
            //+
            Mesh.CharacteristicLengthMax = {lc};

            Box(2) = {{0, 0, 0, 1, 1, 1}};
            //+
            Delete {{
            Volume{{1}}; Volume{{2}};
            }}
            //+
            Surface Loop(3) = {{7, 2, 4, 6, 3, 5}};
            //+
            Surface Loop(4) = {{1}};
            //+
            Volume(1) = {{3, 4}};
            //+
            Physical Volume("1", 15) = {{1}};
            """

            # ---- change here ----
            lc_max = 1 / N
            # ---------------------

            with open("torus_box.geo", "w") as f:
                f.write(GEO.format(lc=lc_max))
            assert dim == 3

            mdg = pp.fracs.fracture_importer.dfm_from_gmsh(Path("torus_box.geo"), 3)

        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        poin = pg.Poincare(mdg)

        table[N_i, 0] = np.mean(poin.top_sd.cell_diameters())

        for k in range(dim):
            R = pg.create_restriction(poin.bar_spaces[k])
            M = R @ mass_matrix(mdg, dim - k, None) @ R.T
            S = R @ stiff_matrix(mdg, dim - k, None) @ R.T

            if False:  # Remove the constants
                # cohom = poin.compute_basis_harmonic_forms(k)
                # M_full = mass_matrix(mdg, dim - k, None)

                # def M_op(v):
                #     M_cohom = cohom.T @ M_full @ cohom
                #     v_proj = (
                #         R @ cohom @ np.linalg.solve(M_cohom, cohom.T @ M_full @ R.T @ v)
                #     )
                #     return M @ (v - v_proj)

                # M2 = sps.linalg.LinearOperator(matvec=M_op, shape=M.shape)
                # labda = sps.linalg.eigsh(
                #     M2, 1, M=S, which="LM", tol=1e-6, return_eigenvectors=False
                # )
                pass

            else:
                labda = sps.linalg.eigsh(
                    M, 1, M=S, which="LM", tol=1e-6, return_eigenvectors=False
                )

            table[N_i, k + 1] = np.sqrt(labda[0])

    true_constants = 1 / (np.pi * np.arange(1, dim + 1))
    if donut:
        true_constants *= 0

    ans1 = np.abs(table[-2, 1:] - true_constants)
    ans2 = np.abs(table[-1, 1:] - true_constants)
    h1, h2 = table[-2:, 0]

    rates = np.log(ans1 / ans2) / np.log(h1 / h2)
    with np.printoptions(formatter={"float": "{: 0.3f}".format}):
        print(f"Rates = {rates}")

    with np.printoptions(formatter={"float": "{: 0.3f}".format}):
        print(table)

    with np.printoptions(formatter={"float": "{: 1.2e}".format}):
        print("Mesh sizes = ", table[:, 0])


def test_aux_precond(dim=2, k=1, solver="direct"):
    print(f"n = {dim}, k = {k}")

    N_list = generate_N_list(dim)
    alpha_list = np.power(10.0, np.arange(-4, 1))

    cond_table = np.zeros((len(N_list), len(alpha_list) + 1))
    iter_table = np.zeros((len(N_list), len(alpha_list)), dtype=int)

    for N_i, N in enumerate(N_list):
        calc_cond = False

        sd = pg.unit_grid(dim, 1 / N, as_mdg=False)
        mdg = pg.as_mdg(sd)

        pg.convert_from_pp(mdg)
        mdg.compute_geometry()

        poin = pg.Poincare(mdg)

        cond_table[N_i, 0] = np.mean(sd.cell_diameters())

        f = generate_sin_source(mdg)

        # assemble matrices
        M = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
        S = [stiff_matrix(mdg, dim - k, None) for k in range(dim + 1)]
        D = [diff(mdg, dim - k) for k in range(dim)]

        # Multiply right-hand side
        f = [M[k] @ f[k] for k in range(dim + 1)]

        def nonlocal_iterate(arr):
            nonlocal iters
            iters += 1

        # Define preconditioners
        match solver:
            case "direct":

                def precond_base(f, a_squared, *, poin=poin, k=k, S=S, D=D):
                    LS1 = pg.LinearSystem(S[k], f)
                    LS1.flag_ess_bc(
                        ~poin.bar_spaces[k], np.zeros_like(poin.bar_spaces[k])
                    )
                    first_term = LS1.solve()

                    LS2 = pg.LinearSystem(a_squared * S[k - 1], D[k - 1].T @ f)
                    LS2.flag_ess_bc(
                        ~poin.bar_spaces[k - 1], np.zeros_like(poin.bar_spaces[k - 1])
                    )
                    second_term = D[k - 1] @ LS2.solve()

                    return first_term + second_term

            case "FSP-AMG":
                AMG1 = compute_AMG_solver(S[k], poin.bar_spaces[k])
                AMG2 = compute_AMG_solver(S[k - 1], poin.bar_spaces[k - 1])

                def precond_base(
                    f, a_squared, *, poin=poin, k=k, S=S, D=D, AMG1=AMG1, AMG2=AMG2
                ):
                    LS1 = pg.LinearSystem(S[k], f)
                    LS1.flag_ess_bc(
                        ~poin.bar_spaces[k], np.zeros_like(poin.bar_spaces[k])
                    )
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

        # print(f"N = {N}, h = {sd.cell_diameters().mean():.2e}")
        for alpha_i, alpha in enumerate(alpha_list):
            a_squared = alpha**2
            A = a_squared * M[k] + S[k]

            iters = 0

            if solver == "AMG":
                indices = np.ones(S[k].shape[0], dtype=bool)
                AMG3 = compute_AMG_solver(A, indices)

                def precond(f, *, AMG3=AMG3):
                    return AMG3(0, f)

            else:

                def precond(f, *, precond_base=precond_base, a_squared=a_squared):
                    return precond_base(f, a_squared)

            P = sps.linalg.LinearOperator(matvec=precond, shape=A.shape)

            _x, _info = sps.linalg.cg(
                A, f[k], maxiter=750, M=P, callback=nonlocal_iterate
            )
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

    print(solver)
    print(iter_table)


def compute_AMG_solver(A, bar_space):
    LS = pg.LinearSystem(A)
    LS.flag_ess_bc(~bar_space, np.zeros_like(bar_space))
    A_red = sps.csr_matrix(LS.reduce_system()[0])

    A_red.indices = A_red.indices.astype(np.int32)
    A_red.indptr = A_red.indptr.astype(np.int32)

    import pyamg

    AMG = pyamg.smoothed_aggregation_solver(A_red)
    AMG = AMG.aspreconditioner()

    def solver(_, f):
        return AMG(f)

    return solver


def plot_trees():
    mdg = pg.unit_grid(2, 1 / 5)
    mdg.compute_geometry()

    plotter.visualize_2d(
        mdg,
        "figures/all_cotree.pdf",
        draw_grid=True,
        draw_tree=True,
        draw_cotree=False,
        start_color="blue",
    )
    plotter.visualize_2d(
        mdg,
        "figures/all_tree.pdf",
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

    plotter.visualize_2d(
        mdg,
        "figures/mdg_grid.svg",
        draw_grid=True,
        draw_tree=False,
        draw_cotree=False,
    )

    plotter.visualize_2d(
        mdg,
        "figures/mdg_cotree.svg",
        draw_grid=True,
        draw_tree=True,
        draw_cotree=False,
        start_color="blue",
    )
    plotter.visualize_2d(
        mdg,
        "figures/mdg_tree.svg",
        draw_grid=False,
        draw_tree=True,
        draw_cotree=True,
        start_color="blue",
    )


def test_properties_holes_2D():
    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(
        Path(dirname + "/geo_files/two_holes_2D.geo"), 2
    )

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    test_properties(mdg)

    # test_solver(mdg)


def test_properties_one_hole_2D():
    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(
        Path(dirname + "/geo_files/one_hole_2D.geo"), 2
    )

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    test_properties(mdg)

    test_solver(mdg)


def test_properties_holes_3D():
    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(
        Path(dirname + "/geo_files/missing_donut.geo"),
        # dirname + "/geo_files/one_hole_3D.geo",
        # Path(dirname + "/geo_files/two_holes_3D.geo"),
        3,
        # dirname + "/geo_files/two_holes_3D.geo", 3
    )

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    # test_properties(mdg)
    test_solver(mdg)


def plot_tree_with_holes():
    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(
        Path(dirname + "/geo_files/one_hole_2D.geo"),
        # Path(dirname + "/geo_files/two_holes_2D.geo"),
        2,
    )

    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    plotter.visualize_2d(
        mdg,
        "figures/holes_cotree.pdf",
        draw_grid=True,
        draw_tree=True,
        draw_cotree=False,
        start_color="blue",
    )
    plotter.visualize_2d(
        mdg,
        "figures/holes_trees.pdf",
        draw_grid=False,
        draw_tree=True,
        draw_cotree=True,
        start_color="blue",
    )


def test_viz_cohom():
    mdg = pp.fracs.fracture_importer.dfm_from_gmsh(
        Path(dirname + "/geo_files/missing_donut.geo"), 3
    )
    pg.convert_from_pp(mdg)
    mdg.compute_geometry()

    data_list = visualize_cohomology(mdg, 1)
    data_list = visualize_cohomology(mdg, 2)

    exporter = pp.Exporter(mdg, "cohomology")
    exporter.write_vtu(data_list)


if __name__ == "__main__":
    # test_properties_one_hole_2D()
    # test_properties_holes_3D()
    # test_solver_cube()
    # test_scaling_solver()
    # plot_scaling()

    # plot_trees()
    plot_tree_with_holes()
    # plot_trees_mdg()
    # test_properties_holes_2D()
    # test_viz_cohom()

    # test_properties_cube()
    # test_properties_mdg(5)
    # test_properties_seven()
    # print("Testing decomposition and co-chain properties")
    # for dim in [2, 3]:
    #     test_properties_cube(dim=dim)

    #     print("Solving the Hodge-Laplace problem")

    # print("Computing the Poincaré constants")
    # for dim in [2, 3]:
    #     test_Poincare_constants(dim)

    # print("Computing the Poincaré constants: donut")
    # test_Poincare_constants(3, True)

    # print("Preconditioning the projection problem")
    # for dim in [2, 3]:
    #     for k in range(1, dim):
    #         # for solver in ["direct", "FSP-AMG"]:
    #         for solver in ["AMG"]:
    #             test_aux_precond(dim, k, solver)

    # print("Preconditioning the projection problem with AMG")
    # test_aux_precond(3, 1, False)
