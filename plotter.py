import pygeon as pg
import porepy as pp
import numpy as np
import scipy.sparse as sps


def visualize_2d(self, mdg: pg.MixedDimensionalGrid, fig_name=None, **kwargs):
    """
    Create a graphical illustration of the spanning tree superimposed on the grid.

    Args:
        mdg (pg.MixedDimensionalGrid) The object representing the grid.
        fig_name (Optional[str], optional). The name of the figure file to save the
            visualization.

    Optional Args
        draw_grid (bool): Plot the grid
        draw_tree (bool): Plot the tree spanning the cells
        draw_cotree (bool): Plot the tree spanning the nodes
        start_color (str): Color of the "starting" cells, next to the boundary
    """
    import matplotlib.pyplot as plt
    import networkx as nx  # type: ignore

    assert mdg.dim_max() == 2
    sd_top = mdg.subdomains()[0]

    draw_grid = kwargs.get("draw_grid", True)
    draw_tree = kwargs.get("draw_tree", True)
    draw_complement = kwargs.get("draw_cotree", False)
    start_color = kwargs.get("start_color", "green")

    fig_num = 1

    # Draw grid
    if draw_grid:
        pp.plot_grid(
            mdg, alpha=0, fig_num=fig_num, plot_2d=True, if_plot=False, title=""
        )

    # Define the figure and axes
    fig = plt.figure(fig_num)

    # The grid is drawn by PorePy if desired
    if draw_grid:
        pp_ax = fig.gca()
        pp_ax.set_xlabel("")
        pp_ax.set_ylabel("")
        pp_ax.set_aspect("equal")
        plt.tick_params(
            left=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
        )
        ax = fig.add_subplot(111)
        ax.set_xlim(pp_ax.get_xlim())
        ax.set_ylim(pp_ax.get_ylim())

    # If there is no PorePy grid plot, we create our own axes
    else:
        ax = fig.gca()

        min_coord = np.min(sd_top.nodes, axis=1)
        max_coord = np.max(sd_top.nodes, axis=1)

        ax.set_xlim((min_coord[0], max_coord[0]))
        ax.set_ylim((min_coord[1], max_coord[1]))

    ax.set_aspect("equal")

    # Draw the tree that spans all cells
    if draw_tree:
        graph = nx.from_scipy_sparse_array(self.tree)
        cell_centers = np.hstack([sd.cell_centers for sd in mdg.subdomains()])
        node_color = ["blue"] * cell_centers.shape[1]
        for sc in self.starting_cells:
            node_color[sc] = start_color

        nx.draw(
            graph,
            cell_centers[: mdg.dim_max(), :].T,
            node_color=node_color,
            node_size=40,
            edge_color="red",
            ax=ax,
        )

        # Add connections from the roots to the starting faces
        num_bdry = len(self.starting_faces)
        bdry_graph = sps.diags_array(  # type: ignore[call-overload]
            np.ones(num_bdry),
            num_bdry,
            shape=(2 * num_bdry, 2 * num_bdry),
        )
        graph = nx.from_scipy_sparse_array(bdry_graph)

        face_centers = np.hstack([sd.face_centers for sd in mdg.subdomains()])
        cell_centers = np.hstack([sd.cell_centers for sd in mdg.subdomains()])
        node_centers = np.hstack(
            (
                face_centers[: mdg.dim_max(), self.starting_faces],
                cell_centers[: mdg.dim_max(), self.starting_cells],
            )
        ).T

        nx.draw(
            graph,
            node_centers,
            node_size=0,
            edge_color="red",
            ax=ax,
        )

    # Draw the tree that spans all nodes
    if draw_complement:
        curl = pg.curl(mdg)[~self.flagged_faces, :]
        incidence = curl.T @ curl
        incidence -= sps.triu(incidence)

        graph = nx.from_scipy_sparse_array(incidence)

        node_color = ["black"] * sd_top.nodes.shape[1]

        nx.draw(
            graph,
            sd_top.nodes[: mdg.dim_max(), :].T,
            node_color=node_color,
            node_size=30,
            edge_color="purple",
            width=1.5,
            ax=ax,
        )

    plt.draw()
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.1)

    plt.close()
