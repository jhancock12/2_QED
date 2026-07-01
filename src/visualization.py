# Standard libraries
from pathlib import Path

# Local modules
from lattice_class import Lattice

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch


def _lattice_plot_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 25,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
        'lines.linewidth': 1.5
    })


def _to_real_if_close(x):
    if isinstance(x, complex):
        if abs(x.imag) < 1e-10:
            return float(x.real)
        return x
    return x


def _fmt_value(x, decimals = 3):
    x = _to_real_if_close(x)
    if isinstance(x, complex):
        return f"{x.real:.{decimals}f}+{x.imag:.{decimals}f}i"
    return f"{x:.{decimals}f}"


def _blend_with_white(color, amount = 0.3):
    r, g, b, a = mcolors.to_rgba(color)
    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)
    return (r, g, b, a)


def _blend_with_black(color, amount = 0.2):
    r, g, b, a = mcolors.to_rgba(color)
    r = r * (1 - amount)
    g = g * (1 - amount)
    b = b * (1 - amount)
    return (r, g, b, a)


def _draw_shaded_node(ax, x, y, radius, base_color, edgecolor = 'black', linewidth = 1.0, zorder = 3):
    outer_color = _blend_with_black(base_color, 0.10)
    ax.add_patch(Circle(
        (x, y), radius,
        facecolor = outer_color,
        edgecolor = edgecolor,
        linewidth = linewidth,
        zorder = zorder
    ))

    n_layers = 12
    for i in range(n_layers, 0, -1):
        frac = i / n_layers
        rr = radius * (0.78 * frac)
        shift = radius * 0.10 * frac
        color = _blend_with_white(base_color, 0.08 + 0.28 * frac)

        ax.add_patch(Circle(
            (x - shift, y + shift),
            rr,
            facecolor = color,
            edgecolor = 'none',
            zorder = zorder + 0.001 * i
        ))


def _lattice_final_formatting(ax, site_positions, label_title = None, xpad = 0.3, ypad = 0.4):
    if label_title:
        ax.set_title(label_title, fontsize = 16, fontname = 'Times New Roman')

    ax.set_aspect('equal')

    xs = [p[0] for p in site_positions.values()]
    ys = [p[1] for p in site_positions.values()]

    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    ax.axis('off')
    plt.tight_layout()


def geometry_from_lattice_scaled(lattice : Lattice, sx = 1.6, sy = 1.2):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(sx, int | float): raise TypeError(f"sx must be a number, you have entered a {type(sx)}")
    if not isinstance(sy, int | float): raise TypeError(f"sy must be a number, you have entered a {type(sy)}")

    site_positions = {
        lattice.labels[n]: (sx * lattice.labels[n][0], sy * lattice.labels[n][1])
        for n in lattice.labels
    }

    link_pairs = {}
    for link in lattice.links_list:
        (x, y), direction = link
        start_site = (x, y)

        if direction == 1:
            end_site = (x + 1, y)
        elif direction == 2:
            end_site = (x, y + 1)
        else:
            raise ValueError(f"Unknown direction {direction} for link {link}")

        link_pairs[link] = (start_site, end_site)

    return site_positions, link_pairs


def lattice_observable_plotter(
    lattice : Lattice,
    results_dict : dict,
    site_positions = None,
    link_pairs = None,
    link_values_key = 'electric_field_dict',
    charge_values_key = 'charge_field_dict',
    label_title = None,
    save = False,
    label_save_title = "lattice_observable_plot",
    figsize = (10, 4),
    dpi = 140,
    node_radius = 0.18,
    link_label_offset = 0.15,
    background_color = "white",
    show_link_names = False,
    charge_decimals = 3,
    field_decimals = 3,
    dynamic_link_color = 'black',
    nondynamic_link_color = '#c4c4c4',
    dynamic_link_width = 1.8,
    nondynamic_link_width = 1.2,
    arrow_scale = 14,
    mid_arrow_fraction = 0.24,
    charge_gamma = 2.0,
    ring_linewidth = 2.4,
    node_text_fontsize = 11,
    link_text_fontsize = 12,
    sx = 1.6,
    sy = 1.2
):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(results_dict, dict): raise TypeError(f"results_dict must be a dictionary, you have entered a {type(results_dict)}")
    if not isinstance(save, bool): raise TypeError(f"save must be a bool, you have entered a {type(save)}")

    if site_positions is None or link_pairs is None:
        site_positions, link_pairs = geometry_from_lattice_scaled(lattice, sx = sx, sy = sy)

    _lattice_plot_style()

    link_values = results_dict.get(link_values_key, {})
    charge_values = results_dict.get(charge_values_key, {})
    dynamical_links = set(lattice.dynamical_links_list)

    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    real_charge_values = [_to_real_if_close(v) for v in charge_values.values()]
    real_charge_values = [v.real if isinstance(v, complex) else v for v in real_charge_values]

    qmax = max([abs(v) for v in real_charge_values], default = 1.0)
    if qmax < 0.1:
        qmax = 0.5

    charge_cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_grey_red",
        ["#5b6cff", "#d9d9d9", "#ff6b6b"]
    )

    def node_color_from_charge(q):
        q = _to_real_if_close(q)
        if isinstance(q, complex):
            q = q.real

        x = q / qmax
        x = max(-1.0, min(1.0, x))

        x_scaled = np.sign(x) * (abs(x) ** charge_gamma)

        t = 0.5 * (x_scaled + 1.0)
        return charge_cmap(t)

    # Draw links first
    for link_name, (site_a, site_b) in link_pairs.items():
        if site_a not in site_positions or site_b not in site_positions:
            continue

        x1, y1 = site_positions[site_a]
        x2, y2 = site_positions[site_b]

        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            continue

        ux = dx / length
        uy = dy / length

        is_dynamic = link_name in dynamical_links
        link_color = dynamic_link_color if is_dynamic else nondynamic_link_color
        link_width = dynamic_link_width if is_dynamic else nondynamic_link_width

        ax.plot(
            [x1, x2], [y1, y2],
            color = link_color,
            linewidth = link_width,
            zorder = 1
        )

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        half_arrow_len = 0.5 * mid_arrow_fraction * length
        ax1 = mx - half_arrow_len * ux
        ay1 = my - half_arrow_len * uy
        ax2 = mx + half_arrow_len * ux
        ay2 = my + half_arrow_len * uy

        mid_arrow = FancyArrowPatch(
            (ax1, ay1), (ax2, ay2),
            arrowstyle = '-|>',
            mutation_scale = arrow_scale,
            linewidth = link_width,
            color = link_color,
            shrinkA = 0,
            shrinkB = 0,
            zorder = 2
        )
        ax.add_patch(mid_arrow)

        if link_name in link_values:
            px = -uy
            py = ux

            if abs(ux) > abs(uy):
                text_offset = link_label_offset
            else:
                text_offset = link_label_offset - 0.03

            tx = mx + (text_offset + 0.08) * px
            ty = my + text_offset * py

            label = _fmt_value(link_values[link_name], field_decimals)
            if show_link_names:
                label = f"{link_name}: {label}"

            ax.text(
                tx, ty,
                label,
                ha = 'center',
                va = 'center',
                fontsize = link_text_fontsize,
                color = 'black',
                zorder = 5
            )

    # Draw nodes on top
    for site_name, (x, y) in site_positions.items():
        q = charge_values.get(site_name, 0.0)
        base_color = node_color_from_charge(q)

        edgecolor = 'none'
        linewidth = 0.0

        if site_name == lattice.charge_site:
            edgecolor = "#5b6cff"
            linewidth = ring_linewidth
        elif site_name == lattice.anticharge_site:
            edgecolor = "#ff6b6b"
            linewidth = ring_linewidth

        _draw_shaded_node(
            ax, x, y, node_radius,
            base_color = base_color,
            edgecolor = edgecolor,
            linewidth = linewidth,
            zorder = 3
        )

        ax.text(
            x, y,
            _fmt_value(q, charge_decimals),
            ha = 'center',
            va = 'center',
            fontsize = node_text_fontsize,
            color = 'black',
            zorder = 6
        )

    _lattice_final_formatting(ax, site_positions, label_title = label_title)

    if save:
        save_dir = Path(__file__).resolve().parent.parent / "lattice_saved_plots"
        save_dir.mkdir(parents = True, exist_ok = True)
        plt.savefig(save_dir / (label_save_title + ".pdf"), bbox_inches = 'tight')
        print("Plot saved")
    else:
        print("Plot not saved")

    return fig, ax


def lattice_geometry_plotter(
    lattice : Lattice,
    site_positions = None,
    link_pairs = None,
    label_title = None,
    save = False,
    label_save_title = "lattice_geometry_plot",
    figsize = (10, 4),
    dpi = 140,
    node_radius = 0.18,
    background_color = "white",
    dynamic_link_color = 'black',
    nondynamic_link_color = '#c4c4c4',
    dynamic_link_width = 1.8,
    nondynamic_link_width = 1.2,
    arrow_scale = 14,
    mid_arrow_fraction = 0.24,
    node_color = "#d9d9d9",
    node_edgecolor = "black",
    node_linewidth = 1.0,
    sx = 1.6,
    sy = 1.2
):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(lattice)}")
    if not isinstance(save, bool): raise TypeError(f"save must be a bool, you have entered a {type(save)}")

    if site_positions is None or link_pairs is None:
        site_positions, link_pairs = geometry_from_lattice_scaled(lattice, sx = sx, sy = sy)

    _lattice_plot_style()

    dynamical_links = set(lattice.dynamical_links_list)

    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Draw links first
    for link_name, (site_a, site_b) in link_pairs.items():
        if site_a not in site_positions or site_b not in site_positions:
            continue

        x1, y1 = site_positions[site_a]
        x2, y2 = site_positions[site_b]

        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            continue

        ux = dx / length
        uy = dy / length

        is_dynamic = link_name in dynamical_links
        link_color = dynamic_link_color if is_dynamic else nondynamic_link_color
        link_width = dynamic_link_width if is_dynamic else nondynamic_link_width

        ax.plot(
            [x1, x2], [y1, y2],
            color = link_color,
            linewidth = link_width,
            zorder = 1
        )

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        half_arrow_len = 0.5 * mid_arrow_fraction * length
        ax1 = mx - half_arrow_len * ux
        ay1 = my - half_arrow_len * uy
        ax2 = mx + half_arrow_len * ux
        ay2 = my + half_arrow_len * uy

        mid_arrow = FancyArrowPatch(
            (ax1, ay1), (ax2, ay2),
            arrowstyle = '-|>',
            mutation_scale = arrow_scale,
            linewidth = link_width,
            color = link_color,
            shrinkA = 0,
            shrinkB = 0,
            zorder = 2
        )
        ax.add_patch(mid_arrow)

    # Draw plain nodes
    for site_name, (x, y) in site_positions.items():
        _draw_shaded_node(
            ax, x, y, node_radius,
            base_color = node_color,
            edgecolor = node_edgecolor,
            linewidth = node_linewidth,
            zorder = 3
        )

    _lattice_final_formatting(ax, site_positions, label_title = label_title)

    if save:
        save_dir = Path(__file__).resolve().parent.parent / "lattice_saved_plots"
        save_dir.mkdir(parents = True, exist_ok = True)
        plt.savefig(save_dir / (label_save_title + ".pdf"), bbox_inches = 'tight')
        print("Plot saved")
    else:
        print("Plot not saved")

    return fig, ax