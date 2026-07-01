# Standard libraries
from pathlib import Path

# Local modules

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

def nice_scatter_plotter(
    data_x=None, data_y=None, data_y_errors=None,
    data_x_line=None, data_y_line=None,
    label_x="", label_y="", label_title="",
    save=False, label_save_title="", marker="-x",
    labels=None, labels_line=None,
    log_x_scale=False, log_y_scale=False,
    same_color=False, square = False
):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
        'lines.markersize': 8,
        'lines.linewidth': 1.5
    })
    if square:
        plt.figure(figsize=(6, 6), dpi=100)
    else:
        plt.figure(figsize=(10, 6), dpi=100)

    # Safer defaults
    if data_x is None:
        data_x = []
    if data_y is None:
        data_y = []
    if data_x_line is None:
        data_x_line = []
    if data_y_line is None:
        data_y_line = []
    if labels is None:
        labels = []
    if labels_line is None:
        labels_line = []

    def is_1d_numeric_sequence(obj):
        """True for [1,2,3] or np.array([1,2,3]), False for [[...],[...]]."""
        if obj is None or len(obj) == 0:
            return False
        first = obj[0]
        return not isinstance(first, (list, tuple, np.ndarray))

    def normalize_y(y):
        """Ensure y is always a list of series."""
        if y is None or len(y) == 0:
            return []
        if is_1d_numeric_sequence(y):
            return [list(y)]
        return [list(series) for series in y]

    def normalize_x(x, y):
        """
        Return x as a list of x-series matching y-series.
        Cases:
        - x missing: generate default x for each y
        - x is one 1D list: reuse for all y-series
        - x is list of lists: use one x-series per y-series
        """
        if len(y) == 0:
            return []

        if x is None or len(x) == 0:
            return [list(range(1, len(series) + 1)) for series in y]

        if is_1d_numeric_sequence(x):
            x_single = list(x)
            return [x_single for _ in y]

        x_out = [list(series) for series in x]

        if len(x_out) != len(y):
            raise ValueError(
                f"Number of x series ({len(x_out)}) must match number of y series ({len(y)})"
            )

        return x_out

    # Normalize y data
    data_y = normalize_y(data_y)
    data_y_line = normalize_y(data_y_line)

    # Normalize error bars
    if data_y_errors is not None:
        data_y_errors = normalize_y(data_y_errors)
        if len(data_y_errors) != len(data_y):
            raise ValueError(
                f"Number of error-bar series ({len(data_y_errors)}) must match number of y series ({len(data_y)})"
            )

    # Normalize x data
    data_x = normalize_x(data_x, data_y)
    data_x_line = normalize_x(data_x_line, data_y_line)

    # Check lengths match pairwise
    for i, (xvals, yvals) in enumerate(zip(data_x, data_y)):
        if len(xvals) != len(yvals):
            raise ValueError(
                f"Scatter series {i}: x and y lengths do not match ({len(xvals)} vs {len(yvals)})"
            )
        if data_y_errors is not None and len(data_y_errors[i]) != len(yvals):
            raise ValueError(
                f"Scatter series {i}: y and y_errors lengths do not match ({len(yvals)} vs {len(data_y_errors[i])})"
            )

    for i, (xvals, yvals) in enumerate(zip(data_x_line, data_y_line)):
        if len(xvals) != len(yvals):
            raise ValueError(
                f"Line series {i}: x and y lengths do not match ({len(xvals)} vs {len(yvals)})"
            )

    # Optional safety check if matching colors by index
    if same_color and len(data_y_line) > len(data_y):
        raise ValueError(
            f"same_color=True requires at least as many scatter series as line series "
            f"({len(data_y)} scatter vs {len(data_y_line)} line)"
        )

    # Plot settings
    marker_style = 'o'
    ms = 6
    mew = 0

    # Store scatter colors so matching lines can reuse them
    scatter_colors = []

    # Scatter / errorbar series
    for i, y_data in enumerate(data_y):
        x_data = data_x[i]
        label = labels[i] if i < len(labels) else None

        if data_y_errors is not None:
            cont = plt.errorbar(
                x_data, y_data,
                yerr=data_y_errors[i],
                fmt=marker_style,
                linestyle='none',
                markersize=ms,
                markeredgewidth=mew,
                elinewidth=1,
                capsize=0,
                label=label
            )
            scatter_colors.append(cont[0].get_color())
        else:
            line_obj, = plt.plot(
                x_data, y_data,
                marker=marker_style,
                linestyle='none',
                markersize=ms,
                markeredgewidth=mew,
                label=label
            )
            scatter_colors.append(line_obj.get_color())

    # Line series
    for i, y_data_line in enumerate(data_y_line):
        x_line = data_x_line[i]
        label = labels_line[i] if i < len(labels_line) else None

        if same_color and i < len(scatter_colors):
            plt.plot(x_line, y_data_line, label=label, color=scatter_colors[i])
        else:
            plt.plot(x_line, y_data_line, label=label)

    if log_x_scale:
        plt.xscale('log')
    if log_y_scale:
        plt.yscale('log')

    plt.xlabel(label_x, fontsize=14, fontname='Times New Roman')
    plt.ylabel(label_y, fontsize=14, fontname='Times New Roman')
    plt.title(label_title, fontsize=16, fontname='Times New Roman')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    if len(labels) > 0 or len(labels_line) > 0:
        plt.legend()

    plt.tight_layout()

    if save:
        save_dir = Path(__file__).resolve().parent.parent / "scatter_saved_plots"
        save_dir.mkdir(parents = True, exist_ok = True)
        plt.savefig(save_dir / (label_save_title + ".pdf"), bbox_inches = 'tight')
        print("Plot saved")
    else:
        print("Plot not saved")

    plt.show()

