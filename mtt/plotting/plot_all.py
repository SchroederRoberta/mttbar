# coding: utf-8

"""
Custom plot methods.
"""

from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.plotting.plot_util import get_position
from columnflow.plotting.plot_all import (
    draw_error_bands,
    draw_stack,
    draw_hist,
    draw_errorbars,
)

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")


def draw_roc_curve(
    ax: plt.Axes,
    hists: dict[hist.Hist],
    norm: float = 1.0,
    data: dict | None = None,
    **kwargs,
) -> None:


    # obtain ROC curve support points
    # from histogram bin edges
    values = np.array(list(hists.values())[0].axes[-1].edges)

    # number of total events and events that pass the discriminator
    # thresholds for the for the supplied histograms
    totals, passes = {}, {}
    pass_fractions = {}

    for key, h in hists.items():
        totals[key] = h.sum(flow=True).value
        passes[key] = np.array([
            h[:hist.loc(v)].sum().value
            for v in values
        ])
        pass_fractions[key] = (
            passes[key] / totals[key]
        )

    plot_kwargs = {
        # "linewidth": 1,
        "color": "black",
    }
    plot_kwargs.update(kwargs)

    # if data is given, fill the values/event counts
    if data is not None:
        data["discriminator_values"] = values.tolist()
        data["counts"] = {
            key: {
                "pass": passes[key].tolist(),
                "total": totals[key],
            }
            for key in hists
        }

    # obtain signal and background pass fractions 
    pass_fraction_signal = pass_fractions.get("signal", 0.0)
    pass_fraction_background = pass_fractions.get("background", 0.0)

    # broadcast to ensure shape compatibility
    pass_fraction_signal, pass_fraction_background = np.broadcast_arrays(
        pass_fraction_signal,
        pass_fraction_background,
    )

    # plot and return artist
    return ax.plot(
        pass_fraction_signal,
        pass_fraction_background,
        **plot_kwargs,
    )


def plot_all(
    plot_config: dict,
    style_config: dict,
    skip_ratio: bool = False,
    skip_legend: bool = False,
    cms_label: str = "wip",
    **kwargs,
) -> plt.Figure:
    """
    plot_config expects dictionaries with fields:
    "method": str, identical to the name of a function defined above,
    "hist": hist.Hist or hist.Stack,
    "kwargs": dict (optional),
    "ratio_kwargs": dict (optional),

    style_config expects fields (all optional):
    "ax_cfg": dict,
    "rax_cfg": dict,
    "legend_cfg": dict,
    "cms_label_cfg": dict,
    """

    # additional data to be filled by the plot methods
    # and returned to the caller
    data = {}

    plt.style.use(mplhep.style.CMS)

    rax = None
    if not skip_ratio:
        fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True)
        (ax, rax) = axs
    else:
        fig, ax = plt.subplots()
        axs = (ax,)

    for key, cfg in plot_config.items():
        if "method" not in cfg:
            raise ValueError(f"no method given in plot_cfg entry {key}")
        method = cfg["method"]

        if "hist" not in cfg:
            raise ValueError(f"no histogram(s) given in plot_cfg entry {key}")
        hist = cfg["hist"]
        kwargs = cfg.get("kwargs", {})
        plot_all.plot_methods[method](ax, hist, data=data, **kwargs)

        if not skip_ratio and "ratio_kwargs" in cfg:
            # take ratio_method if the ratio plot requires a different plotting method
            method = cfg.get("ratio_method", method)
            plot_all.plot_methods[method](rax, hist, **cfg["ratio_kwargs"])

    # axis styling
    ax_kwargs = {
        "ylabel": "Counts",
        "xlabel": "variable",
        "yscale": "linear",
    }

    # some default ylim settings based on yscale
    log_y = style_config.get("ax_cfg", {}).get("yscale", "linear") == "log"

    ax_ymin = ax.get_ylim()[1] / 10**4 if log_y else 0.00001
    ax_ymax = get_position(ax_ymin, ax.get_ylim()[1], factor=1.4, logscale=log_y)
    ax_kwargs.update({"ylim": (ax_ymin, ax_ymax)})

    # prioritize style_config ax settings
    ax_kwargs.update(style_config.get("ax_cfg", {}))

    # ax configs that can not be handled by `ax.set`
    minorxticks = ax_kwargs.pop("minorxticks", None)
    minoryticks = ax_kwargs.pop("minoryticks", None)

    ax.set(**ax_kwargs)

    if minorxticks is not None:
        ax.set_xticks(minorxticks, minor=True)
    if minoryticks is not None:
        ax.set_xticks(minoryticks, minor=True)

    if not skip_ratio:
        # hard-coded line at 1
        rax.axhline(y=1.0, linestyle="dashed", color="gray")
        rax_kwargs = {
            "ylim": (0.72, 1.28),
            "ylabel": "Ratio",
            "xlabel": "Variable",
            "yscale": "linear",
        }
        rax_kwargs.update(style_config.get("rax_cfg", {}))
        rax.set(**rax_kwargs)
        fig.align_ylabels()

    # legend
    if not skip_legend:
        # resolve legend kwargs
        legend_kwargs = {
            "ncol": 1,
            "loc": "upper right",
        }
        legend_kwargs.update(style_config.get("legend_cfg", {}))

        # retrieve the legend handles and their labels
        handles, labels = ax.get_legend_handles_labels()

        # assume all `StepPatch` objects are part of MC stack
        in_stack = [
            isinstance(handle, mpl.patches.StepPatch)
            for handle in handles
        ]

        # reverse order of entries that are part of the stack
        if any(in_stack):
            def shuffle(entries, mask):
                entries = np.array(entries, dtype=object)
                entries[mask] = entries[mask][::-1]
                return list(entries)

            handles = shuffle(handles, in_stack)
            labels = shuffle(labels, in_stack)

        # make legend using ordered handles/labels
        ax.legend(handles, labels, **legend_kwargs)

    # custom annotation
    log_x = style_config.get("ax_cfg", {}).get("xscale", "linear") == "log"
    annotate_kwargs = {
        "text": "",
        "xy": (
            get_position(*ax.get_xlim(), factor=0.05, logscale=log_x),
            get_position(*ax.get_ylim(), factor=0.95, logscale=log_y),
        ),
        "xycoords": "data",
        "color": "black",
        "fontsize": 22,
        "horizontalalignment": "left",
        "verticalalignment": "top",
    }
    annotate_kwargs.update(style_config.get("annotate_cfg", {}))
    ax.annotate(**annotate_kwargs)

    # cms label
    if cms_label != "skip":
        label_options = {
            "wip": "Work in progress",
            "pre": "Preliminary",
            "pw": "Private work",
            "sim": "Simulation",
            "simwip": "Simulation work in progress",
            "simpre": "Simulation preliminary",
            "simpw": "Simulation private work",
            "od": "OpenData",
            "odwip": "OpenData work in progress",
            "odpw": "OpenData private work",
            "public": "",
        }
        cms_label_kwargs = {
            "ax": ax,
            "llabel": label_options.get(cms_label, cms_label),
            "fontsize": 22,
            "data": False,
        }

        cms_label_kwargs.update(style_config.get("cms_label_cfg", {}))
        mplhep.cms.label(**cms_label_kwargs)

    plt.tight_layout()

    return fig, axs, data


# register available plot methods mapped to their names
plot_all.plot_methods = {
    func.__name__: func
    for func in [
        draw_error_bands,
        draw_stack,
        draw_hist,
        draw_errorbars,
        draw_roc_curve,
    ]
}
