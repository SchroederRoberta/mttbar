# coding: utf-8

"""
Plot function for drawing ROC curve
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_util import remove_residual_axis

from mtt.plotting.plot_all import plot_all


hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")


def plot_roc_curve(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    # variable_insts: list[od.Variable],
    style_config: dict | None = None,
    # hide_errors: bool | None = None,
    # variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Given a dictionary of *hists* with identical binning, compute the corresponding ROC curve
    showing the background rejection as a function of the signal efficiency.

    *hists* should contain the keys *signal* and *background*, which should correspond to the
    distribution of the discriminating variable for signal and background events, respectively.
    """

    # remove residual `shift` axis
    remove_residual_axis(hists, "shift")

    if "signal" not in hists or "background" not in hists:
        hists_keys_str = ", ".join(hists)
        print(
            f"WARNING: `hists` should contain the keys 'signal' and 'background', got: {hists_keys_str}"
        )

    # plot config with a single entry for drawing the ROC curve
    plot_config = {
        "roc_curve": {
            "method": "draw_roc_curve",
            "hist": hists,
            # "kwargs": {},
        },
    }

    # style config for setting axis ranges, labels, etc.
    default_style_config = {
        "ax_cfg": {
            "xlim": (0, 1),
            "ylim": (0, 1),
            "xlabel": "Signal efficiency ($\\varepsilon$)",
            "ylabel": "1 $-$ background rejection",
            "xscale": "linear",
            "yscale": "linear",
        },
        "legend_cfg": {},
        "annotate_cfg": {
            "text": category_inst.label,
        },
        "cms_label_cfg": {
            # "lumi": config_inst.x.luminosity.get("nominal") / 1000,  # pb -> fb
        },
    }
    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(
        plot_config,
        style_config,
        skip_ratio=True,
        **kwargs,
    )
