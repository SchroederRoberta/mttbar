# coding: utf-8
"""
Custom base tasks.
"""

import luigi
import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.plotting import PlotBase
from columnflow.tasks.plotting import PlotVariablesBaseSingleShift

from mtt.tasks.base import MTTTask


class PlotROCCurveBase(
    MTTTask,
    PlotVariablesBaseSingleShift,
    PlotBase,
):
    """
    Calculate and plot the ROC curve (background rejection vs. signal efficiency) resulting
    from a cut on discriminating variables.

    Accepts multiple *datasets*, which are classed into signal or background based on the
    ``is_signal`` attribute of the underlying :py:class:`order.Datasets` instance. The choice
    of datasets considered for the ROC curve measurement may further be restricted by setting
    the *processes* parameter.
    """

    # upstream requirements
    reqs = Requirements(
        PlotVariablesBaseSingleShift.reqs,
    )

    plot_function = PlotBase.plot_function.copy(
        default="mtt.plotting.plot_roc_curve.plot_roc_curve",
        add_default_to_description=True,
    )

    def output(self):
        output = super().output()

        # add 'data' output containing ROC curve data points
        b = self.branch_data
        branch_repr = f"proc_{self.processes_repr}__cat_{b.category}__var_{b.variable}"
        output["data"] = self.target(f"data__{branch_repr}.json")

        return output

    def get_hists_key(self, dataset_inst):
        return None

    def process_hists(self, hists):
        return hists

    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import hist

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # prepare config objects
        # variable_tuple = self.variable_tuples[self.branch_data.variable]
        # variable_insts = [
        #     self.config_inst.get_variable(var_name)
        #     for var_name in variable_tuple
        # ]

        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]

        process_insts = list(map(self.config_inst.get_process, self.processes))
        leaf_process_insts = {
            leaf_proc
            for proc in process_insts
            for leaf_proc in proc.get_leaf_processes()
        }

        # histogram data, summed up for background and
        # signal processes
        hists = {}

        with self.publish_step(f"plotting ROC curve for {self.branch_data.variable} in {category_inst.name}"):
            for dataset, inp in self.input().items():
                dataset_inst = self.config_inst.get_dataset(dataset)

                # skip when the dataset does not contain any leaf process
                if not any(map(dataset_inst.has_process, leaf_process_insts)):
                    continue

                h_in = inp["collection"][0]["hists"].targets[self.branch_data.variable].load(formatter="pickle")

                # work on a copy
                h = h_in.copy()

                # axis selections
                h = h[{
                    "process": [
                        hist.loc(p.id)
                        for p in leaf_process_insts
                        if p.id in h.axes["process"]
                    ],
                    "category": [
                        hist.loc(c.id)
                        for c in leaf_category_insts
                        if c.id in h.axes["category"]
                    ],
                    "shift": [
                        hist.loc(s.id)
                        for s in plot_shifts
                        if s.id in h.axes["shift"]
                    ],
                }]

                # axis reductions
                h = h[{"process": sum, "category": sum}]

                # add the histogram
                hists_key = self.get_hists_key(dataset_inst)
                if hists_key in hists:
                    hists[hists_key] += h
                else:
                    hists[hists_key] = h

            # there should be hists to plot
            if not hists:
                raise Exception(
                    "no histograms found to plot; possible reasons:\n" +
                    "  - requested variable requires columns that were missing during histogramming\n" +
                    "  - selected --processes did not match any value on the process axis of the input histogram",
                )

            # post-process histograms
            hists = self.process_hists(hists)

            # call the plot function
            fig, axs, data = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=self.config_inst,
                category_inst=category_inst.copy_shallow(),
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")

            # save the ROC curve data
            self.output()["data"].dump(data, formatter="json")


class PlotROCCurveByDatasetTag(
    PlotROCCurveBase
):
    """
    Calculate and plot the ROC curve (background rejection vs. signal efficiency) resulting
    from a cut on discriminating variables.

    Accepts multiple *datasets*, which are classed into signal or background depending on the
    presence of a tag ``signal_tag`` of the underlying :py:class:`order.Datasets` instance. The
    choice of datasets considered for the ROC curve measurement may further be restricted by
    setting the *processes* parameter.
    """

    signal_tag = luigi.Parameter(
        description="datasets marked with this tag are considered signal, otherwise background",
    )

    def get_hists_key(self, dataset_inst):
        return (
            "signal"
            if dataset_inst.has_tag(self.signal_tag)
            else "background"
        )

    def process_hists(self, hists):
        return hists


class PlotROCCurveByVariable(
    PlotROCCurveBase
):
    """
    Calculate and plot the ROC curve (background rejection vs. signal efficiency) resulting
    from a cut on discriminating variables.

    An event is identified as a signal event based on the value of the variable ``signal_variable``.

    Accepts multiple *datasets*. The choice of datasets considered for the ROC curve measurement
    may further be restricted by setting the *processes* parameter.
    """

    signal_variable = luigi.Parameter(
        description="events where this variable is 0 (1) are considered background (signal)",
    )

    @classmethod
    def resolve_param_values(cls, params: dict) -> dict:
        params = super().resolve_param_values(params)

        # add `signal_variables` to variables for histograms
        if "variables" in params:
            if len(params["variables"]) != 1:
                raise ValueError(f"only 1 variable supported, got {len(params['variables'])}")
            discriminant_variable = params["variables"][0]
            signal_variable = params["signal_variable"]
            if not discriminant_variable.endswith(f"-{signal_variable}"):
                params["variables"] = law.util.make_tuple(f"{discriminant_variable}-{signal_variable}")

        return params

    def get_hists_key(self, dataset_inst):
        return "all"

    def process_hists(self, hists):
        """Split 'all' histogram into signal and background."""
        import hist
        hists_key = self.get_hists_key(None)
        h = hists[hists_key]
        return {
            "signal": h[{self.signal_variable: hist.loc(1)}],
            "background": h[{self.signal_variable: hist.loc(0)}],
        }
