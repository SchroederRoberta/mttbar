# coding: utf-8
"""
Custom wrapper factory
"""

from __future__ import annotations

import inspect
import itertools
import law
import order as od

from typing import Sequence

from columnflow.tasks.framework.base import (
    AnalysisTask,
    ConfigTask,
    DatasetTask,
    ShiftTask,
)
from columnflow.tasks.production import ProduceColumns

from mtt.tasks.base import MTTTask


def wrapper_factory(
    base_cls: (law.Task, MTTTask),
    require_cls: AnalysisTask,
    enable: Sequence[str],
    cls_name: str | None = None,
    attributes: dict | None = None,
    docs: str | None = None,
) -> law.task.base.Register:
    """
    Factory function creating wrapper task classes, inheriting from *base_cls* and
    :py:class:`~law.WrapperTask`, that do nothing but require multiple instances of *require_cls*.
        Unless *cls_name* is defined, the name of the created class defaults to the name of
        *require_cls* plus "Wrapper". Additional *attributes* are added as class-level members when
        given.

    The instances of *require_cls* to be required in the
    :py:meth:`~.wrapper_factory.Wrapper.requires()` method can be controlled by task parameters.
    These parameters can be enabled through the string sequence *enable*, which currently accepts:

        - ``configs``, ``skip_configs``
        - ``shifts``, ``skip_shifts``
        - ``datasets``, ``skip_datasets``
        - ``producers``

    This allows to easily build wrapper tasks that loop over (combinations of) parameters that are
    either defined in the analysis or config, which would otherwise lead to mostly redundant code.
    Example:

    .. code-block:: python

        class MyTask(DatasetTask):
            ...

        MyTaskWrapper = wrapper_factory(
            base_cls=ConfigTask,
            require_cls=MyTask,
            enable=["datasets", "skip_datasets"],
        )

        # this allows to run (e.g.)
        # law run MyTaskWrapper --datasets st_* --skip-datasets *_tbar

    When building the requirements, the full combinatorics of parameters is considered. However,
    certain conditions apply depending on enabled features. For instance, in order to use the
    "configs" feature (adding a parameter "--configs" to the created class, allowing to loop over a
    list of config instances known to an analysis), *require_cls* must be at least a
    :py:class:`ConfigTask` accepting "--config" (mind the singular form), whereas *base_cls* must
        explicitly not.
    """
    # check known features
    known_features = [
        "configs", "skip_configs",
        "shifts", "skip_shifts",
        "datasets", "skip_datasets",
        "producers",
    ]
    for feature in enable:
        if feature not in known_features:
            raise ValueError(
                f"unknown enabled feature '{feature}', known features are "
                f"'{','.join(known_features)}'",
            )

    # treat base_cls as a tuple
    base_classes = law.util.make_tuple(base_cls)
    base_cls = base_classes[0]

    # define wrapper feature flags
    has_configs = "configs" in enable
    has_skip_configs = has_configs and "skip_configs" in enable
    has_shifts = "shifts" in enable
    has_skip_shifts = has_shifts and "skip_shifts" in enable
    has_datasets = "datasets" in enable
    has_skip_datasets = has_datasets and "skip_datasets" in enable
    has_producers = "producers" in enable

    # helper to check if enabled features are compatible with required and base class
    def check_class_compatibility(name, min_require_cls, max_base_cls):
        if not issubclass(require_cls, min_require_cls):
            raise TypeError(
                f"when the '{name}' feature is enabled, require_cls must inherit from "
                f"{min_require_cls}, but {require_cls} does not",
            )
        if issubclass(base_cls, min_require_cls):
            raise TypeError(
                f"when the '{name}' feature is enabled, base_cls must not inherit from "
                f"{min_require_cls}, but {base_cls} does",
            )
        if not issubclass(max_base_cls, base_cls):
            raise TypeError(
                f"when the '{name}' feature is enabled, base_cls must be a super class of "
                f"{max_base_cls}, but {base_cls} is not",
            )

    # check classes
    if has_configs:
        check_class_compatibility("configs", ConfigTask, AnalysisTask)
    if has_shifts:
        check_class_compatibility("shifts", ShiftTask, ConfigTask)
    if has_datasets:
        check_class_compatibility("datasets", DatasetTask, ShiftTask)
    if has_producers:
        check_class_compatibility("producers", ProduceColumns, ShiftTask)

    # create the class
    class Wrapper(*base_classes, law.WrapperTask):

        if has_configs:
            configs = law.CSVParameter(
                default=("*",),
                description="names or name patterns of configs to use; can also be the key of a "
                "mapping defined in the 'config_groups' auxiliary data of the analysis; default: "
                "('*',)",
                brace_expand=True,
            )
        if has_skip_configs:
            skip_configs = law.CSVParameter(
                default=(),
                description="names or name patterns of configs to skip after evaluating --configs; "
                "can also be the key of a mapping defined in the 'config_groups' auxiliary data "
                "of the analysis; empty default",
                brace_expand=True,
            )
        if has_datasets:
            datasets = law.CSVParameter(
                default=("*",),
                description="names or name patterns of datasets to use; can also be the key of a "
                "mapping defined in the 'dataset_groups' auxiliary data of the corresponding "
                "config; default: ('*',)",
                brace_expand=True,
            )
        if has_skip_datasets:
            skip_datasets = law.CSVParameter(
                default=(),
                description="names or name patterns of datasets to skip after evaluating "
                "--datasets; can also be the key of a mapping defined in the 'dataset_groups' "
                "auxiliary data of the corresponding config; empty default",
                brace_expand=True,
            )
        if has_shifts:
            shifts = law.CSVParameter(
                default=("nominal",),
                description="names or name patterns of shifts to use; can also be the key of a "
                "mapping defined in the 'shift_groups' auxiliary data of the corresponding "
                "config; default: ('nominal',)",
                brace_expand=True,
            )
        if has_skip_shifts:
            skip_shifts = law.CSVParameter(
                default=(),
                description="names or name patterns of shifts to skip after evaluating --shifts; "
                "can also be the key of a mapping defined in the 'shift_groups' auxiliary data "
                "of the corresponding config; empty default",
                brace_expand=True,
            )
        if has_producers:
            producers = law.CSVParameter(
                default=("*",),
                description="names or name patterns of producers to use; can also be the key of a "
                "mapping defined in the 'producer_groups' auxiliary data of the corresponding "
                "config; default: ('*',)",
                brace_expand=True,
            )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # store wrapper flags
            self.wrapper_require_cls = require_cls
            self.wrapper_has_configs = has_configs and self.configs
            self.wrapper_has_skip_configs = has_skip_configs
            self.wrapper_has_shifts = has_shifts and self.shifts
            self.wrapper_has_skip_shifts = has_skip_shifts
            self.wrapper_has_datasets = has_datasets and self.datasets
            self.wrapper_has_skip_datasets = has_skip_datasets
            self.wrapper_has_producers = has_producers and self.producers

            # store wrapped fields
            self.wrapper_fields = [
                field for field in ["config", "shift", "dataset", "producer"]
                if getattr(self, f"wrapper_has_{field}s")
            ]

            # build the parameter space
            self.wrapper_parameters = self._build_wrapper_parameters()

        def _build_wrapper_parameters(self):
            # collect a list of tuples whose order corresponds to wrapper_fields
            params = []

            # get the target config instances
            if self.wrapper_has_configs:
                configs = self.find_config_objects(
                    self.configs,
                    self.analysis_inst,
                    od.Config,
                    self.analysis_inst.x("config_groups", {}),
                )
                if not configs:
                    raise ValueError(
                        f"no configs found in analysis {self.analysis_inst} matching {self.configs}",
                    )
                if self.wrapper_has_skip_configs:
                    skip_configs = self.find_config_objects(
                        self.skip_configs,
                        self.analysis_inst,
                        od.Config,
                        self.analysis_inst.x("config_groups", {}),
                    )
                    configs = [c for c in configs if c not in skip_configs]
                    if not configs:
                        raise ValueError(
                            f"no configs found in analysis {self.analysis_inst} after skipping "
                            f"{self.skip_configs}",
                        )
                config_insts = list(map(self.analysis_inst.get_config, sorted(configs)))
            else:
                config_insts = [self.config_inst]

            # for the remaining fields, build the full combinatorics per config_inst
            for config_inst in config_insts:
                # sequences for building combinatorics
                prod_sequences = []
                if self.wrapper_has_configs:
                    prod_sequences.append([config_inst.name])

                # find all shifts
                if self.wrapper_has_shifts:
                    shifts = self.find_config_objects(
                        self.shifts,
                        config_inst,
                        od.Shift,
                        config_inst.x("shift_groups", {}),
                    )
                    if not shifts:
                        raise ValueError(
                            f"no shifts found in config {config_inst} matching {self.shifts}",
                        )
                    if self.wrapper_has_skip_shifts:
                        skip_shifts = self.find_config_objects(
                            self.skip_shifts,
                            config_inst,
                            od.Shift,
                            config_inst.x("shift_groups", {}),
                        )
                        shifts = [s for s in shifts if s not in skip_shifts]
                    if not shifts:
                        raise ValueError(
                            f"no shifts found in config {config_inst} after skipping "
                            f"{self.skip_shifts}",
                        )
                    # move "nominal" to the front if present
                    shifts = sorted(shifts)
                    if "nominal" in shifts:
                        shifts.insert(0, shifts.pop(shifts.index("nominal")))
                    prod_sequences.append(shifts)

                # find all datasets
                if self.wrapper_has_datasets:
                    datasets = self.find_config_objects(
                        self.datasets,
                        config_inst,
                        od.Dataset,
                        config_inst.x("dataset_groups", {}),
                    )
                    if not datasets:
                        raise ValueError(
                            f"no datasets found in config {config_inst} matching "
                            f"{self.datasets}",
                        )
                    if self.wrapper_has_skip_datasets:
                        skip_datasets = self.find_config_objects(
                            self.skip_datasets,
                            config_inst,
                            od.Dataset,
                            config_inst.x("dataset_groups", {}),
                        )
                        datasets = [d for d in datasets if d not in skip_datasets]
                        if not datasets:
                            raise ValueError(
                                f"no datasets found in config {config_inst} after skipping "
                                f"{self.skip_datasets}",
                            )
                    prod_sequences.append(sorted(datasets))

                # find all producers
                if self.wrapper_has_producers:
                    producer_groups = config_inst.x("producer_groups", {})
                    producers = [
                        producer_
                        for producer in self.producers
                        for producer_ in law.util.make_list(producer_groups.get(producer, [producer]))
                    ]
                    if not producers:
                        raise ValueError(
                            f"no producers found matching "
                            f"{self.producers}",
                        )
                    prod_sequences.append(sorted(producers))

                # add the full combinatorics
                params.extend(itertools.product(*prod_sequences))

            return params

        def requires(self):
            # build all requirements based on the parameter space
            reqs = {}

            for values in self.wrapper_parameters:
                params = dict(zip(self.wrapper_fields, values))

                # allow custom checks and updates
                params = self.update_wrapper_params(params)
                if not params:
                    continue

                # add the requirement if not present yet
                req = self.wrapper_require_cls.req(self, **params)
                if req not in reqs.values():
                    reqs[values] = req

            return reqs

        def update_wrapper_params(self, params):
            return params

        # add additional class-level members
        if attributes:
            locals().update(attributes)

    # overwrite __module__ to point to the module of the calling stack
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    Wrapper.__module__ = module.__name__

    # overwrite __name__
    Wrapper.__name__ = cls_name or require_cls.__name__ + "Wrapper"

    # set docs
    if docs:
        Wrapper.__docs__ = docs

    return Wrapper
