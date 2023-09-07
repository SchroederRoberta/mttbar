# coding: utf-8
"""
Custom production-related tasks
"""
from columnflow.tasks.framework.base import AnalysisTask
from columnflow.tasks.production import ProduceColumns

from mtt.tasks.base import MTTTask
from mtt.tasks.wrapper import wrapper_factory


ProduceColumnsWrapper = wrapper_factory(
    base_cls=(AnalysisTask, MTTTask),
    require_cls=ProduceColumns,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts", "producers"],
)
