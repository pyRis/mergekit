# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import List, Optional

from pydantic import BaseModel, model_validator

from mergekit.evo.genome import ModelGenomeDefinition


class TaskConfiguration(BaseModel, frozen=True):
    name: str
    weight: float = 1.0
    metric: str = "acc,none"

    @model_validator(mode="before")
    def validate_string(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value


class EvolMergeConfiguration(BaseModel, frozen=True):
    """
    Configuration for evolutionary model merging.
    
    For custom qu-du-tasks evaluation, the 'tasks' field is optional.
    If not specified, all supported tasks will be used with equal weights.
    Supported tasks include:
    - Query expansion (gov2, trec_robust, trec_covid, fire, query2doc, trec_cast, trec_web)
    - Query reformulation (codec, qrecc, canard, trec_cast, gecor)
    - Fact verification (fever, climate_fever, scifact)
    - Summarization (cnndm, xsum, wikisum, multinews)
    
    Example minimal configuration:
    ```yaml
    genome:
      models:
        - model_1
        - model_2
      merge_method: dare_ties
      base_model: base_model_if_needed
    # tasks: # Optional - if omitted, all supported tasks will be used
    ```
    """
    genome: ModelGenomeDefinition
    tasks: Optional[List[TaskConfiguration]] = None
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    shuffle: bool = False
    random_init: bool = False
    apply_chat_template: bool = True
    fewshot_as_multiturn: bool = True

    @model_validator(mode="after")
    def set_default_tasks(self):
        if self.tasks is None:
            # Use predefined tasks for custom qu-du-tasks evaluation
            default_tasks = [
                # Query expansion tasks
                TaskConfiguration(name="query_expansion_gov2", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_expansion_trec_robust", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_expansion_trec_covid", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_expansion_fire", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_expansion_query2doc", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_expansion_trec_cast", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_expansion_trec_web", weight=1.0, metric="ROUGE-L"),
                
                # Query reformulation tasks
                TaskConfiguration(name="query_reformulation_codec", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_reformulation_qrecc", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_reformulation_canard", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_reformulation_trec_cast", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="query_reformulation_gecor", weight=1.0, metric="ROUGE-L"),
                
                # Fact verification tasks
                TaskConfiguration(name="fact_verification_fever", weight=1.0, metric="Acc"),
                TaskConfiguration(name="fact_verification_climate_fever", weight=1.0, metric="Acc"),
                TaskConfiguration(name="fact_verification_scifact", weight=1.0, metric="Acc"),
                
                # Summarization tasks
                TaskConfiguration(name="summarization_cnndm", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="summarization_xsum", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="summarization_wikisum", weight=1.0, metric="ROUGE-L"),
                TaskConfiguration(name="summarization_multinews", weight=1.0, metric="ROUGE-L"),
            ]
            object.__setattr__(self, 'tasks', default_tasks)
        return self


NAUGHTY_PREFIXES = [
    "mmlu",
    "hendrycks",
    "agieval",
    "gsm8k",
    "hellaswag",
    "winogrande",
    "arc_",
    "ai2_arc",
    "truthfulqa",
    "bigbench",
    "piqa",
    "openbookqa",
    "leaderboard",
]


def check_for_naughty_config(config: EvolMergeConfiguration, allow: bool = False):
    """
    Check if the given configuration is naughty and should be disallowed.

    mergekit-evolve is perfectly set up to directly optimize against the test set
    of common benchmarks, which just makes the world a worse place. There are
    cases where this is useful but it deserves a giant honking warning.
    """
    suffix = ""
    if not allow:
        suffix = (
            " To proceed, set the "
            "--i-understand-the-depths-of-the-evils-i-am-unleashing flag."
        )
    for task in config.tasks:
        for prefix in NAUGHTY_PREFIXES:
            if task.name.startswith(prefix):
                if task.name.endswith("_train"):
                    # there aren't any tasks that match this pattern in base
                    # lm-eval, but it'd be a sane thing to do to add tasks for
                    # the training sets of these benchmarks. don't warn about
                    # them
                    continue

                message = (
                    f"Task {task.name} is a common benchmark task. "
                    "Optimizing against this task directly is unsporting at best "
                    "and outright malicious at worst. Using mergekit-evolve to "
                    "game benchmarks will be a black mark on your name for a "
                    f"thousand generations.{suffix}"
                )
                if not allow:
                    raise ValueError(message)
                else:
                    logging.warning(message)
