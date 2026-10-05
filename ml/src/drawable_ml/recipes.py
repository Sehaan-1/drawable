from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import StrictRecord
from .io import MLError, load_json, sha256_file


class SemanticRecipe(StrictRecord):
    model_id: str
    max_epochs: int = Field(ge=1, le=100)
    early_stop_patience: int = Field(ge=0, le=20)
    final_image_stage_learning_rate: float = Field(gt=0, le=0.01)
    projection_learning_rate: float = Field(gt=0, le=0.01)
    scope_head_learning_rate: float = Field(gt=0, le=0.01)
    contrastive_temperature: float = Field(gt=0, le=1)
    scope_loss_weight: float = Field(ge=0, le=10)
    consistency_loss_weight: float = Field(ge=0, le=10)
    scope_count: Literal[11]
    view: Literal["tight"]


class StructuralRecipe(StrictRecord):
    model_id: str
    projection_dimensions: tuple[Literal[384], Literal[512], Literal[256]]
    triplet_margin: float = Field(gt=0, le=2)
    learning_rate: float = Field(gt=0, le=0.01)
    max_epochs: int = Field(ge=1, le=100)
    early_stop_patience: int = Field(ge=0, le=20)


class RetentionRecipe(StrictRecord):
    latest: Literal[2]
    best: Literal[1]
    final: Literal[1]


class TrainingRecipe(StrictRecord):
    schema_version: Literal[1]
    recipe_id: str
    seed: int = Field(ge=0, le=2**32 - 1)
    optimizer: Literal["AdamW"]
    weight_decay: float = Field(ge=0, le=1)
    effective_batch_size: int = Field(ge=1, le=512)
    microbatch_size: int = Field(ge=1, le=128)
    gradient_cache: Literal[True]
    semantic: SemanticRecipe
    structural: StructuralRecipe
    precision: Literal["fp16_if_forward_backward_validated_else_fp32"]
    session_budget_minutes: int = Field(ge=1, le=24 * 60)
    stop_starting_steps_minutes_before_budget: int = Field(ge=1, le=30)
    checkpoint_interval_seconds: int = Field(ge=30, le=3600)
    checkpoint_retention: RetentionRecipe
    protected_splits: tuple[Literal["threshold_selection", "calibration_audit", "locked_test"], ...]
    automatic_training: Literal[False]
    automatic_upload: Literal[False]

    @model_validator(mode="after")
    def coherent_budget(self) -> TrainingRecipe:
        if self.effective_batch_size % self.microbatch_size:
            raise ValueError("Effective batch size must be divisible by the microbatch size")
        if self.session_budget_minutes <= self.stop_starting_steps_minutes_before_budget:
            raise ValueError("A session must reserve time to stop safely before its hard budget")
        if len(set(self.protected_splits)) != 3:
            raise ValueError("Threshold, calibration, and locked-test splits must all remain protected")
        return self


def load_recipe(path: Path) -> tuple[TrainingRecipe, str]:
    try:
        recipe = TrainingRecipe.model_validate(load_json(path))
    except Exception as error:
        raise MLError(f"Training recipe is invalid: {error}") from error
    return recipe, sha256_file(path)
