from __future__ import annotations

from typing import Any


def masked_multi_positive_infonce(
    query_features: Any,
    reference_features: Any,
    positive_mask: Any,
    allowed_mask: Any,
    *,
    temperature: float = 0.07,
) -> tuple[Any, int]:
    """Exact masked multi-positive InfoNCE over a complete effective batch.

    `allowed_mask` must contain only a query's reviewed positives and reviewed
    negatives. Unjudged references must be false in both masks.
    """
    import torch

    if temperature <= 0 or query_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("InfoNCE inputs or temperature are invalid")
    if positive_mask.shape != allowed_mask.shape or positive_mask.shape != (query_features.shape[0], reference_features.shape[0]):
        raise ValueError("Positive and allowed masks must match the query/reference matrix")
    positive_mask = positive_mask.to(dtype=torch.bool, device=query_features.device)
    allowed_mask = allowed_mask.to(dtype=torch.bool, device=query_features.device)
    if torch.any(positive_mask & ~allowed_mask):
        raise ValueError("Every positive must be an allowed candidate")
    q = torch.nn.functional.normalize(query_features.float(), dim=-1)
    r = torch.nn.functional.normalize(reference_features.float(), dim=-1)
    logits = (q @ r.transpose(0, 1)) / temperature
    losses: list[Any] = []
    for row in range(logits.shape[0]):
        positives = positive_mask[row]
        allowed = allowed_mask[row]
        # Missing a positive or an explicitly reviewed negative is a skip, not a
        # fabricated negative or a zero loss.
        if not torch.any(positives) or torch.count_nonzero(allowed & ~positives) == 0:
            continue
        losses.append(torch.logsumexp(logits[row][allowed], dim=0) - torch.logsumexp(logits[row][positives], dim=0))
    if not losses:
        return query_features.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def masked_scope_bce(logits: Any, labels: Any, known_mask: Any) -> Any:
    import torch

    if logits.shape != labels.shape or labels.shape != known_mask.shape or logits.shape[-1] != 11:
        raise ValueError("Scope logits, labels, and masks must share the 11-output shape")
    labels = labels.to(dtype=torch.float32, device=logits.device)
    mask = known_mask.to(dtype=torch.bool, device=logits.device)
    known_targets = labels[:, :10]
    unknown_targets = labels[:, 10]
    if torch.any(unknown_targets.bool() & known_targets.bool().any(dim=1)):
        raise ValueError("Unknown is exclusive of known-scope labels")
    if not torch.any(mask):
        return logits.sum() * 0.0
    per_label = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    return per_label[mask].mean()


def same_prefix_consistency(first: Any, second: Any) -> Any:
    import torch

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Consistency views must be equally shaped embedding batches")
    first = torch.nn.functional.normalize(first.float(), dim=-1)
    second = torch.nn.functional.normalize(second.float(), dim=-1)
    return (1.0 - (first * second).sum(dim=-1)).mean()
