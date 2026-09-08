"""``/api/v1/pins`` — durable, gallery-namespaced pinned references.

Pins are state, not learning. These endpoints work identically whether or not
``learning_enabled`` is set, and an affinity reset never clears them. Every
response is the full, revalidated pin list so a client can replace its local
view atomically instead of guessing what changed.
"""

from __future__ import annotations

from fastapi import APIRouter

from linescout_api import pins as pin_store
from linescout_api.deps import State
from linescout_api.errors import not_found
from linescout_api.schemas import ErrorResponse, PinsResponse

router = APIRouter(tags=["pins"])


@router.get("/pins", response_model=PinsResponse)
def get_pins(state: State) -> PinsResponse:
    """Pins for this API's gallery namespace, revalidated on every read."""
    return pin_store.list_pins(state.connection, pin_store.gallery_kind(state.settings))


@router.put(
    "/pins/{asset_id}",
    response_model=PinsResponse,
    responses={404: {"model": ErrorResponse, "description": "Unknown or ineligible asset"}},
)
def pin_asset(state: State, asset_id: str) -> PinsResponse:
    """Pin an asset. Idempotent: re-pinning keeps the original pin time."""
    kind = pin_store.gallery_kind(state.settings)
    if not pin_store.is_pinnable(state.connection, asset_id):
        raise not_found("asset_not_found", "asset not found")
    pin_store.add_pin(state.connection, kind, asset_id)
    return pin_store.list_pins(state.connection, kind)


@router.delete("/pins/{asset_id}", response_model=PinsResponse)
def unpin_asset(state: State, asset_id: str) -> PinsResponse:
    """Unpin an asset. Idempotent, and never a negative learning signal.

    Unpinning an asset that is not pinned (or no longer exists) succeeds: the
    caller's intent — "this must not be pinned" — is already true.
    """
    kind = pin_store.gallery_kind(state.settings)
    pin_store.remove_pin(state.connection, kind, asset_id)
    return pin_store.list_pins(state.connection, kind)
