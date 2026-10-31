from fastapi import HTTPException

from app.schemas.auth import AuthPrincipal
from app.schemas.inspection import ActiveRentalContext


def get_active_rental_context(principal: AuthPrincipal) -> ActiveRentalContext:
    """Resolve inspection context from the authenticated rental account."""
    if not principal.rental_id or not principal.vehicle_id or not principal.vehicle_label:
        raise HTTPException(status_code=403, detail="The authenticated account has no active rental context")
    return ActiveRentalContext(
        rental_id=principal.rental_id,
        vehicle_id=principal.vehicle_id,
        vehicle_label=principal.vehicle_label,
        license_plate=principal.license_plate,
        source="authenticated_rental_account",
    )


def get_principal_rental_identity(principal: AuthPrincipal) -> tuple[str, str]:
    """Return the authenticated rental identity."""
    if not principal.rental_id or not principal.vehicle_id:
        raise ValueError("The authenticated account has no rental or vehicle identity")
    return principal.rental_id, principal.vehicle_id
