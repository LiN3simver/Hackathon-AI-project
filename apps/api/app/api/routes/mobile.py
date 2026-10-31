from fastapi import APIRouter, Request, Response

from app.services.mobile_access import generate_mobile_access

router = APIRouter(prefix="/mobile", tags=["mobile"])


@router.get("/access")
def mobile_access(request: Request, response: Response) -> dict:
    response.headers["Cache-Control"] = "no-store"
    if not getattr(request.app.state, "mobile_access", None):
        request.app.state.mobile_access = generate_mobile_access()
    return request.app.state.mobile_access
