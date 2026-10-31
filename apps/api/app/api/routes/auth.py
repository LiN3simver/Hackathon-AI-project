from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas.auth import AuthPrincipal, AuthSession, OpsLoginRequest, UserLoginRequest
from app.services.auth import (
    authenticate_ops,
    authenticate_user,
    issue_access_token,
    require_authenticated,
    revoke_access_session,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


def _session(principal: AuthPrincipal) -> AuthSession:
    token, expires_at = issue_access_token(principal)
    return AuthSession(access_token=token, expires_at=expires_at, principal=principal)


@router.post("/user/login", response_model=AuthSession)
def login_user(request: UserLoginRequest, http_request: Request) -> AuthSession:
    principal = authenticate_user(request)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="租單編號或密碼不正確")
    http_request.state.audit_actor = principal.subject
    return _session(principal)


@router.post("/ops/login", response_model=AuthSession)
def login_ops(request: OpsLoginRequest, http_request: Request) -> AuthSession:
    principal = authenticate_ops(request)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="帳號或密碼不正確")
    http_request.state.audit_actor = principal.subject
    return _session(principal)


@router.get("/me", response_model=AuthPrincipal)
def get_current_principal(
    principal: Annotated[AuthPrincipal, Depends(require_authenticated)],
) -> AuthPrincipal:
    return principal


@router.post("/logout", status_code=204)
def logout(
    principal: Annotated[AuthPrincipal, Depends(require_authenticated)],
) -> None:
    revoke_access_session(principal)
