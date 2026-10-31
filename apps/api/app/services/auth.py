import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request

from app.schemas.auth import (
    AuthAudience,
    AuthPrincipal,
    AuthRole,
    OpsLoginRequest,
    UserLoginRequest,
)
from app.services.account_store import get_account_store

_TOKEN_LIFETIME_SECONDS = 8 * 60 * 60
_TOKEN_SIGNING_KEY = secrets.token_bytes(32)
_bearer = HTTPBearer(auto_error=False)


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_access_token(principal: AuthPrincipal) -> tuple[str, datetime]:
    expires_at = int(time.time()) + _TOKEN_LIFETIME_SECONDS
    expires_at_datetime = datetime.fromtimestamp(expires_at, timezone.utc)
    session_id = get_account_store().create_session(
        principal.subject, expires_at_datetime
    )
    principal.session_id = session_id
    payload = {
        "exp": expires_at,
        "principal": principal.model_dump(mode="json"),
    }
    encoded = _encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _encode(
        hmac.new(_TOKEN_SIGNING_KEY, encoded.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{encoded}.{signature}", expires_at_datetime


def decode_access_token(token: str) -> AuthPrincipal:
    try:
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = _encode(
            hmac.new(_TOKEN_SIGNING_KEY, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError("invalid signature")
        payload = json.loads(_decode(encoded))
        if int(payload["exp"]) <= int(time.time()):
            raise ValueError("expired")
        principal = AuthPrincipal.model_validate(payload["principal"])
        if not principal.session_id or not get_account_store().touch_session(
            principal.session_id, principal.subject
        ):
            raise ValueError("inactive session")
        current = get_account_store().current_principal(
            principal.subject, principal.session_id
        )
        if current is None:
            raise ValueError("account no longer exists")
        return current
    except (
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        binascii.Error,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登入已失效，請重新登入",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def authenticate_user(request: UserLoginRequest) -> AuthPrincipal | None:
    return get_account_store().authenticate_rental(request.rental_id, request.password)


def authenticate_ops(request: OpsLoginRequest) -> AuthPrincipal | None:
    return get_account_store().authenticate_ops(request.username, request.password)


def verify_admin_password(principal: AuthPrincipal, password: str) -> bool:
    if principal.role != AuthRole.ADMIN or not principal.subject.startswith("ops:"):
        return False
    return get_account_store().verify_password(principal.subject, password)


def revoke_access_session(principal: AuthPrincipal) -> None:
    get_account_store().revoke_session(principal.session_id)


def require_authenticated(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    access_token: Annotated[str | None, Query()] = None,
) -> AuthPrincipal:
    token = credentials.credentials if credentials else access_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="請先登入",
            headers={"WWW-Authenticate": "Bearer"},
        )
    principal = decode_access_token(token)
    request.state.audit_actor = principal.subject
    return principal


def require_user(
    principal: Annotated[AuthPrincipal, Depends(require_authenticated)],
) -> AuthPrincipal:
    if principal.audience != AuthAudience.USER:
        raise HTTPException(status_code=403, detail="此功能只提供租車使用者")
    return principal


def require_ops(
    principal: Annotated[AuthPrincipal, Depends(require_authenticated)],
) -> AuthPrincipal:
    if principal.audience != AuthAudience.OPS:
        raise HTTPException(status_code=403, detail="此功能只提供營運人員")
    return principal


def require_admin(
    principal: Annotated[AuthPrincipal, Depends(require_ops)],
) -> AuthPrincipal:
    if principal.role != AuthRole.ADMIN:
        raise HTTPException(status_code=403, detail="只有後台管理員可以修改此設定")
    return principal
