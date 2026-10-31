from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.schemas.accounts import (
    AccountManagementResponse,
    DeleteAccountRequest,
    DeleteAccountResponse,
    ManagedAccount,
    OpsAccountCreate,
    OpsAccountUpdate,
    RentalAccountCreate,
    RentalAccountUpdate,
)
from app.schemas.admin import (
    ActivateManagedModelRequest,
    ActivateManagedModelResponse,
    DeleteManagedModelRequest,
    DeleteManagedModelResponse,
    ModelManagementResponse,
    ModelQueueSettings,
    RecognitionSettings,
    RecognitionSettingsResponse,
    ReferenceGuide,
)
from app.schemas.auth import AuthPrincipal
from app.schemas.quality import ExpectedView
from app.services.account_store import (
    AccountConflictError,
    AccountNotFoundError,
    get_account_store,
)
from app.services.auth import require_admin, verify_admin_password
from app.services.model_management import (
    ManagedModelDeleteConflict,
    ManagedModelNotFoundError,
    get_model_management_service,
)
from app.services.operations_store import get_operations_store
from app.services.runtime_settings import get_runtime_settings_store
from app.services.upload_validation import read_validated_image

router = APIRouter(prefix="/admin", tags=["admin settings"])


def _raise_account_error(exc: ValueError) -> None:
    if isinstance(exc, AccountNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, AccountConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise HTTPException(status_code=422, detail=str(exc)) from exc


def _account_management(current_subject: str) -> AccountManagementResponse:
    management = get_account_store().list_accounts(current_subject)
    return management.model_copy(
        update=get_operations_store().inspection_load_summary()
    )


@router.get("/accounts", response_model=AccountManagementResponse)
def list_managed_accounts(
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> AccountManagementResponse:
    return _account_management(principal.subject)


@router.post(
    "/accounts/rentals", response_model=ManagedAccount, status_code=status.HTTP_201_CREATED
)
def create_rental_account(
    request: RentalAccountCreate,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ManagedAccount:
    try:
        return get_account_store().create_rental(request, principal)
    except ValueError as exc:
        _raise_account_error(exc)


@router.put("/accounts/rentals/{rental_id}", response_model=ManagedAccount)
def update_rental_account(
    rental_id: str,
    request: RentalAccountUpdate,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ManagedAccount:
    try:
        return get_account_store().update_rental(rental_id, request, principal)
    except ValueError as exc:
        _raise_account_error(exc)


@router.delete(
    "/accounts/rentals/{rental_id}", response_model=DeleteAccountResponse
)
def delete_rental_account(
    rental_id: str,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> DeleteAccountResponse:
    subject = f"rental:{rental_id}"
    try:
        get_account_store().delete_account(subject, principal.subject)
    except ValueError as exc:
        _raise_account_error(exc)
    return DeleteAccountResponse(
        deleted_subject=subject,
        message="租單已刪除",
        management=_account_management(principal.subject),
    )


@router.post(
    "/accounts/ops", response_model=ManagedAccount, status_code=status.HTTP_201_CREATED
)
def create_ops_account(
    request: OpsAccountCreate,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ManagedAccount:
    try:
        return get_account_store().create_ops(request, principal)
    except ValueError as exc:
        _raise_account_error(exc)


@router.put("/accounts/ops/{username}", response_model=ManagedAccount)
def update_ops_account(
    username: str,
    request: OpsAccountUpdate,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ManagedAccount:
    try:
        return get_account_store().update_ops(username, request, principal)
    except ValueError as exc:
        _raise_account_error(exc)


@router.delete("/accounts/ops/{username}", response_model=DeleteAccountResponse)
def delete_ops_account(
    username: str,
    request: DeleteAccountRequest,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> DeleteAccountResponse:
    if request.confirmation_identifier != username:
        raise HTTPException(status_code=422, detail="刪除確認的帳號不一致")
    if not verify_admin_password(principal, request.admin_password):
        raise HTTPException(status_code=401, detail="管理員密碼錯誤，帳號未刪除")
    subject = f"ops:{username}"
    try:
        get_account_store().delete_account(subject, principal.subject)
    except ValueError as exc:
        _raise_account_error(exc)
    return DeleteAccountResponse(
        deleted_subject=subject,
        message="帳號已刪除",
        management=_account_management(principal.subject),
    )


@router.get("/recognition-settings", response_model=RecognitionSettingsResponse)
def get_recognition_settings(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> RecognitionSettingsResponse:
    return get_runtime_settings_store().get_recognition_settings()


@router.put("/recognition-settings", response_model=RecognitionSettingsResponse)
def update_recognition_settings(
    request: RecognitionSettings,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> RecognitionSettingsResponse:
    return get_runtime_settings_store().update_recognition_settings(
        request, actor=principal.subject
    )


@router.get("/model-management", response_model=ModelManagementResponse)
def get_model_management(
    _: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ModelManagementResponse:
    return get_model_management_service().get_management()


@router.put("/model-management", response_model=ModelManagementResponse)
def update_model_management(
    request: ModelQueueSettings,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ModelManagementResponse:
    try:
        return get_model_management_service().update_queue(
            request, actor=principal.subject
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/model-management/damage-pair/activate", response_model=ActivateManagedModelResponse
)
def activate_damage_pair_model(
    request: ActivateManagedModelRequest,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> ActivateManagedModelResponse:
    if not verify_admin_password(principal, request.admin_password):
        raise HTTPException(status_code=401, detail="管理員密碼錯誤，模型未切換")
    try:
        management, backup = get_model_management_service().activate_damage_pair_model(
            request.confirmation_model_id
        )
    except ManagedModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ManagedModelDeleteConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ActivateManagedModelResponse(
        activated_model_id=request.confirmation_model_id,
        backup_model_id=backup,
        message="前後車損比對已改用此模型，下一組比對立即生效",
        management=management,
    )


@router.delete(
    "/model-management/{model_id:path}", response_model=DeleteManagedModelResponse
)
def delete_managed_model(
    model_id: str,
    request: DeleteManagedModelRequest,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
) -> DeleteManagedModelResponse:
    if request.confirmation_model_id != model_id:
        raise HTTPException(status_code=422, detail="刪除確認的模型不一致")
    if not verify_admin_password(principal, request.admin_password):
        raise HTTPException(status_code=401, detail="管理員密碼錯誤，模型未刪除")
    try:
        management = get_model_management_service().delete_model(model_id)
    except ManagedModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ManagedModelDeleteConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return DeleteManagedModelResponse(
        deleted_model_id=model_id,
        message="模型已永久刪除",
        management=management,
    )


@router.get("/reference-guides", response_model=list[ReferenceGuide])
def list_reference_guides() -> list[ReferenceGuide]:
    return get_runtime_settings_store().list_reference_guides()


@router.get("/reference-guides/{view}/image", response_class=FileResponse)
def get_reference_guide_image(view: ExpectedView) -> FileResponse:
    path = get_runtime_settings_store().get_reference_guide_path(view)
    if path is None:
        raise HTTPException(status_code=404, detail="此拍攝位置尚未上傳正確示範照片")
    return FileResponse(
        path,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.put("/reference-guides/{view}", response_model=ReferenceGuide)
async def update_reference_guide(
    view: ExpectedView,
    principal: Annotated[AuthPrincipal, Depends(require_admin)],
    title: Annotated[str, Form(min_length=1, max_length=120)],
    description: Annotated[str, Form(min_length=1, max_length=500)],
    image: Annotated[UploadFile | None, File()] = None,
) -> ReferenceGuide:
    content = None
    filename = None
    if image is not None:
        content = await read_validated_image(image, get_settings().max_upload_mb)
        filename = (
            (image.filename or "reference.jpg")
            .replace("\\", "/")
            .rsplit("/", 1)[-1][:255]
        )
    current = get_runtime_settings_store().get_reference_guide(view)
    if content is None and (current is None or not current.image_available):
        raise HTTPException(status_code=422, detail="第一次建立示範時必須選擇照片")
    return get_runtime_settings_store().update_reference_guide(
        view,
        title=title,
        description=description,
        actor=principal.subject,
        image_content=content,
        filename=filename,
    )
