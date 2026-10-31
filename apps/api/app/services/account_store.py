import base64
import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock

from app.core.config import get_settings
from app.schemas.accounts import (
    AccountManagementResponse,
    ManagedAccount,
    ManagedAccountKind,
    OpsAccountCreate,
    OpsAccountUpdate,
    RentalAccountCreate,
    RentalAccountUpdate,
)
from app.schemas.auth import AuthAudience, AuthPrincipal, AuthRole
from app.services.activity_audit import install_activity_audit

PROJECT_ROOT = Path(__file__).resolve().parents[4]
_PASSWORD_ITERATIONS = 310_000
_ACTIVE_DEVICE_WINDOW_MINUTES = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PASSWORD_ITERATIONS)
    return "$".join(
        (
            "pbkdf2_sha256",
            str(_PASSWORD_ITERATIONS),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        )
    )


def _password_matches(password: str, encoded: str | None) -> bool:
    if not encoded:
        return False
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            base64.urlsafe_b64decode(salt),
            int(iterations),
        )
        return hmac.compare_digest(base64.urlsafe_b64encode(digest).decode("ascii"), expected)
    except (ValueError, TypeError):
        return False


class AccountConflictError(ValueError):
    pass


class AccountNotFoundError(ValueError):
    pass


class ActiveAccountError(AccountConflictError):
    pass


class LastAdminError(AccountConflictError):
    pass


class AccountStore:
    def __init__(self, database_path: Path | None = None) -> None:
        raw_database = database_path or get_settings().database_path
        self.database_path = (
            raw_database if raw_database.is_absolute() else PROJECT_ROOT / raw_database
        )
        self._lock = RLock()

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS auth_accounts (
                    subject TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (kind IN ('rental', 'staff', 'admin')),
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    customer_id TEXT,
                    password_hash TEXT,
                    rental_id TEXT UNIQUE,
                    vehicle_id TEXT,
                    vehicle_label TEXT,
                    license_plate TEXT,
                    created_by TEXT NOT NULL,
                    created_by_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_auth_accounts_kind
                    ON auth_accounts(kind, created_at);
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL REFERENCES auth_accounts(subject)
                        ON DELETE RESTRICT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_active
                    ON auth_sessions(subject, expires_at, revoked_at);
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(auth_accounts)")
            }
            if "license_plate" not in columns:
                connection.execute("ALTER TABLE auth_accounts ADD COLUMN license_plate TEXT")
            if "customer_id" not in columns:
                connection.execute("ALTER TABLE auth_accounts ADD COLUMN customer_id TEXT")
            if "rental_date" not in columns:
                connection.execute("ALTER TABLE auth_accounts ADD COLUMN rental_date TEXT")
            if "return_date" not in columns:
                connection.execute("ALTER TABLE auth_accounts ADD COLUMN return_date TEXT")
            connection.execute(
                """UPDATE auth_accounts SET customer_id = rental_id
                WHERE kind = 'rental'
                  AND (customer_id IS NULL OR TRIM(customer_id) = '')"""
            )
            connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_auth_accounts_customer
                ON auth_accounts(customer_id, created_at)"""
            )
            install_activity_audit(connection)
            settings = get_settings()
            if (
                settings.bootstrap_admin_username
                and not connection.execute("SELECT 1 FROM auth_accounts LIMIT 1").fetchone()
            ):
                now = _now()
                username = settings.bootstrap_admin_username
                connection.execute(
                    """INSERT INTO auth_accounts
                    (subject, kind, username, display_name, password_hash,
                     created_by, created_by_name, created_at, updated_at)
                    VALUES (?, 'admin', ?, ?, ?, 'system:bootstrap',
                            'Administrator bootstrap', ?, ?)""",
                    (
                        f"ops:{username}",
                        username,
                        username,
                        _password_hash(settings.bootstrap_admin_password.get_secret_value()),
                        now,
                        now,
                    ),
                )

    @staticmethod
    def _principal(row: sqlite3.Row, session_id: str | None = None) -> AuthPrincipal:
        kind = ManagedAccountKind(row["kind"])
        if kind == ManagedAccountKind.RENTAL:
            return AuthPrincipal(
                subject=row["subject"],
                audience=AuthAudience.USER,
                role=AuthRole.USER,
                display_name=row["display_name"],
                customer_id=row["customer_id"],
                rental_id=row["rental_id"],
                vehicle_id=row["vehicle_id"],
                vehicle_label=row["vehicle_label"],
                license_plate=row["license_plate"],
                session_id=session_id,
            )
        return AuthPrincipal(
            subject=row["subject"],
            audience=AuthAudience.OPS,
            role=AuthRole.ADMIN if kind == ManagedAccountKind.ADMIN else AuthRole.STAFF,
            display_name=row["display_name"],
            customer_id=row["customer_id"],
            session_id=session_id,
        )

    def authenticate_rental(self, rental_id: str, password: str) -> AuthPrincipal | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_accounts WHERE kind = 'rental' AND rental_id = ?",
                (rental_id.strip(),),
            ).fetchone()
        if row is None or not _password_matches(password, row["password_hash"]):
            return None
        return self._principal(row)

    def authenticate_ops(self, username: str, password: str) -> AuthPrincipal | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM auth_accounts
                WHERE kind IN ('staff', 'admin') AND username = ?""",
                (username.strip(),),
            ).fetchone()
        if row is None or not _password_matches(password, row["password_hash"]):
            return None
        return self._principal(row)

    def current_principal(
        self, subject: str, session_id: str | None = None
    ) -> AuthPrincipal | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_accounts WHERE subject = ?", (subject,)
            ).fetchone()
        return self._principal(row, session_id) if row else None

    def verify_password(self, subject: str, password: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT kind, password_hash FROM auth_accounts WHERE subject = ?",
                (subject,),
            ).fetchone()
        return bool(
            row
            and row["kind"] == ManagedAccountKind.ADMIN.value
            and _password_matches(password, row["password_hash"])
        )

    def create_session(self, subject: str, expires_at: datetime) -> str:
        session_id = secrets.token_urlsafe(24)
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO auth_sessions
                (id, subject, created_at, expires_at, last_seen_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, NULL)""",
                (session_id, subject, now, expires_at.isoformat(), now),
            )
        return session_id

    def touch_session(self, session_id: str, subject: str) -> bool:
        now = _now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE auth_sessions SET last_seen_at = ?
                WHERE id = ? AND subject = ? AND revoked_at IS NULL AND expires_at > ?""",
                (now, session_id, subject, now),
            )
        return cursor.rowcount == 1

    def revoke_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE auth_sessions SET revoked_at = ?
                WHERE id = ? AND revoked_at IS NULL""",
                (_now(), session_id),
            )

    @staticmethod
    def _managed_account(
        row: sqlite3.Row, active_subjects: set[str], current_subject: str
    ) -> ManagedAccount:
        return ManagedAccount(
            subject=row["subject"],
            kind=ManagedAccountKind(row["kind"]),
            username=row["username"],
            display_name=row["display_name"],
            customer_id=row["customer_id"],
            rental_id=row["rental_id"],
            rental_date=row["rental_date"],
            return_date=row["return_date"],
            vehicle_id=row["vehicle_id"],
            vehicle_label=row["vehicle_label"],
            license_plate=row["license_plate"],
            created_by=row["created_by"],
            created_by_name=row["created_by_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=row["subject"] in active_subjects or row["subject"] == current_subject,
            is_current=row["subject"] == current_subject,
        )

    def list_accounts(self, current_subject: str) -> AccountManagementResponse:
        now_datetime = datetime.now(timezone.utc)
        now = now_datetime.isoformat()
        active_device_cutoff = (
            now_datetime - timedelta(minutes=_ACTIVE_DEVICE_WINDOW_MINUTES)
        ).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM auth_accounts
                ORDER BY CASE kind WHEN 'rental' THEN 1 WHEN 'admin' THEN 2 ELSE 3 END,
                         created_at, username"""
            ).fetchall()
            sessions = connection.execute(
                """SELECT subject, last_seen_at FROM auth_sessions
                WHERE revoked_at IS NULL AND expires_at > ?""",
                (now,),
            ).fetchall()
        active_subjects = {row["subject"] for row in sessions}
        return AccountManagementResponse(
            accounts=[self._managed_account(row, active_subjects, current_subject) for row in rows],
            active_account_count=len(active_subjects),
            active_session_count=len(sessions),
            active_device_count=sum(
                1 for row in sessions if row["last_seen_at"] >= active_device_cutoff
            ),
            active_device_window_minutes=_ACTIVE_DEVICE_WINDOW_MINUTES,
            generated_at=now,
        )

    def create_rental(self, request: RentalAccountCreate, actor: AuthPrincipal) -> ManagedAccount:
        now = _now()
        subject = f"rental:{request.rental_id}"
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """INSERT INTO auth_accounts
                    (subject, kind, username, display_name, customer_id, password_hash, rental_id,
                     vehicle_id, vehicle_label, license_plate, created_by, created_by_name,
                     created_at, updated_at, rental_date, return_date)
                    VALUES (?, 'rental', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        subject,
                        request.rental_id,
                        request.display_name or request.vehicle_label,
                        request.customer_id or request.rental_id,
                        _password_hash(request.password),
                        request.rental_id,
                        request.vehicle_id or request.license_plate,
                        request.vehicle_label,
                        request.license_plate,
                        actor.subject,
                        actor.display_name,
                        now,
                        now,
                        request.rental_date.isoformat() if request.rental_date else None,
                        request.return_date.isoformat() if request.return_date else None,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise AccountConflictError("租單編號已存在") from exc
        return self._get_managed(subject, actor.subject)

    def update_rental(
        self, rental_id: str, request: RentalAccountUpdate, actor: AuthPrincipal
    ) -> ManagedAccount:
        subject = f"rental:{rental_id}"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT rental_date, return_date FROM auth_accounts WHERE subject = ?",
                (subject,),
            ).fetchone()
            if current is None:
                raise AccountNotFoundError("找不到租單")
            start_date = (request.rental_date.isoformat() if request.rental_date else None) \
                if "rental_date" in request.model_fields_set else current["rental_date"]
            end_date = (request.return_date.isoformat() if request.return_date else None) \
                if "return_date" in request.model_fields_set else current["return_date"]
            if start_date and end_date and end_date < start_date:
                raise ValueError("還車日期不能早於租車日期")
            if request.password:
                cursor = connection.execute(
                    """UPDATE auth_accounts
                    SET display_name = COALESCE(?, display_name),
                        rental_date = CASE WHEN ? THEN ? ELSE rental_date END,
                        return_date = CASE WHEN ? THEN ? ELSE return_date END,
                        customer_id = COALESCE(?, customer_id),
                        vehicle_id = ?, vehicle_label = ?, license_plate = ?,
                        password_hash = ?, updated_at = ?
                    WHERE subject = ? AND kind = 'rental'""",
                    (
                        request.display_name,
                        "rental_date" in request.model_fields_set,
                        request.rental_date.isoformat() if request.rental_date else None,
                        "return_date" in request.model_fields_set,
                        request.return_date.isoformat() if request.return_date else None,
                        request.customer_id,
                        request.vehicle_id or request.license_plate,
                        request.vehicle_label,
                        request.license_plate,
                        _password_hash(request.password),
                        _now(),
                        subject,
                    ),
                )
            else:
                cursor = connection.execute(
                    """UPDATE auth_accounts
                    SET display_name = COALESCE(?, display_name),
                        rental_date = CASE WHEN ? THEN ? ELSE rental_date END,
                        return_date = CASE WHEN ? THEN ? ELSE return_date END,
                        customer_id = COALESCE(?, customer_id),
                        vehicle_id = ?, vehicle_label = ?, license_plate = ?, updated_at = ?
                    WHERE subject = ? AND kind = 'rental'""",
                    (
                        request.display_name,
                        "rental_date" in request.model_fields_set,
                        request.rental_date.isoformat() if request.rental_date else None,
                        "return_date" in request.model_fields_set,
                        request.return_date.isoformat() if request.return_date else None,
                        request.customer_id,
                        request.vehicle_id or request.license_plate,
                        request.vehicle_label,
                        request.license_plate,
                        _now(),
                        subject,
                    ),
                )
            if cursor.rowcount == 1 and request.customer_id is not None:
                for table in ("rental_events", "automation_decisions"):
                    if connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                        (table,),
                    ).fetchone():
                        connection.execute(
                            f"UPDATE {table} SET user_id = ? WHERE rental_id = ?",
                            (request.customer_id, rental_id),
                        )
        if cursor.rowcount != 1:
            raise AccountNotFoundError("找不到租單")
        return self._get_managed(subject, actor.subject)

    def get_rental_license_plate(self, rental_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT license_plate FROM auth_accounts
                WHERE kind = 'rental' AND rental_id = ?""",
                (rental_id.strip(),),
            ).fetchone()
        if row is None or not row["license_plate"]:
            return None
        return str(row["license_plate"])

    def get_customer_id_for_rental(self, rental_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT customer_id FROM auth_accounts
                WHERE kind = 'rental' AND rental_id = ?""",
                (rental_id.strip(),),
            ).fetchone()
        if row is None or not row["customer_id"]:
            return rental_id.strip()
        return str(row["customer_id"])

    def create_ops(self, request: OpsAccountCreate, actor: AuthPrincipal) -> ManagedAccount:
        now = _now()
        subject = f"ops:{request.username}"
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """INSERT INTO auth_accounts
                    (subject, kind, username, display_name, password_hash, rental_id,
                     vehicle_id, vehicle_label, created_by, created_by_name,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?)""",
                    (
                        subject,
                        request.kind.value,
                        request.username,
                        request.display_name,
                        _password_hash(request.password),
                        actor.subject,
                        actor.display_name,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise AccountConflictError("帳號已存在") from exc
        return self._get_managed(subject, actor.subject)

    def update_ops(
        self, username: str, request: OpsAccountUpdate, actor: AuthPrincipal
    ) -> ManagedAccount:
        subject = f"ops:{username}"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT kind FROM auth_accounts WHERE subject = ?", (subject,)
            ).fetchone()
            if current is None or current["kind"] == ManagedAccountKind.RENTAL.value:
                raise AccountNotFoundError("找不到營運帳號")
            if subject == actor.subject and current["kind"] != request.kind.value:
                raise AccountConflictError("目前登入帳號不可變更自己的角色")
            if (
                current["kind"] == ManagedAccountKind.ADMIN.value
                and request.kind != ManagedAccountKind.ADMIN
                and connection.execute(
                    "SELECT COUNT(*) FROM auth_accounts WHERE kind = 'admin'"
                ).fetchone()[0]
                <= 1
            ):
                raise LastAdminError("至少必須保留一個管理者帳號")
            if request.password:
                connection.execute(
                    """UPDATE auth_accounts SET kind = ?, display_name = ?,
                    password_hash = ?, updated_at = ? WHERE subject = ?""",
                    (
                        request.kind.value,
                        request.display_name,
                        _password_hash(request.password),
                        _now(),
                        subject,
                    ),
                )
            else:
                connection.execute(
                    """UPDATE auth_accounts SET kind = ?, display_name = ?,
                    updated_at = ? WHERE subject = ?""",
                    (request.kind.value, request.display_name, _now(), subject),
                )
        return self._get_managed(subject, actor.subject)

    def delete_account(self, subject: str, current_subject: str) -> None:
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT kind FROM auth_accounts WHERE subject = ?", (subject,)
            ).fetchone()
            if row is None:
                raise AccountNotFoundError("找不到要刪除的帳號")
            active = connection.execute(
                """SELECT COUNT(*) FROM auth_sessions
                WHERE subject = ? AND revoked_at IS NULL AND expires_at > ?""",
                (subject, now),
            ).fetchone()[0]
            if subject == current_subject or active:
                raise ActiveAccountError("此帳號目前使用中，登出前不可刪除")
            if (
                row["kind"] == ManagedAccountKind.ADMIN.value
                and connection.execute(
                    "SELECT COUNT(*) FROM auth_accounts WHERE kind = 'admin'"
                ).fetchone()[0]
                <= 1
            ):
                raise LastAdminError("至少必須保留一個管理者帳號")
            connection.execute("DELETE FROM auth_sessions WHERE subject = ?", (subject,))
            connection.execute("DELETE FROM auth_accounts WHERE subject = ?", (subject,))

    def _get_managed(self, subject: str, current_subject: str) -> ManagedAccount:
        management = self.list_accounts(current_subject)
        for account in management.accounts:
            if account.subject == subject:
                return account
        raise AccountNotFoundError("找不到帳號")


_store: AccountStore | None = None
_store_lock = RLock()


def get_account_store() -> AccountStore:
    global _store
    with _store_lock:
        if _store is None:
            candidate = AccountStore()
            candidate.initialize()
            _store = candidate
    return _store
