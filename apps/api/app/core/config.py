from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Email 只透過 OneDrive 待寄資料夾交給 Power Automate 雲端流程寄出。
POWER_AUTOMATE_PROVIDER = "power_automate_onedrive"


class Settings(BaseSettings):
    app_name: str = "Guardian API"
    environment: str = "development"
    web_origin: str = "https://localhost:5173"
    mobile_web_url: str = ""
    mobile_web_port: int = Field(default=5173, ge=1, le=65535)
    max_upload_mb: int = 15
    view_model_path: Path = Path("models/exterior_view_classifier.pt")
    database_path: Path = Path("data/guardian.sqlite3")
    inspection_storage_path: Path = Path("data/inspection_images")
    inspection_analysis_queue_capacity: int = Field(default=50, ge=1, le=10_000)
    log_directory: Path = Path("logs")
    log_level: str = "INFO"
    bootstrap_admin_username: str = ""
    bootstrap_admin_password: SecretStr = SecretStr("")
    active_learning_storage_path: Path = Path("data/active_learning/images")
    # Where the competition photo folders live on this machine. Manifests written on
    # another machine keep their original root; see app.services.dataset_paths.
    competition_data_root: Path | None = None
    # Original manifest root, when paths need to be mapped to this machine.
    dataset_source_root: Path | None = None
    # The folder holding the 0921 supplemental case folders (001 … 500).
    supplemental_photo_root: Path | None = None
    active_learning_enabled: bool = True
    active_learning_score_threshold: int = Field(default=90, ge=0, le=100)
    active_learning_confidence_threshold: float = Field(default=0.90, ge=0, le=1)
    active_learning_min_training_score: int = Field(default=90, ge=0, le=100)
    active_learning_collect_good_samples: bool = False
    active_learning_auto_train: bool = True
    active_learning_min_approved: int = Field(default=20, ge=1)
    active_learning_max_metric_drop: float = Field(default=0.005, ge=0, le=1)
    active_learning_min_test_macro_f1: float = Field(default=0.85, ge=0, le=1)
    active_learning_min_accepted_accuracy: float = Field(default=0.90, ge=0, le=1)
    active_learning_max_coverage_drop: float = Field(default=0.10, ge=0, le=1)
    inspection_plate_check_enabled: bool = True
    plate_recognition_confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_vision_model: str = "qwen3-vl:4b-instruct"
    cleanliness_vision_model: str = "qwen3-vl:4b-instruct"
    cleanliness_localization_timeout_seconds: float = Field(default=180, ge=1, le=600)
    ollama_timeout_seconds: float = 180.0
    notification_email_enabled: bool = False
    notification_email_recipient: str = ""
    # Who gets the "please clean" email after staff confirm dirt; blank reuses the recipient above.
    notification_cleaning_email_recipient: str = ""
    notification_max_attempts: int = Field(default=3, ge=1, le=10)
    notification_retry_base_seconds: int = Field(default=30, ge=1, le=3600)
    power_automate_outbox_path: Path | None = None
    automation_routing_enabled: bool = True
    automation_shadow_mode: bool = True
    automation_auto_clear_confidence: float = Field(default=0.95, ge=0, le=1)
    automation_auto_case_confidence: float = Field(default=0.90, ge=0, le=1)
    automation_policy_version: str = "automation-routing-v1"
    history_rag_generation_enabled: bool = True
    history_rag_max_events: int = Field(default=20, ge=3, le=50)
    # Wall-clock limit for the one streamed summary; the deterministic answer card never waits on it.
    history_rag_generation_timeout_seconds: float = Field(default=25.0, ge=1, le=60)
    # A text model for the assistant (e.g. qwen3.5:2b-q4_K_M); unset falls back to the vision model.
    history_rag_model: str | None = None
    # Extra text models the assistant's 「Model」 picker offers, as a JSON list; see history_llm_models.py.
    history_llm_models: list[dict] = Field(default_factory=list)
    history_llm_keep_alive: str = "30m"
    history_llm_num_ctx: int = Field(default=4096, ge=1024, le=32768)
    history_llm_num_thread: int | None = Field(default=None, ge=1, le=64)
    history_llm_queue_seconds: float = Field(default=5.0, ge=0, le=60)
    history_llm_warmup: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GUARDIAN_",
        extra="ignore",
        hide_input_in_errors=True,
    )

    @field_validator("bootstrap_admin_username")
    @classmethod
    def _normalize_bootstrap_username(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _validate_bootstrap_credentials(self) -> "Settings":
        password = self.bootstrap_admin_password.get_secret_value()
        if bool(self.bootstrap_admin_username) != bool(password):
            raise ValueError("Set both bootstrap administrator username and password, or leave both unset")
        if password and len(password) < 12:
            raise ValueError("Bootstrap administrator password must contain at least 12 characters")
        return self

    @field_validator(
        "competition_data_root",
        "dataset_source_root",
        "supplemental_photo_root",
        "power_automate_outbox_path",
        "history_rag_model",
        "history_llm_num_thread",
        mode="before",
    )
    @classmethod
    def _blank_path_is_unset(cls, value: object) -> object:
        # `GUARDIAN_X=` in .env would otherwise become Path("."), the current directory.
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
