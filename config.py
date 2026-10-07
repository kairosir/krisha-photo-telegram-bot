from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    bot_token: SecretStr
    webhook_secret: SecretStr | None = None
    database_url: SecretStr | None = None
    log_level: str = "INFO"
    http_timeout_seconds: float = Field(default=25.0, gt=0)
    http_retries: int = Field(default=3, ge=1, le=8)
    max_page_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    max_image_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    max_images_per_listing: int = Field(default=50, ge=1, le=100)
    min_image_width: int = Field(default=320, ge=1)
    min_image_height: int = Field(default=240, ge=1)
    global_processing_limit: int = Field(default=4, ge=1, le=32)
    download_concurrency: int = Field(default=5, ge=1, le=20)
    send_originals_as_files: bool = True
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 KrishaPhotoBot/1.0"
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
