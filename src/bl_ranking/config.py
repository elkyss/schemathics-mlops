"""Runtime configuration. Every setting can be overridden by an environment variable of the same name."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["local", "databricks"] = "local"
    data_csv: Path = Path("data/bl_full_data.csv")
    warehouse_dir: Path = Path("warehouse")
    table_name: str = "bl.bl_full_data"

    mlflow_tracking_uri: str = "http://localhost:5001"
    model_name: str = "bl_ranker"

    tabpfn_backend: Literal["auto", "api", "local", "fallback"] = "auto"
    tabpfn_token: SecretStr | None = None
    tabpfn_deadline_s: float = 2.0

    holdout_days_prod: int = 0
    auto_promote: bool = True
    gate_max_regression: float = 0.05

    cache_ttl_s: int = 600
    cache_max_items: int = 10_000

    api_url: str = "http://localhost:8000"
    admin_token: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
