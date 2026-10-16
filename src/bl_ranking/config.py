"""Runtime configuration. Every setting can be overridden by an environment variable of the same name."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ENV=databricks: same code, Databricks-managed MLflow + Unity Catalog names (explicit env values still win).
DATABRICKS_DEFAULTS = {"mlflow_tracking_uri": "databricks", "mlflow_registry_uri": "databricks-uc",
                       "model_name": "main.bl.bl_ranker", "table_name": "main.bl.bl_full_data",
                       "warehouse_dir": Path("/local_disk0/bl_ranking")}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["local", "databricks"] = "local"
    data_csv: Path = Path("data/bl_full_data.csv")
    warehouse_dir: Path = Path("warehouse")
    table_name: str = "bl.bl_full_data"

    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_registry_uri: str | None = None  # defaults to the tracking server (locally) / Unity Catalog (Databricks)
    model_name: str = "bl_ranker"

    tabpfn_backend: Literal["auto", "api", "local", "fallback"] = "local"  # auto = local (kept for compatibility)
    tabpfn_token: SecretStr | None = None
    tabpfn_deadline_s: float = 2.0
    serve_payout_backend: Literal["api", "local", "fallback"] | None = None  # override at serve time

    holdout_days_prod: int = 0
    schedule_cron: str = "0 5 * * 0"  # weekly production training: Sunday 05:00
    scheduler_tz: str = "Asia/Jerusalem"
    auto_promote: bool = True
    gate_max_regression: float = 0.05

    cache_ttl_s: int = 600
    cache_max_items: int = 10_000

    api_url: str = "http://localhost:8000"
    admin_token: SecretStr | None = None

    @model_validator(mode="after")
    def _databricks_defaults(self):
        if self.env == "databricks":
            for field, value in DATABRICKS_DEFAULTS.items():
                if field not in self.model_fields_set:
                    setattr(self, field, value)
        return self

    @property
    def registry_uri(self) -> str:
        return self.mlflow_registry_uri or self.mlflow_tracking_uri

    @property
    def experiment_prefix(self) -> str:
        return "/Shared/bl-ranking" if self.env == "databricks" else "bl-ranking"


@lru_cache
def get_settings() -> Settings:
    return Settings()
