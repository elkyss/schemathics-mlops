"""Model registry lifecycle for bl_ranker: register → @challenger → (gate) → @champion, and rollback.

Every promotion records the champion it replaced in the new version's ``previous_champion`` tag, so a rollback is
a single alias move — the serving API picks it up on ``/admin/reload`` without a restart.
"""
from __future__ import annotations

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

CHAMPION, CHALLENGER = "champion", "challenger"


class PromotionRefused(RuntimeError):
    pass


class ModelRegistryManager:
    def __init__(self, client: MlflowClient, name: str):
        self.client, self.name = client, name

    def _ensure_model(self) -> None:
        try:
            self.client.get_registered_model(self.name)
        except MlflowException:
            self.client.create_registered_model(self.name, description="Business Loans brand ranking "
                                                "(CatBoost P(lead) × TabPFN payout).")

    def register(self, model_uri: str, tags: dict[str, str], description: str | None = None) -> str:
        self._ensure_model()
        mv = self.client.create_model_version(self.name, source=model_uri, tags=tags, description=description)
        self.set_alias(CHALLENGER, str(mv.version))
        return str(mv.version)

    def get(self, alias: str) -> ModelVersion | None:
        try:
            return self.client.get_model_version_by_alias(self.name, alias)
        except MlflowException:
            return None

    def version(self, version: str) -> ModelVersion:
        return self.client.get_model_version(self.name, version)

    def set_alias(self, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(self.name, alias, version)

    def promote(self, version: str, *, force: bool = False, reason: str | None = None) -> str:
        new = self.version(version)
        current = self.get(CHAMPION)
        version = str(version)
        if current is not None and str(current.version) == version:
            return version
        if (not force and current is not None and new.tags.get("payout_backend") == "fallback"
                and current.tags.get("payout_backend") != "fallback"):
            raise PromotionRefused(
                f"v{version} was trained with the fallback payout model and would replace TabPFN champion "
                f"v{current.version}; use force=True with a reason to override")
        if current is not None:
            self.client.set_model_version_tag(self.name, version, "previous_champion", str(current.version))
        if reason:
            self.client.set_model_version_tag(self.name, version, "promoted_reason", reason)
        self.set_alias(CHAMPION, version)
        challenger = self.get(CHALLENGER)
        if challenger is not None and str(challenger.version) == version:
            self.client.delete_registered_model_alias(self.name, CHALLENGER)
        return version

    def rollback(self) -> str:
        current = self.get(CHAMPION)
        previous = current.tags.get("previous_champion") if current is not None else None
        if not previous:
            raise PromotionRefused("no previous champion to roll back to")
        self.set_alias(CHAMPION, previous)
        self.client.set_model_version_tag(self.name, current.version, "rolled_back", "true")
        return previous
