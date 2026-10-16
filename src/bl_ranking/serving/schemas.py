"""Request contract for /v1/predict: the post-funnel dict from bl_exp_payout_predictor.py's example.

Validation covers what the original code would crash on (missing register_date, non-numeric cellphone) so those
become 422s with a clear message instead of 500s. Values are passed through untouched otherwise — e.g. ``sub1``
arrives as an int in the researcher's example and as a zero-padded string in the data; the original code
(``astype(str)``) decides what that means, not the API.
"""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class UserRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_dt: str
    conversion_dt: str | None = None
    register_date: str
    campaign_id: int | str | None = None
    page: str  # a categorical model feature the original never fills; None reaches CatBoost as an error
    auto_city: str | None = None
    auto_country: str | None = None
    auto_state: str | None = None
    device_type: str | None = None
    sub1: int | str | None = None
    sub2: int | str | None = None
    sub3: int | str | None = None
    business_type: str | None = None
    credit_score: str | None = None
    industry: str | None = None
    loan_amount: str | None = None
    loan_reason: str | None = None
    monthly_revenue: str | None = None
    time_in_business: str | None = None
    fname: str | None = None
    lname: str | None = None
    cellphone: int

    @field_validator("register_date")
    @classmethod
    def _registered(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("register_date is required: a user without it cannot be a lead")
        return v

    @field_validator("session_dt", "register_date")
    @classmethod
    def _timestamp(cls, v: str) -> str:
        try:  # the model derives day, weekday, hour and seconds-to-register from these
            datetime.fromisoformat(v.strip())
        except ValueError:
            raise ValueError("must be a timestamp like 2026-01-06 19:24:22") from None
        return v

    @field_validator("page")
    @classmethod
    def _page(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("page (landing page) is required")
        return v

    @field_validator("cellphone", mode="before")
    @classmethod
    def _numeric_phone(cls, v):
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("cellphone must be numeric digits (the model uses its 3-digit prefix)")
        return v
