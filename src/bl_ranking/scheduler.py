"""Weekly production training (Sunday 05:00 local time) for the Docker stack.

MLflow records runs but does not schedule them. Locally this loop fires `bl train --mode production
--trigger schedule` on a cron expression in a configurable timezone; on Databricks the job's own cron schedule
(databricks/databricks.yml) does the same. A failed run is logged and the loop waits for the next slot — the
failure is also visible as a FAILED MLflow run and a stale `last_success`.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from croniter import croniter

log = logging.getLogger(__name__)


def next_run(cron: str, after: datetime) -> datetime:
    return croniter(cron, after).get_next(datetime)


def run_forever(cron: str, tz: str, command: Callable[[], int], *, now: Callable[[], datetime] | None = None,
                sleep: Callable[[float], None] = time.sleep, max_runs: int | None = None) -> None:
    zone = ZoneInfo(tz)
    now = now or (lambda: datetime.now(zone))
    runs = 0
    while max_runs is None or runs < max_runs:
        current = now()
        due = next_run(cron, current)
        log.info("next scheduled training at %s", due.isoformat())
        sleep(max(due.timestamp() - current.timestamp(), 0))  # elapsed time, correct across DST changes
        code = command()
        runs += 1
        (log.info if code == 0 else log.error)("scheduled training finished with exit code %s", code)
