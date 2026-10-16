from datetime import datetime
from zoneinfo import ZoneInfo

from bl_ranking.scheduler import next_run, run_forever

TZ = ZoneInfo("Asia/Jerusalem")


def test_next_run_is_sunday_0500_local():
    assert next_run("0 5 * * 0", datetime(2026, 10, 16, 10, 0, tzinfo=TZ)) == datetime(2026, 10, 18, 5, 0, tzinfo=TZ)


def test_next_run_after_firing_is_next_week():
    assert next_run("0 5 * * 0", datetime(2026, 10, 18, 5, 0, tzinfo=TZ)) == datetime(2026, 10, 25, 5, 0, tzinfo=TZ)


def test_run_forever_fires_command_at_schedule_and_survives_failure():
    clock = {"now": datetime(2026, 10, 18, 4, 59, 30, tzinfo=TZ)}
    slept, fired = [], []

    def sleep(seconds):
        slept.append(seconds)
        clock["now"] = clock["now"].__class__.fromtimestamp(clock["now"].timestamp() + seconds, TZ)

    def command():
        fired.append(clock["now"])
        return 1 if len(fired) == 1 else 0  # first run fails; the scheduler must keep going

    run_forever("0 5 * * 0", "Asia/Jerusalem", command, now=lambda: clock["now"], sleep=sleep, max_runs=2)
    assert fired == [datetime(2026, 10, 18, 5, 0, tzinfo=TZ), datetime(2026, 10, 25, 5, 0, tzinfo=TZ)]
    assert slept[0] == 30
