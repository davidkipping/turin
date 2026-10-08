"""Make skipped hurin comparisons visible, and optionally fatal.

Several tests compare turin with hurin's code, environment or products and
skip when those are absent. Such a skip means "parity unverified", not
"passing": 0.1.50 broke a parity test in a session without a hurin clone,
its suite read "14 skipped", and the break went unseen until 0.1.53. So:

- every skip whose reason mentions hurin is listed at the end of the run,
  in its own section;
- ``TURIN_REQUIRE_HURIN=1`` turns those skips into failures, for a session
  that is meant to verify parity.
"""
import os
from collections import Counter

import pytest

_REQUIRE = os.environ.get("TURIN_REQUIRE_HURIN", "") not in ("", "0")
_hurin_skips = Counter()


def _skip_reason(report):
    if report.skipped and isinstance(report.longrepr, tuple):
        return str(report.longrepr[2]).removeprefix("Skipped: ")
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    reason = _skip_reason(report)
    if reason is None or "hurin" not in reason.lower():
        return
    if _REQUIRE:
        report.outcome = "failed"
        report.longrepr = ("TURIN_REQUIRE_HURIN is set, but this hurin "
                           f"comparison skipped: {reason}")
    else:
        _hurin_skips[reason.splitlines()[0][:160]] += 1


def pytest_terminal_summary(terminalreporter):
    if not _hurin_skips:
        return
    tr = terminalreporter
    tr.section("hurin parity UNVERIFIED", sep="=", yellow=True, bold=True)
    for reason, n in _hurin_skips.most_common():
        tr.line(f"{n:3d} skipped: {reason}")
    tr.line(f"{sum(_hurin_skips.values())} hurin comparison(s) did not run: "
            "parity with hurin is unverified, not passing. "
            "TURIN_REQUIRE_HURIN=1 makes them fail instead.")
