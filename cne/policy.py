"""A deliberately simple, SYNTHETIC prior-authorisation policy.

Not medical guidance. Design rule worth keeping: the system never denies on
its own. It either approves or pends the case for a clinician, which is the
human-in-the-loop boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Extraction

MIN_CONSERVATIVE_WEEKS = 6


@dataclass(frozen=True)
class Decision:
    decision: str  # "approve" or "pend"
    reasons: tuple[str, ...] = field(default_factory=tuple)


def adjudicate(ex: Extraction) -> Decision:
    proc = ex.requested_procedure
    if not proc:
        return Decision("pend", ("no requested procedure identified",))

    if proc == "mri brain":
        if ex.red_flag is True:
            return Decision("approve", ("red flag documented",))
        if ex.red_flag is None:
            return Decision("pend", ("red flag status missing",))
        return Decision("pend", ("no red flag, route to clinician review",))

    if proc in ("mri lumbar spine", "mri knee"):
        if proc == "mri lumbar spine" and ex.red_flag is True:
            return Decision("approve", ("red flag documented",))
        weeks = ex.conservative_therapy_weeks
        if weeks is None:
            return Decision("pend", ("conservative therapy duration missing",))
        if weeks >= MIN_CONSERVATIVE_WEEKS:
            return Decision("approve", (f"{weeks} weeks of conservative therapy",))
        return Decision("pend", (f"only {weeks} weeks of conservative therapy",))

    return Decision("pend", (f"no policy for procedure {proc!r}",))
