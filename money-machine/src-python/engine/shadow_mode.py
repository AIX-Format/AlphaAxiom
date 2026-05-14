"""
Shadow-mode runtime for dry-run decision capture and baseline comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional


@dataclass
class ShadowDecision:
    timestamp: float
    symbol: str
    entry: Optional[float]
    size: float
    risk: float
    exit: Optional[float]
    action: str
    confidence: float


@dataclass
class AcceptanceCriteria:
    max_entry_drift_pct: float = 0.01
    max_size_drift_pct: float = 0.15
    max_risk_drift_pct: float = 0.20
    min_match_rate: float = 0.70
    max_error_rate: float = 0.05


class ShadowModeMonitor:
    def __init__(self, criteria: Optional[AcceptanceCriteria] = None):
        self.criteria = criteria or AcceptanceCriteria()
        self.decisions: List[ShadowDecision] = []
        self.baseline: List[ShadowDecision] = []
        self.errors: List[Dict[str, str]] = []

    def record_decision(self, decision: ShadowDecision, *, baseline: bool = False) -> None:
        if baseline:
            self.baseline.append(decision)
        else:
            self.decisions.append(decision)

    def record_error(self, message: str) -> None:
        self.errors.append({"timestamp": datetime.now(tz=timezone.utc).isoformat(), "message": message})

    def summary(self, window_minutes: int = 60) -> Dict:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=window_minutes)
        live = [d for d in self.decisions if datetime.fromtimestamp(d.timestamp, tz=timezone.utc) >= cutoff]
        baseline = [d for d in self.baseline if datetime.fromtimestamp(d.timestamp, tz=timezone.utc) >= cutoff]
        compared = min(len(live), len(baseline))
        if compared == 0:
            return {
                "window_minutes": window_minutes,
                "compared": 0,
                "drift_rate": 0.0,
                "error_rate": 0.0,
                "accepted": False,
                "criteria": asdict(self.criteria),
            }
        entry_drift = 0.0
        size_drift = 0.0
        risk_drift = 0.0
        matches = 0
        for i in range(compared):
            a = live[i]
            b = baseline[i]
            if a.action == b.action:
                matches += 1
            if a.entry and b.entry and b.entry != 0:
                entry_drift += abs(a.entry - b.entry) / abs(b.entry)
            if b.size != 0:
                size_drift += abs(a.size - b.size) / abs(b.size)
            if b.risk != 0:
                risk_drift += abs(a.risk - b.risk) / abs(b.risk)
        avg_entry = entry_drift / compared
        avg_size = size_drift / compared
        avg_risk = risk_drift / compared
        drift_rate = (avg_entry + avg_size + avg_risk) / 3
        error_rate = len(self.errors) / max(1, len(live))
        match_rate = matches / compared
        accepted = (
            avg_entry <= self.criteria.max_entry_drift_pct
            and avg_size <= self.criteria.max_size_drift_pct
            and avg_risk <= self.criteria.max_risk_drift_pct
            and match_rate >= self.criteria.min_match_rate
            and error_rate <= self.criteria.max_error_rate
        )
        return {
            "window_minutes": window_minutes,
            "compared": compared,
            "avg_entry_drift_pct": avg_entry,
            "avg_size_drift_pct": avg_size,
            "avg_risk_drift_pct": avg_risk,
            "drift_rate": drift_rate,
            "match_rate": match_rate,
            "error_rate": error_rate,
            "accepted": accepted,
            "criteria": asdict(self.criteria),
            "recent_errors": self.errors[-10:],
        }
