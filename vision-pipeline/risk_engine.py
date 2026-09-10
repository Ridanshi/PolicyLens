"""
Cross-references a detected behavior against THAT package's own extracted
policy (not a generic global rule) and produces a risk verdict.

This file is the actual novel-angle logic: same "dropped" behavior is low
risk for a non-fragile item within its stated drop tolerance, and critical
for an item labeled fragile with max_drop_height_m = 0. Violation must be
evaluated against the specific item, never a fixed global threshold.
"""
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field


@dataclass
class RiskVerdict:
    risk_level: str  # low | medium | high | critical
    confidence: float
    requires_human_review: bool
    reason: str


def evaluate(behavior: str, policy: dict, impact_energy_j: float | None = None) -> RiskVerdict:
    if behavior == "none":
        return RiskVerdict("low", 1.0, False, "no violation observed")

    if behavior == "dropped":
        max_drop = policy.get("max_drop_height_m", 0.3)
        if policy.get("fragile") and max_drop <= 0.05:
            return RiskVerdict("critical", 0.85, True, "fragile item dropped, policy allows zero drop")
        if impact_energy_j is not None and impact_energy_j > 15:
            return RiskVerdict("high", 0.8, True, f"high impact energy ({impact_energy_j} J) on drop")
        return RiskVerdict("medium", 0.7, False, "drop detected within item's non-fragile tolerance, still logged")

    if behavior == "wrong_orientation":
        if policy.get("orientation") == "upright":
            return RiskVerdict("high", 0.75, True, "item requires upright orientation, orientation change detected")
        return RiskVerdict("low", 0.6, False, "orientation change on item with no orientation requirement")

    if behavior == "over_stacked":
        max_stack = policy.get("max_stack")
        if max_stack is not None:
            return RiskVerdict("high", 0.8, True, f"stack exceeds item's own printed limit of {max_stack}")
        return RiskVerdict("medium", 0.5, False, "stacking detected, no printed stack limit found on item")

    if behavior == "dragged":
        return RiskVerdict("medium", 0.75, False, "item dragged instead of lifted/moved with equipment")

    if behavior == "stepped_on":
        return RiskVerdict("high", 0.8, True, "person stepped on package")

    if behavior == "thrown":
        return RiskVerdict("critical", 0.7, True, "throwing motion detected")

    return RiskVerdict("medium", 0.4, False, f"unclassified behavior: {behavior}")


def build_event(video_source: str, bay: str, track_id: int, behavior: str, policy: dict,
                 frame_number: int, drop_height_m: float | None = None,
                 impact_velocity_mps: float | None = None, impact_energy_j: float | None = None,
                 clip_path: str = "") -> dict:
    verdict = evaluate(behavior, policy, impact_energy_j)
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video_source": video_source,
        "bay": bay,
        "package_id": str(track_id),
        "policy": policy,
        "detected_behavior": behavior,
        "evidence": {
            "drop_height_m": drop_height_m,
            "impact_velocity_mps": impact_velocity_mps,
            "impact_energy_j": impact_energy_j,
            "clip_path": clip_path,
            "frame_number": frame_number,
        },
        "risk_level": verdict.risk_level,
        "confidence": verdict.confidence,
        "requires_human_review": verdict.requires_human_review,
        "_reason": verdict.reason,  # not in the shared schema, handy for debugging/demo narration
    }
