"""
AuditRx — Task Graders
Each grader returns a float in [0.0, 1.0].
All graders are deterministic given the same inputs.
"""

from __future__ import annotations
from typing import Any


# ──────────────────────────────────────────────────────────────
# TASK 1: classify_deviation  (EASY)
# The agent must correctly classify severity + category.
# Partial credit: 0.5 for severity, 0.5 for category.
# ──────────────────────────────────────────────────────────────

def grade_classification(
    agent_severity: str | None,
    agent_category: str | None,
    ground_truth: dict[str, Any],
) -> tuple[float, str]:
    """
    Returns (score 0.0–1.0, feedback string).
    0.5 per correct field.
    """
    feedback_parts = []
    score = 0.0

    gt_severity = ground_truth["severity"]
    gt_category = ground_truth["category"]

    if agent_severity is None:
        feedback_parts.append("No severity provided.")
    elif agent_severity == gt_severity:
        score += 0.5
        feedback_parts.append(f"Severity '{agent_severity}' ✓ correct.")
    else:
        feedback_parts.append(f"Severity '{agent_severity}' ✗ (expected '{gt_severity}').")

    if agent_category is None:
        feedback_parts.append("No category provided.")
    elif agent_category == gt_category:
        score += 0.5
        feedback_parts.append(f"Category '{agent_category}' ✓ correct.")
    else:
        feedback_parts.append(f"Category '{agent_category}' ✗ (expected '{gt_category}').")

    return round(score, 2), " ".join(feedback_parts)


# ──────────────────────────────────────────────────────────────
# TASK 2: draft_capa  (MEDIUM)
# Scores root_cause, corrective_action, preventive_action, timeline.
# Partial credit via keyword matching + timeline range check.
# ──────────────────────────────────────────────────────────────

def _keyword_score(text: str | None, keywords: list[str]) -> float:
    """What fraction of expected keywords appear in the agent's text."""
    if not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords) if keywords else 0.0


def grade_capa(
    root_cause: str | None,
    corrective_action: str | None,
    preventive_action: str | None,
    timeline_days: int | None,
    ground_truth: dict[str, Any],
) -> tuple[float, str]:
    """
    Returns (score 0.0–1.0, feedback string).
    Components:
      root_cause:         30% (keyword match)
      corrective_action:  30% (keyword match)
      preventive_action:  25% (keyword match)
      timeline_days:      15% (within acceptable range)
    """
    gt = ground_truth
    feedback_parts = []

    rc_score   = _keyword_score(root_cause,        gt["root_cause_keywords"])
    ca_score   = _keyword_score(corrective_action, gt["corrective_action_keywords"])
    pa_score   = _keyword_score(preventive_action, gt["preventive_action_keywords"])

    tl_score = 0.0
    tl_min, tl_max = gt["timeline_days_range"]
    if timeline_days is None:
        feedback_parts.append("No timeline provided.")
    elif tl_min <= timeline_days <= tl_max:
        tl_score = 1.0
        feedback_parts.append(f"Timeline {timeline_days}d ✓ within [{tl_min}–{tl_max}] days.")
    elif timeline_days < tl_min:
        # Partial credit for being close
        tl_score = max(0.0, 1.0 - (tl_min - timeline_days) / tl_min)
        feedback_parts.append(f"Timeline {timeline_days}d too aggressive (min {tl_min}d).")
    else:
        tl_score = max(0.0, 1.0 - (timeline_days - tl_max) / tl_max)
        feedback_parts.append(f"Timeline {timeline_days}d too long (max {tl_max}d).")

    feedback_parts.insert(0,
        f"Root cause coverage: {rc_score:.0%}. "
        f"Corrective action coverage: {ca_score:.0%}. "
        f"Preventive action coverage: {pa_score:.0%}."
    )

    total = (rc_score * 0.30) + (ca_score * 0.30) + (pa_score * 0.25) + (tl_score * 0.15)
    return round(min(total, 1.0), 2), " ".join(feedback_parts)


# ──────────────────────────────────────────────────────────────
# TASK 3: negotiate_escalation  (HARD)
# Multi-turn: agent talks to adversarial coordinator then escalates.
# Score components:
#   - Regulatory rebuttal quality (per turn): 40%
#   - Correct final escalation decision:      40%
#   - Efficiency (fewer turns used):          20%
# ──────────────────────────────────────────────────────────────

def grade_negotiation_turn(
    agent_message: str | None,
    ground_truth: dict[str, Any],
) -> tuple[float, str]:
    """
    Partial reward per negotiation turn.
    Checks if agent correctly rebuts the coordinator's incorrect claim.
    Returns (turn_score 0.0–0.4, feedback).
    """
    if not agent_message:
        return 0.0, "No message provided."

    msg_lower = agent_message.lower()
    correct_rebuttal = ground_truth["correct_regulatory_rebuttal"].lower()

    # Extract key regulatory terms from the correct rebuttal
    regulatory_keywords = [
        w for w in correct_rebuttal.split()
        if len(w) > 4 and w.isalpha()
    ]
    # Pick every 3rd keyword to get a representative sample
    sample_keywords = regulatory_keywords[::3] if regulatory_keywords else []

    keyword_hits = sum(1 for kw in sample_keywords if kw in msg_lower)
    keyword_ratio = keyword_hits / len(sample_keywords) if sample_keywords else 0.0

    # Also check if agent cites the coordinator's incorrect claim (shows awareness)
    incorrect_claim_words = ground_truth["coordinator_incorrect_claim"].lower().split()
    claim_sample = incorrect_claim_words[::4]
    claim_awareness = sum(1 for w in claim_sample if w in msg_lower) / len(claim_sample) if claim_sample else 0.0

    turn_score = (keyword_ratio * 0.7 + claim_awareness * 0.3) * 0.4
    feedback = (
        f"Regulatory keyword coverage: {keyword_ratio:.0%}. "
        f"Claim awareness: {claim_awareness:.0%}. "
        f"Turn partial score: {turn_score:.2f}."
    )
    return round(min(turn_score, 0.4), 2), feedback


def grade_escalation_decision(
    agent_escalation: str | None,
    ground_truth: dict[str, Any],
    negotiation_turns_used: int,
    max_negotiation_turns: int,
) -> tuple[float, str]:
    """
    Score the final escalation decision.
    Correct escalation: 0.40 points
    Efficiency bonus:   0.20 points (fewer turns = higher bonus)
    Returns (score 0.0–0.60, feedback).
    """
    feedback_parts = []
    score = 0.0

    correct = ground_truth["correct_escalation"]

    if agent_escalation is None:
        feedback_parts.append("No escalation decision made.")
    elif agent_escalation == correct:
        score += 0.40
        feedback_parts.append(f"Escalation '{agent_escalation}' ✓ correct.")
    else:
        # Partial credit for "close" decisions
        escalation_order = ["self_correct", "sponsor_notify", "irb_report", "halt_site"]
        try:
            agent_idx  = escalation_order.index(agent_escalation)
            correct_idx = escalation_order.index(correct)
            distance = abs(agent_idx - correct_idx)
            partial = max(0.0, 0.40 - distance * 0.15)
            score += partial
            feedback_parts.append(
                f"Escalation '{agent_escalation}' ✗ (expected '{correct}'). "
                f"Partial credit: {partial:.2f}."
            )
        except ValueError:
            feedback_parts.append(f"Invalid escalation '{agent_escalation}'.")

    # Efficiency bonus — reward for not dragging out the negotiation
    turns_ratio = negotiation_turns_used / max(max_negotiation_turns, 1)
    efficiency = max(0.0, 1.0 - turns_ratio) * 0.20
    score += efficiency
    feedback_parts.append(f"Efficiency bonus: {efficiency:.2f} ({negotiation_turns_used}/{max_negotiation_turns} turns used).")

    return round(min(score, 0.60), 2), " ".join(feedback_parts)