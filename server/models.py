"""
AuditRx — Clinical Trial Deviation Triage Environment
Pydantic models for Action, Observation, and State.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class DeviationSeverity(str, Enum):
    CRITICAL = "critical"       # immediate patient safety risk
    MAJOR    = "major"          # significant protocol violation
    MINOR    = "minor"          # minimal impact, self-correcting


class DeviationCategory(str, Enum):
    INFORMED_CONSENT      = "informed_consent"
    ELIGIBILITY_CRITERIA  = "eligibility_criteria"
    DOSING_ERROR          = "dosing_error"
    VISIT_WINDOW          = "visit_window"
    DATA_INTEGRITY        = "data_integrity"
    SAFETY_REPORTING      = "safety_reporting"
    RANDOMIZATION         = "randomization"


class EscalationDecision(str, Enum):
    SELF_CORRECT   = "self_correct"    # site can handle internally
    SPONSOR_NOTIFY = "sponsor_notify"  # notify sponsor within 24h
    IRB_REPORT     = "irb_report"      # mandatory IRB/ethics board report
    HALT_SITE      = "halt_site"       # suspend enrollment


class ActionType(str, Enum):
    CLASSIFY      = "classify"       # Task 1
    DRAFT_CAPA    = "draft_capa"     # Task 2
    NEGOTIATE     = "negotiate"      # Task 3 — send message to coordinator
    ESCALATE      = "escalate"       # Task 3 — final escalation decision
    REQUEST_INFO  = "request_info"   # ask for more case detail (costs a step)


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

class AuditRxAction(BaseModel):
    """Single action sent by the agent each step."""

    action_type: ActionType = Field(..., description="Which action the agent is taking")

    # Task 1 — classify
    severity: Optional[DeviationSeverity] = Field(None, description="Classified severity (Task 1)")
    category: Optional[DeviationCategory] = Field(None, description="Classified category (Task 1)")

    # Task 2 — CAPA
    root_cause: Optional[str]    = Field(None, max_length=500, description="Root cause statement (Task 2)")
    corrective_action: Optional[str] = Field(None, max_length=800, description="Corrective action (Task 2)")
    preventive_action: Optional[str] = Field(None, max_length=800, description="Preventive action (Task 2)")
    timeline_days: Optional[int] = Field(None, ge=1, le=180, description="Days to complete CAPA (Task 2)")

    # Task 3 — negotiation / escalation
    message: Optional[str]           = Field(None, max_length=1000, description="Message to site coordinator (Task 3)")
    escalation: Optional[EscalationDecision] = Field(None, description="Final escalation decision (Task 3)")

    # request_info
    info_request: Optional[str] = Field(None, max_length=200, description="Specific info requested")


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class AuditRxObservation(BaseModel):
    """What the agent sees after each step."""

    task_name: str
    step_number: int
    max_steps: int

    # The deviation case
    case_id: str
    case_description: str
    protocol_id: str
    site_id: str
    patient_id: str
    event_date: str
    reported_by: str
    additional_evidence: Optional[str] = None

    # Multi-turn state (Task 3)
    coordinator_message: Optional[str] = None   # coordinator's last reply
    negotiation_history: list[dict]    = Field(default_factory=list)
    negotiation_turn: int = 0
    max_negotiation_turns: int = 5

    # Feedback
    last_action_feedback: Optional[str] = None
    last_action_valid: bool = True

    # Episode status
    done: bool = False
    episode_reward_so_far: float = 0.0


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class AuditRxState(BaseModel):
    """Full internal state (returned by state() endpoint)."""

    task_name: str
    episode_id: str
    step_number: int
    done: bool

    case: dict[str, Any]          # full case dict including ground truth
    ground_truth: dict[str, Any]  # correct answers (hidden from observation)

    rewards_history: list[float]
    cumulative_reward: float

    negotiation_history: list[dict]
    negotiation_turn: int

    agent_classifications: Optional[dict] = None
    agent_capa: Optional[dict]            = None
    final_escalation: Optional[str]       = None