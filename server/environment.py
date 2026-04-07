"""
AuditRx — Core Environment Class
Manages episode state, task routing, and reward accumulation.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Optional

from models import (
    ActionType,
    AuditRxAction,
    AuditRxObservation,
    AuditRxState,
    DeviationCategory,
    DeviationSeverity,
    EscalationDecision,
)
from tasks import (
    grade_capa,
    grade_classification,
    grade_escalation_decision,
    grade_negotiation_turn,
)
from coordinator import get_coordinator_response, get_initial_observation_message


# Load case bank
_CASES_PATH = Path(__file__).parent / "data" / "deviations.json"
_ALL_CASES: list[dict] = json.loads(_CASES_PATH.read_text())

_TASK_CASES: dict[str, list[dict]] = {
    "classify_deviation":    [c for c in _ALL_CASES if c["difficulty"] in ("easy",)],
    "draft_capa":            [c for c in _ALL_CASES if c["difficulty"] in ("easy", "medium")],
    "negotiate_escalation":  _ALL_CASES,  # all difficulties
}

MAX_STEPS_PER_TASK = {
    "classify_deviation":   3,
    "draft_capa":           5,
    "negotiate_escalation": 8,
}

MAX_NEGOTIATION_TURNS = 5


class AuditRxEnvironment:
    """
    Stateful environment for one episode.
    One instance per WebSocket session (managed by the server).
    """

    def __init__(self, task_name: str = "classify_deviation", seed: Optional[int] = None):
        self.task_name = task_name
        self._rng = random.Random(seed)
        self._episode_id: str = ""
        self._case: dict = {}
        self._step: int = 0
        self._done: bool = False
        self._rewards: list[float] = []
        self._negotiation_history: list[dict] = []
        self._negotiation_turn: int = 0
        self._agent_classifications: Optional[dict] = None
        self._agent_capa: Optional[dict] = None
        self._final_escalation: Optional[str] = None
        self._cumulative_reward: float = 0.0
        self._last_feedback: str = ""

    # ───────────────────────────── reset ──────────────────────────────

    def reset(self) -> AuditRxObservation:
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._done = False
        self._rewards = []
        self._negotiation_history = []
        self._negotiation_turn = 0
        self._agent_classifications = None
        self._agent_capa = None
        self._final_escalation = None
        self._cumulative_reward = 0.0
        self._last_feedback = ""

        # Pick a case appropriate for this task
        pool = _TASK_CASES.get(self.task_name, _ALL_CASES)
        self._case = self._rng.choice(pool)

        coordinator_msg = None
        if self.task_name == "negotiate_escalation":
            coordinator_msg = get_initial_observation_message(self._case)

        return self._build_observation(coordinator_message=coordinator_msg)

    # ───────────────────────────── step ───────────────────────────────

    def step(self, action: AuditRxAction) -> tuple[AuditRxObservation, float, bool, dict]:
        if self._done:
            obs = self._build_observation()
            return obs, 0.0, True, {"error": "Episode already done."}

        self._step += 1
        max_steps = MAX_STEPS_PER_TASK[self.task_name]
        reward = 0.0
        feedback = ""
        coordinator_msg = None

        # ── request_info costs a small step penalty ──────────────────
        if action.action_type == ActionType.REQUEST_INFO:
            penalty = -0.05
            reward = penalty
            info_given = self._reveal_additional_evidence()
            feedback = f"Additional evidence: {info_given}"
            self._rewards.append(reward)
            self._cumulative_reward += reward
            obs = self._build_observation(
                coordinator_message=coordinator_msg,
                last_feedback=feedback,
                last_valid=True,
            )
            self._check_done(max_steps)
            return obs, reward, self._done, {}

        # ── Task dispatch ────────────────────────────────────────────
        if self.task_name == "classify_deviation":
            reward, feedback = self._step_classify(action)
            self._done = True  # classification resolves in 1 action

        elif self.task_name == "draft_capa":
            reward, feedback = self._step_capa(action)
            self._done = True  # CAPA resolves in 1 action

        elif self.task_name == "negotiate_escalation":
            reward, feedback, coordinator_msg = self._step_negotiate(action)
            # done if escalation made or turns exhausted
            if self._final_escalation is not None:
                self._done = True
            elif self._negotiation_turn >= MAX_NEGOTIATION_TURNS:
                self._done = True
                feedback += " Max negotiation turns reached."
        else:
            feedback = f"Unknown task: {self.task_name}"

        # Max steps guard
        if self._step >= max_steps:
            self._done = True

        self._rewards.append(round(reward, 2))
        self._cumulative_reward = round(self._cumulative_reward + reward, 4)

        obs = self._build_observation(
            coordinator_message=coordinator_msg,
            last_feedback=feedback,
            last_valid=True,
        )
        return obs, round(reward, 2), self._done, {}

    # ───────────────────────────── state ──────────────────────────────

    def state(self) -> AuditRxState:
        return AuditRxState(
            task_name=self.task_name,
            episode_id=self._episode_id,
            step_number=self._step,
            done=self._done,
            case=self._case,
            ground_truth=self._case["ground_truth"],
            rewards_history=self._rewards,
            cumulative_reward=self._cumulative_reward,
            negotiation_history=self._negotiation_history,
            negotiation_turn=self._negotiation_turn,
            agent_classifications=self._agent_classifications,
            agent_capa=self._agent_capa,
            final_escalation=self._final_escalation,
        )

    # ───────────────────────────── task steps ─────────────────────────

    def _step_classify(self, action: AuditRxAction) -> tuple[float, str]:
        self._agent_classifications = {
            "severity": action.severity,
            "category": action.category,
        }
        score, feedback = grade_classification(
            agent_severity=action.severity,
            agent_category=action.category,
            ground_truth=self._case["ground_truth"],
        )
        return score, feedback

    def _step_capa(self, action: AuditRxAction) -> tuple[float, str]:
        self._agent_capa = {
            "root_cause":        action.root_cause,
            "corrective_action": action.corrective_action,
            "preventive_action": action.preventive_action,
            "timeline_days":     action.timeline_days,
        }
        score, feedback = grade_capa(
            root_cause=action.root_cause,
            corrective_action=action.corrective_action,
            preventive_action=action.preventive_action,
            timeline_days=action.timeline_days,
            ground_truth=self._case["ground_truth"],
        )
        return score, feedback

    def _step_negotiate(
        self, action: AuditRxAction
    ) -> tuple[float, str, Optional[str]]:

        gt = self._case["ground_truth"]
        coordinator_reply = None
        total_reward = 0.0
        feedback_parts = []

        # Agent sends a negotiation message
        if action.action_type == ActionType.NEGOTIATE and action.message:
            self._negotiation_turn += 1
            turn_score, turn_feedback = grade_negotiation_turn(action.message, gt)
            total_reward += turn_score
            feedback_parts.append(turn_feedback)

            # Get coordinator's adversarial response
            coordinator_reply = get_coordinator_response(
                case=self._case,
                negotiation_history=self._negotiation_history,
                agent_message=action.message,
                turn_number=self._negotiation_turn,
            )

            # Log to history
            self._negotiation_history.append({
                "turn": self._negotiation_turn,
                "agent_message": action.message,
                "coordinator_response": coordinator_reply,
            })

        # Agent makes final escalation decision
        if action.action_type == ActionType.ESCALATE and action.escalation:
            self._final_escalation = action.escalation
            esc_score, esc_feedback = grade_escalation_decision(
                agent_escalation=action.escalation,
                ground_truth=gt,
                negotiation_turns_used=self._negotiation_turn,
                max_negotiation_turns=MAX_NEGOTIATION_TURNS,
            )
            total_reward += esc_score
            feedback_parts.append(esc_feedback)

        # Combined NEGOTIATE + ESCALATE in same step
        if action.action_type == ActionType.NEGOTIATE and action.escalation:
            self._final_escalation = action.escalation
            esc_score, esc_feedback = grade_escalation_decision(
                agent_escalation=action.escalation,
                ground_truth=gt,
                negotiation_turns_used=self._negotiation_turn,
                max_negotiation_turns=MAX_NEGOTIATION_TURNS,
            )
            total_reward += esc_score
            feedback_parts.append(esc_feedback)

        return round(total_reward, 2), " ".join(feedback_parts), coordinator_reply

    # ───────────────────────────── helpers ────────────────────────────

    def _build_observation(
        self,
        coordinator_message: Optional[str] = None,
        last_feedback: Optional[str] = None,
        last_valid: bool = True,
    ) -> AuditRxObservation:
        c = self._case
        return AuditRxObservation(
            task_name=self.task_name,
            step_number=self._step,
            max_steps=MAX_STEPS_PER_TASK[self.task_name],
            case_id=c["case_id"],
            case_description=c["case_description"],
            protocol_id=c["protocol_id"],
            site_id=c["site_id"],
            patient_id=c["patient_id"],
            event_date=c["event_date"],
            reported_by=c["reported_by"],
            additional_evidence=c.get("additional_evidence"),
            coordinator_message=coordinator_message,
            negotiation_history=self._negotiation_history,
            negotiation_turn=self._negotiation_turn,
            max_negotiation_turns=MAX_NEGOTIATION_TURNS,
            last_action_feedback=last_feedback or self._last_feedback,
            last_action_valid=last_valid,
            done=self._done,
            episode_reward_so_far=self._cumulative_reward,
        )

    def _reveal_additional_evidence(self) -> str:
        return self._case.get("additional_evidence", "No additional evidence available.")

    def _check_done(self, max_steps: int) -> None:
        if self._step >= max_steps:
            self._done = True