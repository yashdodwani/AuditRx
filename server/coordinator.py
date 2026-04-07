"""
AuditRx — Simulated Adversarial Site Coordinator
Uses the LLM API to generate realistic pushback responses.
Falls back to scripted responses if API unavailable.
"""

from __future__ import annotations

import os
import textwrap
from typing import Optional

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


_API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
_API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
_MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

_COORDINATOR_SYSTEM_PROMPT = textwrap.dedent("""
You are a site coordinator at a clinical trial site being audited by a sponsor representative.
You are defensive, busy, and slightly resistant to escalation — not because you are corrupt,
but because you are overworked and believe your team did their best.

You will be given context about a protocol deviation and the agent's message to you.
You have an incorrect belief or justification that you will defend (provided in context).
Your job is to push back realistically: defend your team, offer alternative explanations,
minimize the severity, but remain professional.

IMPORTANT RULES:
- Stay in character as a site coordinator. Never break character.
- Keep your response under 150 words.
- Do NOT suddenly agree with the agent. Maintain your position unless the agent provides
  a very specific regulatory citation that directly contradicts your claim.
- You may soften slightly (acknowledge the concern) but never fully capitulate.
- Do not use bullet points. Respond in natural conversational prose.
""").strip()


def _build_coordinator_prompt(
    case_description: str,
    incorrect_claim: str,
    negotiation_history: list[dict],
    agent_message: str,
    turn_number: int,
) -> str:
    history_text = ""
    if negotiation_history:
        history_text = "\n".join(
            f"Turn {h['turn']} — Agent: {h['agent_message']}\nYour response: {h['coordinator_response']}"
            for h in negotiation_history
        )

    # On later turns, coordinator can soften slightly but not fully agree
    softening_hint = ""
    if turn_number >= 3:
        softening_hint = "You may acknowledge some validity to their concern, but still defend your team."

    return textwrap.dedent(f"""
        CASE CONTEXT:
        {case_description}

        YOUR POSITION (defend this):
        {incorrect_claim}

        PREVIOUS CONVERSATION:
        {history_text if history_text else "None — this is the first exchange."}

        AGENT'S LATEST MESSAGE:
        {agent_message}

        {softening_hint}

        Respond as the site coordinator now:
    """).strip()


def get_coordinator_response(
    case: dict,
    negotiation_history: list[dict],
    agent_message: str,
    turn_number: int,
) -> str:
    """
    Generate the site coordinator's adversarial response.
    Tries LLM first, falls back to scripted response.
    """
    ground_truth = case["ground_truth"]
    incorrect_claim = ground_truth["coordinator_incorrect_claim"]
    initial_pushback = ground_truth["coordinator_pushback"]

    # Turn 1: always use the scripted initial pushback for consistency/reproducibility
    if turn_number == 1:
        return initial_pushback

    # Subsequent turns: try LLM, fall back to escalating scripted responses
    if _OPENAI_AVAILABLE and _API_KEY and _API_KEY != "dummy":
        try:
            client = OpenAI(base_url=_API_BASE_URL, api_key=_API_KEY)
            prompt = _build_coordinator_prompt(
                case_description=case["case_description"],
                incorrect_claim=incorrect_claim,
                negotiation_history=negotiation_history,
                agent_message=agent_message,
                turn_number=turn_number,
            )
            completion = client.chat.completions.create(
                model=_MODEL_NAME,
                messages=[
                    {"role": "system", "content": _COORDINATOR_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.7,
                max_tokens=200,
                stream=False,
            )
            response = (completion.choices[0].message.content or "").strip()
            if response:
                return response
        except Exception:
            pass  # fall through to scripted

    # Scripted fallback responses by turn
    fallback_responses = {
        2: (
            f"I understand your concern, but I still believe {incorrect_claim.lower()} "
            "We've handled similar situations before and the sponsor has always been understanding."
        ),
        3: (
            "Look, I appreciate that you have to follow the rules, but you have to understand "
            "the pressure we're under. We're short-staffed and doing our best. "
            "Can we handle this at the site level without escalating?"
        ),
        4: (
            "Fine, I hear you. I'll discuss this with the PI. But I want it noted that "
            "we believe this was handled appropriately given the circumstances."
        ),
        5: (
            "We will comply with whatever the sponsor decides, but I want to formally note "
            "our disagreement with the characterization of this deviation."
        ),
    }
    return fallback_responses.get(turn_number, fallback_responses[5])


def get_initial_observation_message(case: dict) -> str:
    """Opening message from coordinator at episode start for Task 3."""
    return (
        f"Hi, I understand you're here to review the deviation for patient {case['patient_id']}. "
        f"I want to be upfront that our team followed the right procedures. "
        f"I'm happy to walk you through what happened."
    )