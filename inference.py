"""
AuditRx — Baseline Inference Script
=====================================
MANDATORY environment variables:
  API_BASE_URL      LLM endpoint
  MODEL_NAME        Model identifier
  HF_TOKEN          Hugging Face / API key
  AUDITRX_BASE_URL  The deployed HF Space URL (e.g. https://your-space.hf.space)

Runs all 3 tasks sequentially and emits structured stdout logs.

Usage:
  python inference.py
  AUDITRX_TASK=classify_deviation python inference.py
"""

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import OpenAI

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────


def _load_env_file() -> None:
    """Load simple KEY=VALUE entries from a local .env file if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Respect already-exported shell variables.
        os.environ.setdefault(key, value)


_load_env_file()

API_KEY = (os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")).strip()
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("AUDITRX_BASE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "auditrx"

# Run only one task if specified, else all three
SINGLE_TASK = os.getenv("AUDITRX_TASK", "")

TASKS = [
    {"name": "classify_deviation", "max_steps": 3},
    {"name": "draft_capa", "max_steps": 5},
    {"name": "negotiate_escalation", "max_steps": 8},
]
if SINGLE_TASK:
    TASKS = [t for t in TASKS if t["name"] == SINGLE_TASK]

TEMPERATURE = 0.2
MAX_TOKENS = 600


# ─────────────────────────────────────────────
# Logging helpers  (mandatory format)
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────

_CLIENT: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    return _CLIENT


def call_llm(system: str, user: str) -> str:
    try:
        resp = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ─────────────────────────────────────────────
# Task 1: classify_deviation
# ─────────────────────────────────────────────

CLASSIFY_SYSTEM = textwrap.dedent("""
You are a clinical trial compliance expert with deep knowledge of ICH-GCP, FDA 21 CFR,
and protocol deviation management.

You will be given a protocol deviation case. Classify it.

Respond ONLY with a JSON object, no preamble, no markdown fences:
{
  "action_type": "classify",
  "severity": "<critical|major|minor>",
  "category": "<informed_consent|eligibility_criteria|dosing_error|visit_window|data_integrity|safety_reporting|randomization>"
}

Severity definitions:
- critical: immediate patient safety risk or significant regulatory non-compliance
- major: significant protocol violation with potential impact on data integrity or patient welfare
- minor: minor deviation with minimal impact, self-correcting

Category definitions:
- informed_consent: issues with consent process or documentation
- eligibility_criteria: patient did not meet inclusion/exclusion criteria
- dosing_error: incorrect dose, frequency, or drug dispensed
- visit_window: visit occurred outside protocol-specified window
- data_integrity: data recording, audit trail, or source document issues
- safety_reporting: failure to report or late reporting of safety events
- randomization: errors in the randomization process
""").strip()


def classify_action(obs: dict) -> dict:
    case_text = (
        f"Case: {obs['case_description']}\n"
        f"Evidence: {obs.get('additional_evidence', 'None')}"
    )
    raw = call_llm(CLASSIFY_SYSTEM, case_text)
    parsed = parse_json_response(raw)
    return {
        "action_type": "classify",
        "severity": parsed.get("severity", "minor"),
        "category": parsed.get("category", "visit_window"),
    }


# ─────────────────────────────────────────────
# Task 2: draft_capa
# ─────────────────────────────────────────────

CAPA_SYSTEM = textwrap.dedent("""
You are a clinical trial quality assurance expert drafting a CAPA (Corrective and Preventive Action)
response for a protocol deviation.

Respond ONLY with a JSON object:
{
  "action_type": "draft_capa",
  "root_cause": "<concise root cause analysis, max 500 chars>",
  "corrective_action": "<immediate corrective steps, max 800 chars>",
  "preventive_action": "<systemic preventive measures, max 800 chars>",
  "timeline_days": <integer, realistic days to complete CAPA>
}

Root cause must identify the fundamental cause, not just describe the event.
Corrective action must address immediate remediation.
Preventive action must prevent recurrence systemically.
Timeline must be realistic and proportionate to severity.
""").strip()


def capa_action(obs: dict) -> dict:
    case_text = (
        f"Case: {obs['case_description']}\n"
        f"Evidence: {obs.get('additional_evidence', 'None')}"
    )
    raw = call_llm(CAPA_SYSTEM, case_text)
    parsed = parse_json_response(raw)
    return {
        "action_type": "draft_capa",
        "root_cause": parsed.get("root_cause", "Root cause not identified."),
        "corrective_action": parsed.get("corrective_action", "Corrective action not specified."),
        "preventive_action": parsed.get("preventive_action", "Preventive action not specified."),
        "timeline_days": int(parsed.get("timeline_days", 30)),
    }


# ─────────────────────────────────────────────
# Task 3: negotiate_escalation
# ─────────────────────────────────────────────

NEGOTIATE_SYSTEM = textwrap.dedent("""
You are a sponsor representative auditing a clinical trial site.
You are reviewing a protocol deviation with the site coordinator who is pushing back.

Your role:
1. Professionally rebut incorrect regulatory claims using specific ICH-GCP/FDA citations.
2. Maintain a firm but respectful tone.
3. After gathering enough information (usually 1-2 turns), make a final escalation decision.

Available escalation decisions:
- self_correct: site handles internally, no external reporting needed
- sponsor_notify: sponsor must be notified within 24h
- irb_report: mandatory IRB/ethics board report required
- halt_site: suspend enrollment immediately

When ready to escalate, respond with JSON:
{
  "action_type": "escalate",
  "message": "<your final statement to the coordinator>",
  "escalation": "<self_correct|sponsor_notify|irb_report|halt_site>"
}

When negotiating (not ready to escalate yet), respond with JSON:
{
  "action_type": "negotiate",
  "message": "<your response to the coordinator citing specific regulations>"
}
""").strip()


def negotiate_action(obs: dict, step: int) -> dict:
    history = obs.get("negotiation_history", [])
    coordinator_msg = obs.get("coordinator_message", "")

    history_text = ""
    if history:
        history_text = "\n".join(
            f"Turn {h['turn']}:\n  You: {h['agent_message']}\n  Coordinator: {h['coordinator_response']}"
            for h in history
        )

    max_turns = obs.get("max_negotiation_turns", 5)
    turns_used = obs.get("negotiation_turn", 0)
    turns_remaining = max_turns - turns_used

    user_prompt = textwrap.dedent(f"""
        DEVIATION CASE:
        {obs['case_description']}

        ADDITIONAL EVIDENCE:
        {obs.get('additional_evidence', 'None')}

        CONVERSATION HISTORY:
        {history_text if history_text else 'None — this is the opening.'}

        COORDINATOR'S LATEST MESSAGE:
        {coordinator_msg}

        Turns remaining: {turns_remaining}

        {"IMPORTANT: This is your last chance to respond. You MUST include an escalation decision." if turns_remaining <= 1 else ""}

        Respond with JSON only.
    """).strip()

    raw = call_llm(NEGOTIATE_SYSTEM, user_prompt)
    parsed = parse_json_response(raw)

    action_type = parsed.get("action_type", "negotiate")
    message = parsed.get("message", "I need to review the regulatory requirements and will follow up.")
    escalation = parsed.get("escalation")

    # Force escalation on last step
    if turns_remaining <= 1 and not escalation:
        action_type = "escalate"
        escalation = "sponsor_notify"  # safe default

    return {
        "action_type": action_type,
        "message": message[:1000],
        **({"escalation": escalation} if escalation else {}),
    }


# ─────────────────────────────────────────────
# Environment HTTP client
# ─────────────────────────────────────────────

async def env_reset(client: httpx.AsyncClient, task_name: str) -> tuple[str, dict]:
    resp = await client.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["session_id"], data["observation"]


async def env_step(client: httpx.AsyncClient, session_id: str, action: dict) -> tuple[dict, float, bool]:
    resp = await client.post(
        f"{ENV_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["observation"], data["reward"], data["done"]


async def env_close(client: httpx.AsyncClient, session_id: str) -> None:
    try:
        await client.delete(f"{ENV_BASE_URL}/close/{session_id}", timeout=10)
    except Exception:
        pass


# ─────────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────────

async def run_task(task: dict) -> float:
    task_name = task["name"]
    max_steps = task["max_steps"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    async with httpx.AsyncClient() as http:
        session_id = None
        try:
            session_id, obs = await env_reset(http, task_name)

            for step in range(1, max_steps + 1):
                if obs.get("done"):
                    break

                # Choose action based on task
                if task_name == "classify_deviation":
                    action = classify_action(obs)
                elif task_name == "draft_capa":
                    action = capa_action(obs)
                else:  # negotiate_escalation
                    action = negotiate_action(obs, step)

                action_str = json.dumps(action, separators=(",", ":"))[:200]

                obs, reward, done = await env_step(http, session_id, action)

                error = obs.get("last_action_feedback") if not obs.get("last_action_valid", True) else None
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                rewards.append(round(reward, 2))
                steps_taken = step

                if done:
                    break

            score = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= 0.3

        except Exception as exc:
            print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
        finally:
            if session_id:
                await env_close(http, session_id)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

async def main() -> None:
    all_scores = []
    for task in TASKS:
        score = await run_task(task)
        all_scores.append(score)
        print(f"[SUMMARY] {task['name']}: {score:.3f}", flush=True)

    if len(all_scores) > 1:
        avg = sum(all_scores) / len(all_scores)
        print(f"[SUMMARY] Overall average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())