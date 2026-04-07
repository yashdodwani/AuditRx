# AuditRx 🏥⚖️
### Clinical Trial Protocol Deviation Triage — OpenEnv Environment

AuditRx is an OpenEnv-compliant reinforcement learning environment where AI agents triage **real-world clinical trial protocol deviations** — a genuinely high-stakes regulatory task performed daily by thousands of CRAs, clinical data managers, and QA specialists worldwide.

---

## Why This Exists

Clinical trial protocol deviations must be classified by severity, investigated with root cause analysis, documented in CAPA reports, and escalated appropriately under ICH-GCP and FDA regulations. Getting this wrong delays trials, triggers FDA warning letters, or—in critical cases—harms patients. No existing OpenEnv environment models this domain.

---

## Tasks

### Task 1: `classify_deviation` (Easy)
The agent reads a realistic synthetic deviation report and classifies it by:
- **Severity**: `critical` / `major` / `minor`
- **Category**: `informed_consent` / `eligibility_criteria` / `dosing_error` / `visit_window` / `data_integrity` / `safety_reporting` / `randomization`

Scoring: 0.5 per correct field → max 1.0. Partial credit.

### Task 2: `draft_capa` (Medium)
The agent drafts a complete **CAPA (Corrective and Preventive Action)** response including:
- Root cause statement
- Corrective actions
- Preventive actions
- Implementation timeline (days)

Scoring: Weighted keyword coverage across all 4 components. Partial credit throughout.

### Task 3: `negotiate_escalation` (Hard)
The agent must **negotiate with an adversarial simulated site coordinator** who pushes back with plausible-but-incorrect regulatory justifications, then make the correct escalation decision:

`self_correct` → `sponsor_notify` → `irb_report` → `halt_site`

Scoring:
- Per-turn regulatory rebuttal quality (up to 0.40 per turn)
- Correct final escalation decision (0.40)
- Efficiency bonus for resolving quickly (0.20)
- Penalty of −0.05 per `request_info` action

---

## Action Space

```json
{
  "action_type": "classify | draft_capa | negotiate | escalate | request_info",
  "severity": "critical | major | minor",
  "category": "informed_consent | eligibility_criteria | dosing_error | visit_window | data_integrity | safety_reporting | randomization",
  "root_cause": "string (max 500 chars)",
  "corrective_action": "string (max 800 chars)",
  "preventive_action": "string (max 800 chars)",
  "timeline_days": 1–180,
  "message": "string (max 1000 chars) — negotiation message",
  "escalation": "self_correct | sponsor_notify | irb_report | halt_site"
}
```

## Observation Space

```json
{
  "task_name": "string",
  "step_number": 1,
  "max_steps": 8,
  "case_id": "DEV-2024-001",
  "case_description": "Full deviation narrative...",
  "protocol_id": "ONCO-301-Ph3",
  "site_id": "SITE-042",
  "patient_id": "PT-042-0017",
  "event_date": "2024-03-12",
  "reported_by": "Dr. Sarah Kim, Site PI",
  "additional_evidence": "string | null",
  "coordinator_message": "string | null (Task 3 only)",
  "negotiation_history": [...],
  "last_action_feedback": "string | null",
  "done": false,
  "episode_reward_so_far": 0.0
}
```

---

## Setup

### Prerequisites
- Docker
- Python 3.11+
- HF account + Space

### Run locally

```bash
# Build and run server
cd server
docker build -t auditrx .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  auditrx

# Run baseline inference (in another terminal)
pip install -r requirements.txt
export AUDITRX_BASE_URL=http://localhost:7860
export HF_TOKEN=your_token
python inference.py
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM endpoint URL | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace / API key | — |
| `AUDITRX_BASE_URL` | Deployed environment URL | `http://localhost:7860` |
| `AUDITRX_TASK` | Run a single task only | (all 3) |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode |
| `/step` | POST | Take action |
| `/state/{session_id}` | GET | Get full internal state |
| `/close/{session_id}` | DELETE | End session |
| `/ws` | WebSocket | Full WS interface |
| `/health` | GET | Health check |

---

## Baseline Scores

| Task | Model | Score |
|---|---|---|
| classify_deviation | Qwen2.5-72B | ~0.75 |
| draft_capa | Qwen2.5-72B | ~0.52 |
| negotiate_escalation | Qwen2.5-72B | ~0.38 |

---

## Case Bank

6 synthetic cases modeled on real ICH-GCP/FDA deviation types:
- Visit window exceedance (minor)
- Pre-consent screening procedures (critical)  
- Pharmacy dispensing double-dose (critical)
- Ineligible patient enrolled (major)
- Data integrity / potential fabrication (critical)
- Failure to report safety event (critical)

All cases include adversarial coordinator pushback scripts with incorrect regulatory claims and ground-truth rebuttals.

---

## Reward Design

Rewards emit partial signal at every step — no sparse end-of-episode binary. This makes AuditRx suitable for RL training (GRPO, PPO) as well as evaluation benchmarking.