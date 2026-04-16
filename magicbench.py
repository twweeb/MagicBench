#!/usr/bin/env python3
"""
MagicBench v1.0: A Deception-Sensitive Cognitive Benchmark for LLMs
====================================================================
Tests adversarial reasoning, metacognition, theory of mind, and causal
inference through magic-trick understanding — aligned with DeepMind's
AGI cognitive framework (perception, attention, reasoning, metacognition,
social cognition, executive functions, learning, memory, problem solving,
generation).

Usage:
    python magicbench.py --model gpt-4o --provider openai --api-key $KEY
    python magicbench.py --dry-run          # print prompts only
    python magicbench.py --human            # interactive human baseline mode

Outputs:
    results/<model>_<timestamp>.json   — per-item scores
    results/<model>_<timestamp>_profile.json — cognitive profile (5 dimensions)
"""

import json, os, re, time, argparse, random, importlib
from dataclasses import dataclass, asdict
from typing import List, Dict
from enum import Enum
from datetime import datetime
from collections import defaultdict

# ════════════════════════════════════════════════════════════════
# §1  ENUMERATIONS & DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

class EffectType(str, Enum):
    VANISH = "vanish"
    APPEARANCE = "appearance"
    CONTROL = "control"
    TRANSPOSITION = "transposition"
    TRANSFORMATION = "transformation"
    LEVITATION = "levitation"
    PENETRATION = "penetration"
    PREDICTION = "prediction"
    MENTALISM = "mentalism"
    RESTORATION = "restoration"

class ViolationType(str, Enum):
    OBJECT_PERMANENCE = "object_permanence"
    SPATIOTEMPORAL = "spatiotemporal_continuity"
    SUPPORT_GRAVITY = "support_gravity"
    CAUSAL_CHAIN = "causal_chain"
    INFO_ACCESS = "information_access"
    FREE_WILL = "free_will"
    MATERIAL_INTEGRITY = "material_integrity"

class MethodFamily(str, Enum):
    CONCEALMENT = "concealment"
    SUBSTITUTION = "substitution"
    FORCING = "forcing"
    MISDIRECTION_ATT = "attention_misdirection"
    MISDIRECTION_MEM = "memory_misdirection"
    GIMMICK = "gimmick"
    MATHEMATICAL = "mathematical"
    PSYCHOLOGICAL = "psychological"
    DUAL_REALITY = "dual_reality"
    PRE_SHOW = "pre_show"
    MULTIPLE_OUTS = "multiple_outs"

# DeepMind 10 cognitive faculties
class CognitiveFaculty(str, Enum):
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    REASONING = "reasoning"
    METACOGNITION = "metacognition"
    EXECUTIVE = "executive_functions"
    SOCIAL = "social_cognition"
    LEARNING = "learning"
    PROBLEM_SOLVING = "problem_solving"
    GENERATION = "generation"

@dataclass
class BeliefState:
    """What the audience rationally believes at a given step."""
    step: int
    observable_event: str
    audience_belief: str
    actual_reality: str

    @classmethod
    def from_dict(cls, data: dict) -> "BeliefState":
        return cls(**data)

@dataclass
class CounterfactualQ:
    condition: str          # what changes
    question: str
    correct_answer: str     # "yes" / "no" / short phrase
    explanation: str

    @classmethod
    def from_dict(cls, data: dict) -> "CounterfactualQ":
        return cls(**data)

@dataclass
class MagicScenario:
    id: str
    title: str
    effect_type: EffectType
    description: str                        # audience-perspective narrative
    key_moments: List[str]                  # critical observable events
    violation_types: List[ViolationType]
    method_families: List[MethodFamily]     # gold (may be >1)
    method_abstract: str                    # abstract explanation, no secrets
    belief_trace: List[BeliefState]
    counterfactuals: List[CounterfactualQ]
    difficulty: Dict[str, int]              # axis name → 1-5
    primary_faculties: List[CognitiveFaculty]

    @classmethod
    def from_dict(cls, data: dict) -> "MagicScenario":
        return cls(
            id=data["id"],
            title=data["title"],
            effect_type=EffectType(data["effect_type"]),
            description=data["description"],
            key_moments=data["key_moments"],
            violation_types=[ViolationType(v) for v in data["violation_types"]],
            method_families=[MethodFamily(m) for m in data["method_families"]],
            method_abstract=data["method_abstract"],
            belief_trace=[BeliefState.from_dict(b) for b in data["belief_trace"]],
            counterfactuals=[CounterfactualQ.from_dict(c) for c in data["counterfactuals"]],
            difficulty=data["difficulty"],
            primary_faculties=[CognitiveFaculty(f) for f in data["primary_faculties"]],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "effect_type": self.effect_type.value,
            "description": self.description,
            "key_moments": self.key_moments,
            "violation_types": [v.value for v in self.violation_types],
            "method_families": [m.value for m in self.method_families],
            "method_abstract": self.method_abstract,
            "belief_trace": [asdict(b) for b in self.belief_trace],
            "counterfactuals": [asdict(c) for c in self.counterfactuals],
            "difficulty": self.difficulty,
            "primary_faculties": [f.value for f in self.primary_faculties],
        }


HF_DATASET_NAME = "hsiung/MagicBench"
HF_DATASET_SPLIT = "test"


def load_scenarios(dataset_name: str = HF_DATASET_NAME,
                   split: str = HF_DATASET_SPLIT) -> List[MagicScenario]:
    """Load benchmark scenarios from the Hugging Face dataset."""
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as e:
        raise RuntimeError(
            "Loading scenarios requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from e

    data = datasets_module.load_dataset(dataset_name, split=split)
    return [MagicScenario.from_dict(dict(item)) for item in data]

# ════════════════════════════════════════════════════════════════
# §2  DATASET
# ════════════════════════════════════════════════════════════════

SCENARIOS: List[MagicScenario] = load_scenarios()

# ════════════════════════════════════════════════════════════════
# §3  TASK GENERATORS — 6 task types per scenario
# ════════════════════════════════════════════════════════════════

class TaskType(str, Enum):
    EFFECT_RECOGNITION = "effect_recognition"
    VIOLATION_ID = "violation_identification"
    BEST_EXPLANATION = "best_explanation"
    BELIEF_TRACE = "belief_trace"
    CALIBRATION = "metacognitive_calibration"
    COUNTERFACTUAL = "counterfactual_reasoning"


def _merge_faculties(task_base: List[str], scenario: MagicScenario) -> List[str]:
    """Combine task-type intrinsic faculties with scenario-specific faculties.

    Task-type faculties capture WHAT cognitive operation is performed.
    Scenario faculties capture WHICH domain knowledge is required.
    The union (deduplicated, ordered) gives a richer per-item profile.
    """
    scenario_facs = [f.value for f in scenario.primary_faculties]
    merged = list(dict.fromkeys(task_base + scenario_facs))  # dedup, preserve order
    return merged


def generate_effect_recognition(sc: MagicScenario) -> dict:
    """Task 1: What effect type did the audience experience?"""
    options = list({sc.effect_type.value} | set(random.sample(
        [e.value for e in EffectType if e != sc.effect_type],
        min(5, len(EffectType) - 1)
    )))
    random.shuffle(options)
    task_base = [CognitiveFaculty.PERCEPTION.value, CognitiveFaculty.MEMORY.value]
    return {
        "task_type": TaskType.EFFECT_RECOGNITION.value,
        "scenario_id": sc.id,
        "prompt": (
            f"Read the following magic trick description:\n\n"
            f'"{sc.description}"\n\n'
            f"From the audience's perspective, which category of magic effect "
            f"best describes what they witnessed?\n\n"
            f"Options:\n" +
            "\n".join(f"  ({chr(65+i)}) {o}" for i, o in enumerate(options)) +
            "\n\nRespond with ONLY the letter of your answer (e.g., 'A')."
        ),
        "gold": chr(65 + options.index(sc.effect_type.value)),
        "gold_value": sc.effect_type.value,
        "options": options,
        "faculties": _merge_faculties(task_base, sc),
    }


def generate_violation_id(sc: MagicScenario) -> dict:
    """Task 2: Which types of violation does this trick exploit?"""
    all_types = [v.value for v in ViolationType]
    gold_set = set(v.value for v in sc.violation_types)
    task_base = [CognitiveFaculty.REASONING.value, CognitiveFaculty.MEMORY.value]
    return {
        "task_type": TaskType.VIOLATION_ID.value,
        "scenario_id": sc.id,
        "prompt": (
            f"Read the following magic trick description:\n\n"
            f'"{sc.description}"\n\n'
            f"Which of the following expectations or rules does this trick "
            f"appear to violate? Select ALL that apply.\n\n" +
            "\n".join(f"  ({chr(65+i)}) {v}" for i, v in enumerate(all_types)) +
            "\n\nRespond with the letters of ALL correct answers separated "
            "by commas (e.g., 'A, C, E'). Select only those that directly apply."
        ),
        "gold": ", ".join(sorted(
            chr(65 + all_types.index(v)) for v in gold_set
        )),
        "gold_set": sorted(gold_set),
        "options": all_types,
        "faculties": _merge_faculties(task_base, sc),
    }


def generate_best_explanation(sc: MagicScenario) -> dict:
    """Task 3: Open-ended — explain how the trick works."""
    task_base = [CognitiveFaculty.REASONING.value, CognitiveFaculty.PROBLEM_SOLVING.value, CognitiveFaculty.GENERATION.value]
    return {
        "task_type": TaskType.BEST_EXPLANATION.value,
        "scenario_id": sc.id,
        "prompt": (
            f"Read the following magic trick description:\n\n"
            f'"{sc.description}"\n\n'
            f"How does this trick actually work? Explain the most likely "
            f"method the performer uses to achieve this effect. Be specific "
            f"about what is secretly happening and when."
        ),
        "gold": sc.method_abstract,
        "faculties": _merge_faculties(task_base, sc),
    }


def generate_belief_trace(sc: MagicScenario) -> dict:
    """Task 4: Open-ended — describe audience belief at EVERY step.
    Score 1.0 only if ALL steps are correct."""
    task_base = [CognitiveFaculty.SOCIAL.value, CognitiveFaculty.MEMORY.value, CognitiveFaculty.REASONING.value]
    steps_text = "\n".join(
        f"  Step {s.step}: \"{s.observable_event}\""
        for s in sc.belief_trace
    )
    return {
        "task_type": TaskType.BELIEF_TRACE.value,
        "scenario_id": sc.id,
        "prompt": (
            f"Read the following magic trick description:\n\n"
            f'"{sc.description}"\n\n'
            f"At each of the following moments, describe what a typical "
            f"audience member most likely BELIEVES is happening. (Not what "
            f"is actually happening behind the scenes, but what they "
            f"sincerely believe based on what they can see.)\n\n"
            f"{steps_text}\n\n"
            f"For each step, write 1-3 sentences describing the audience's "
            f"belief. Use the format:\n"
            f"Step 1: <belief>\n"
            f"Step 2: <belief>\n"
            f"..."
        ),
        "gold_steps": [
            {
                "step": s.step,
                "observable_event": s.observable_event,
                "audience_belief": s.audience_belief,
                "actual_reality": s.actual_reality,
            }
            for s in sc.belief_trace
        ],
        "n_steps": len(sc.belief_trace),
        "faculties": _merge_faculties(task_base, sc),
    }


def generate_calibration(sc: MagicScenario) -> dict:
    """Task 5: Open-ended metacognitive calibration — explain what you know
    and don't know about the method, and rate your own confidence."""
    task_base = [CognitiveFaculty.METACOGNITION.value, CognitiveFaculty.REASONING.value]
    return {
        "task_type": TaskType.CALIBRATION.value,
        "scenario_id": sc.id,
        "prompt": (
            f"Read the following magic trick description:\n\n"
            f'"{sc.description}"\n\n'
            f"Explain what you think is the most likely method behind this "
            f"trick. Then honestly assess:\n"
            f"1. How confident are you in your explanation? (0-100%)\n"
            f"2. What aspects are you most uncertain about?\n"
            f"3. What alternative explanations could also be plausible?\n\n"
            f"Be honest about the limits of your reasoning."
        ),
        "gold": sc.method_abstract,
        "gold_method_families": [m.value for m in sc.method_families],
        "faculties": _merge_faculties(task_base, sc),
    }


def generate_counterfactual(sc: MagicScenario) -> dict:
    """Task 6: Open-ended — reason about ALL counterfactual conditions.
    Score 1.0 only if ALL sub-questions are correct."""
    task_base = [CognitiveFaculty.REASONING.value, CognitiveFaculty.EXECUTIVE.value]
    cf_text = "\n\n".join(
        f"  Scenario {i+1}:\n"
        f"  Condition: \"{cf.condition}\"\n"
        f"  Question: {cf.question}"
        for i, cf in enumerate(sc.counterfactuals)
    )
    return {
        "task_type": TaskType.COUNTERFACTUAL.value,
        "scenario_id": sc.id,
        "prompt": (
            f"Read the following magic trick description:\n\n"
            f'"{sc.description}"\n\n'
            f"Now consider each of the following hypothetical changes. For "
            f"each one, give your answer and explain your reasoning.\n\n"
            f"{cf_text}\n\n"
            f"For each scenario, use the format:\n"
            f"Scenario 1: <answer and explanation>\n"
            f"Scenario 2: <answer and explanation>\n"
            f"..."
        ),
        "gold_counterfactuals": [
            {
                "condition": cf.condition,
                "question": cf.question,
                "correct_answer": cf.correct_answer,
                "explanation": cf.explanation,
            }
            for cf in sc.counterfactuals
        ],
        "n_counterfactuals": len(sc.counterfactuals),
        "faculties": _merge_faculties(task_base, sc),
    }


TASK_GENERATORS = {
    TaskType.EFFECT_RECOGNITION: generate_effect_recognition,
    TaskType.VIOLATION_ID: generate_violation_id,
    TaskType.BEST_EXPLANATION: generate_best_explanation,
    TaskType.BELIEF_TRACE: generate_belief_trace,
    TaskType.CALIBRATION: generate_calibration,
    TaskType.COUNTERFACTUAL: generate_counterfactual,
}


def build_all_tasks(scenarios=None, seed=42) -> List[dict]:
    """Generate all tasks for all scenarios. Returns list of task dicts."""
    random.seed(seed)
    scenarios = scenarios or SCENARIOS
    tasks = []
    for sc in scenarios:
        for gen_fn in TASK_GENERATORS.values():
            generated = gen_fn(sc)
            generated_tasks = generated if isinstance(generated, list) else [generated]
            for task in generated_tasks:
                task["difficulty"] = sc.difficulty
                task["primary_faculties_scenario"] = [f.value for f in sc.primary_faculties]
                tasks.append(task)
    return tasks


# ════════════════════════════════════════════════════════════════
# §4  SCORING FUNCTIONS
# ════════════════════════════════════════════════════════════════

# ── LLM-as-Judge for open-ended tasks ────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair judge evaluating whether a student's answer "
    "to a magic-trick analysis question is semantically equivalent to the "
    "reference answer. Two answers are 'semantically equivalent' if they "
    "convey the same core mechanism, conclusion, or insight — even if the "
    "wording, structure, or level of detail differs.\n\n"
    "Rules:\n"
    "- Focus on whether the KEY MECHANISM or KEY CONCLUSION matches.\n"
    "- Ignore differences in writing style, length, or extra details, "
    "as long as the core meaning is present.\n"
    "- If the student's answer contains the correct core idea PLUS "
    "additional wrong ideas, still score 1 if the correct core is clearly "
    "the primary answer.\n"
    "- If the student's answer is vague, generic, or only partially "
    "overlaps with the reference, score 0.\n"
    "- If the student's answer identifies a genuinely valid alternative "
    "method that could produce the same effect (even if different from "
    "the reference), score 1. Magic tricks can have multiple valid methods.\n\n"
    "You MUST respond with ONLY a JSON object: {\"score\": 1} or {\"score\": 0}\n"
    "Do NOT include any other text."
)

JUDGE_PROMPT_EXPLANATION = (
    "Task: Does the student correctly explain how this magic trick works?\n\n"
    "Trick description:\n\"{description}\"\n\n"
    "Reference method (gold answer):\n\"{gold}\"\n\n"
    "Student's explanation:\n\"{response}\"\n\n"
    "Is the student's explanation semantically equivalent to the reference? "
    "The student must identify the correct core mechanism (e.g., concealment, "
    "substitution, forcing, gimmick, etc.) and correctly describe WHEN and "
    "HOW the secret action occurs. Minor details can differ.\n\n"
    "Respond with ONLY: {{\"score\": 1}} or {{\"score\": 0}}"
)

JUDGE_PROMPT_BELIEF = (
    "Task: Does the student correctly describe what a typical audience "
    "member BELIEVES at a specific moment during a magic trick?\n\n"
    "Trick description:\n\"{description}\"\n\n"
    "Moment in question:\n\"{moment}\"\n\n"
    "Reference audience belief (gold answer):\n\"{gold}\"\n\n"
    "Student's answer:\n\"{response}\"\n\n"
    "Is the student's description of the audience's belief semantically "
    "equivalent to the reference? The student must describe what the "
    "audience THINKS is happening (their naive belief), NOT what is "
    "actually happening behind the scenes. If the student describes the "
    "reality instead of the belief, score 0.\n\n"
    "Respond with ONLY: {{\"score\": 1}} or {{\"score\": 0}}"
)

JUDGE_PROMPT_CALIBRATION = (
    "Task: Does the student correctly identify the method AND show "
    "appropriate metacognitive awareness?\n\n"
    "Trick description:\n\"{description}\"\n\n"
    "Reference method (gold answer):\n\"{gold}\"\n\n"
    "Known valid method families: {method_families}\n\n"
    "Student's answer:\n\"{response}\"\n\n"
    "Score 1 if BOTH conditions are met:\n"
    "  (a) The student's primary explanation is semantically equivalent "
    "to the reference method (identifies the correct core mechanism), AND\n"
    "  (b) The student shows some metacognitive awareness — acknowledging "
    "uncertainty, mentioning limitations, or noting alternatives — rather "
    "than being blindly overconfident about a wrong answer.\n"
    "Score 0 if the core mechanism is wrong, OR if the student is "
    "confidently wrong with no hedging.\n\n"
    "Respond with ONLY: {{\"score\": 1}} or {{\"score\": 0}}"
)

JUDGE_PROMPT_COUNTERFACTUAL = (
    "Task: Does the student correctly reason about what happens when a "
    "condition of a magic trick is changed?\n\n"
    "Trick description:\n\"{description}\"\n\n"
    "Hypothetical change:\n\"{condition}\"\n\n"
    "Question:\n\"{question}\"\n\n"
    "Reference answer (gold):\n\"{gold_answer}\"\n\n"
    "Reference explanation:\n\"{gold_explanation}\"\n\n"
    "Student's answer:\n\"{response}\"\n\n"
    "Is the student's answer semantically equivalent to the reference? "
    "The student must reach the same core conclusion (e.g., 'yes the trick "
    "still works' vs 'no it would fail') AND give a reasoning that aligns "
    "with the reference explanation. If the conclusion matches but the "
    "reasoning is wrong or absent, score 0.\n\n"
    "Respond with ONLY: {{\"score\": 1}} or {{\"score\": 0}}"
)


def call_judge(prompt: str, judge_model: str = "claude-sonnet-4-20250514",
               judge_provider: str = "anthropic",
               judge_api_key: str = "") -> int:
    """Call the judge LLM and extract a binary score."""
    try:
        response = call_llm(
            prompt, judge_model, judge_provider, judge_api_key,
            temperature=0.0, max_tokens=64
        )
        # Extract JSON score
        match = re.search(r'"score"\s*:\s*([01])', response)
        if match:
            return int(match.group(1))
        # Fallback: look for bare 0 or 1
        stripped = response.strip()
        if stripped in ("0", "1"):
            return int(stripped)
        return 0
    except Exception as e:
        print(f"  JUDGE ERROR: {e}")
        return 0


def _extract_step_response(full_response: str, step_num: int,
                           total_steps: int, prefix: str = "Step") -> str:
    """Extract the response segment for a specific numbered step/scenario.

    Looks for 'Step N:' or 'Scenario N:' markers. Falls back to splitting
    by blank lines or returning the full response if parsing fails.
    """
    pattern = rf'(?:^|\n)\s*{prefix}\s*{step_num}\s*[:\-\.]\s*(.*?)(?=\n\s*{prefix}\s*\d|\Z)'
    match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: try splitting by numbered markers generically
    parts = re.split(rf'\n\s*{prefix}\s*\d+\s*[:\-\.]', full_response, flags=re.IGNORECASE)
    if len(parts) > step_num:
        return parts[step_num].strip()

    # Last resort: return everything (judge will evaluate holistically)
    return full_response.strip()


def score_with_judge(task: dict, response: str, description: str,
                     judge_model: str = "claude-sonnet-4-20250514",
                     judge_provider: str = "anthropic",
                     judge_api_key: str = "") -> float:
    """Route to the correct judge prompt template and score.

    For BELIEF_TRACE and COUNTERFACTUAL: evaluates each sub-item
    independently, returns 1.0 only if ALL sub-items score 1.
    """
    tt = task["task_type"]

    if tt == TaskType.BEST_EXPLANATION.value:
        prompt = JUDGE_PROMPT_EXPLANATION.format(
            description=description,
            gold=task["gold"],
            response=response,
        )
        full_prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + prompt
        return float(call_judge(full_prompt, judge_model, judge_provider, judge_api_key))

    elif tt == TaskType.CALIBRATION.value:
        prompt = JUDGE_PROMPT_CALIBRATION.format(
            description=description,
            gold=task["gold"],
            method_families=", ".join(task.get("gold_method_families", [])),
            response=response,
        )
        full_prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + prompt
        return float(call_judge(full_prompt, judge_model, judge_provider, judge_api_key))

    elif tt == TaskType.BELIEF_TRACE.value:
        gold_steps = task.get("gold_steps", [])
        n = len(gold_steps)
        if n == 0:
            return 0.0

        all_pass = True
        for gs in gold_steps:
            step_response = _extract_step_response(
                response, gs["step"], n, prefix="Step"
            )
            prompt = JUDGE_PROMPT_BELIEF.format(
                description=description,
                moment=gs["observable_event"],
                gold=gs["audience_belief"],
                response=step_response,
            )
            full_prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + prompt
            step_score = call_judge(full_prompt, judge_model, judge_provider, judge_api_key)
            print(f"    Step {gs['step']}: {'✓' if step_score == 1 else '✗'}")
            if step_score != 1:
                all_pass = False

        return 1.0 if all_pass else 0.0

    elif tt == TaskType.COUNTERFACTUAL.value:
        gold_cfs = task.get("gold_counterfactuals", [])
        n = len(gold_cfs)
        if n == 0:
            return 0.0

        all_pass = True
        for i, cf in enumerate(gold_cfs):
            cf_response = _extract_step_response(
                response, i + 1, n, prefix="Scenario"
            )
            prompt = JUDGE_PROMPT_COUNTERFACTUAL.format(
                description=description,
                condition=cf["condition"],
                question=cf["question"],
                gold_answer=cf["correct_answer"],
                gold_explanation=cf["explanation"],
                response=cf_response,
            )
            full_prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + prompt
            cf_score = call_judge(full_prompt, judge_model, judge_provider, judge_api_key)
            print(f"    Scenario {i+1}: {'✓' if cf_score == 1 else '✗'}")
            if cf_score != 1:
                all_pass = False

        return 1.0 if all_pass else 0.0

    else:
        return 0.0


# ── Deterministic scorers for MCQ tasks (Tasks 1 & 2) ───────

def score_effect_recognition(task: dict, response: str) -> float:
    """Binary: correct letter match."""
    ans = re.search(r'[A-Z]', response.strip().upper())
    return 1.0 if ans and ans.group() == task["gold"] else 0.0


def score_violation_id(task: dict, response: str) -> float:
    """Set-based F1 over selected violation types."""
    letters = set(re.findall(r'[A-G]', response.upper()))
    gold_letters = set(task["gold"].replace(" ", "").split(","))
    if not letters and not gold_letters:
        return 1.0
    if not letters or not gold_letters:
        return 0.0
    tp = len(letters & gold_letters)
    precision = tp / len(letters) if letters else 0
    recall = tp / len(gold_letters) if gold_letters else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# ── Scorer dispatch ──────────────────────────────────────────

# Tasks 1 & 2: deterministic MCQ scoring (no judge needed)
DETERMINISTIC_SCORERS = {
    TaskType.EFFECT_RECOGNITION.value: score_effect_recognition,
    TaskType.VIOLATION_ID.value: score_violation_id,
}

# Tasks 3-6: LLM-as-judge scoring
JUDGE_TASK_TYPES = {
    TaskType.BEST_EXPLANATION.value,
    TaskType.BELIEF_TRACE.value,
    TaskType.CALIBRATION.value,
    TaskType.COUNTERFACTUAL.value,
}


# ════════════════════════════════════════════════════════════════
# §5  COGNITIVE PROFILE COMPUTATION
# ════════════════════════════════════════════════════════════════

# Map task types to the 5 benchmark profile dimensions
PROFILE_MAP = {
    "recognition":   [TaskType.EFFECT_RECOGNITION.value],
    "causal_inference": [TaskType.VIOLATION_ID.value, TaskType.BEST_EXPLANATION.value],
    "deception_modeling": [TaskType.BELIEF_TRACE.value],
    "metacognitive_calibration": [TaskType.CALIBRATION.value],
    "transfer_robustness": [TaskType.COUNTERFACTUAL.value],
}

TASK_WEIGHTS = {
    TaskType.EFFECT_RECOGNITION.value: 0.10,
    TaskType.VIOLATION_ID.value: 0.10,
    TaskType.BEST_EXPLANATION.value: 0.20,
    TaskType.BELIEF_TRACE.value: 0.25,
    TaskType.CALIBRATION.value: 0.10,
    TaskType.COUNTERFACTUAL.value: 0.25,
}


def compute_profile(results: List[dict]) -> dict:
    """Compute 5-dimensional cognitive profile from scored tasks."""
    profile = {}
    for dim, task_types in PROFILE_MAP.items():
        scores = [r["score"] for r in results if r["task_type"] in task_types]
        profile[dim] = {
            "mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "n": len(scores),
            "std": round(
                (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
                4
            ) if len(scores) > 1 else 0.0,
        }
    task_type_means = {}
    for task_type, weight in TASK_WEIGHTS.items():
        scores = [r["score"] for r in results if r["task_type"] == task_type]
        if scores:
            task_type_means[task_type] = {
                "mean": sum(scores) / len(scores),
                "weight": weight,
            }
    total_weight = sum(item["weight"] for item in task_type_means.values())
    weighted_overall = (
        sum(item["mean"] * item["weight"] for item in task_type_means.values()) / total_weight
        if total_weight else 0.0
    )
    profile["overall"] = {
        "mean": round(weighted_overall, 4),
        "n": len(results),
        "weights": {task_type: meta["weight"] for task_type, meta in task_type_means.items()},
    }
    return profile


def compute_faculty_profile(results: List[dict]) -> dict:
    """Map scores to DeepMind 10 faculties."""
    faculty_scores = defaultdict(list)
    for r in results:
        for f in r.get("faculties", []):
            faculty_scores[f].append(r["score"])
    return {
        f: {"mean": round(sum(s)/len(s), 4), "n": len(s)}
        for f, s in faculty_scores.items()
    }


def compute_difficulty_analysis(results: List[dict]) -> dict:
    """Analyze performance by difficulty axes."""
    axis_buckets = defaultdict(lambda: defaultdict(list))
    for r in results:
        diff = r.get("difficulty", {})
        for axis, level in diff.items():
            axis_buckets[axis][level].append(r["score"])
    out = {}
    for axis, levels in axis_buckets.items():
        out[axis] = {
            level: {"mean": round(sum(s)/len(s), 4), "n": len(s)}
            for level, s in sorted(levels.items())
        }
    return out


# ════════════════════════════════════════════════════════════════
# §6  LLM EVALUATION HARNESS
# ════════════════════════════════════════════════════════════════

def call_llm(prompt: str, model: str, provider: str = "anthropic",
             api_key: str = "", temperature: float = 0.0,
             max_tokens: int = 1024) -> str:
    """Call an LLM API and return the text response."""
    import urllib.error
    import urllib.request

    def _extract_openai_response_text(data: dict) -> str:
        """Extract plain text from a Responses API payload."""
        if isinstance(data.get("output_text"), str) and data["output_text"]:
            return data["output_text"]

        chunks = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    chunks.append(content.get("text", ""))

        if chunks:
            return "".join(chunks)

        raise ValueError(f"OpenAI response did not contain text output: {data}")

    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
    elif provider == "openai":
        # GPT-5.2 works best with the Responses API, and newer reasoning
        # models are more reliable there than on the legacy chat endpoint.
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(raw)
            if isinstance(err, dict):
                message = err.get("error", {}).get("message", raw)
            else:
                message = raw
        except json.JSONDecodeError:
            message = raw or str(e)
        raise RuntimeError(
            f"{provider} API error {e.code}: {message}"
        ) from e

    if provider == "anthropic":
        return data["content"][0]["text"]
    else:
        return _extract_openai_response_text(data)


def evaluate_model(model: str, provider: str = "anthropic",
                   api_key: str = "", tasks: List[dict] = None,
                   n_repeats: int = 1, delay: float = 1.0,
                   dry_run: bool = False,
                   judge_model: str = "claude-sonnet-4-20250514",
                   judge_provider: str = "anthropic",
                   judge_api_key: str = "") -> List[dict]:
    """Run all tasks through the model and score responses.

    Tasks 1-2 (MCQ): scored deterministically.
    Tasks 3-6 (open-ended): scored by LLM judge.
    """
    tasks = tasks or build_all_tasks()
    # Use same api key for judge if not specified
    if not judge_api_key:
        judge_api_key = api_key
    results = []
    total = len(tasks) * n_repeats

    for i, task in enumerate(tasks):
        for rep in range(n_repeats):
            idx = i * n_repeats + rep + 1
            tt = task["task_type"]
            print(f"[{idx}/{total}] {task['scenario_id']} / {tt} (rep {rep+1})")

            if dry_run:
                print(f"  PROMPT:\n{task['prompt'][:200]}...\n")
                response = ""
                score = 0.0
            else:
                # Step 1: Get model response
                try:
                    response = call_llm(
                        task["prompt"], model, provider, api_key
                    )
                    time.sleep(delay)
                except Exception as e:
                    print(f"  ERROR (all retries failed): {e}")
                    response = f"ERROR: {e}"
                    score = 0.0
                    results.append({
                        "scenario_id": task["scenario_id"],
                        "task_type": tt,
                        "faculties": task.get("faculties", []),
                        "difficulty": task.get("difficulty", {}),
                        "response": response[:1000],
                        "score": 0.0,
                        "scoring_method": "error",
                        "n_sub_items": task.get("n_steps", task.get("n_counterfactuals", 1)),
                        "repeat": rep,
                    })
                    continue

                # Step 2: Score
                if tt in DETERMINISTIC_SCORERS:
                    score = DETERMINISTIC_SCORERS[tt](task, response)
                    print(f"  Score (deterministic): {score}")
                elif tt in JUDGE_TASK_TYPES:
                    desc = task.get("scenario_description", "")
                    try:
                        score = score_with_judge(
                            task, response, desc,
                            judge_model, judge_provider, judge_api_key
                        )
                    except Exception as e:
                        print(f"  JUDGE ERROR (all retries failed): {e}")
                        score = 0.0
                    time.sleep(delay)  # rate limit for judge calls
                    n_sub = task.get("n_steps", task.get("n_counterfactuals", 1))
                    print(f"  Score (judge, {n_sub} sub-items): {score}")
                else:
                    score = 0.0

                _result = {
                    "scenario_id": task["scenario_id"],
                    "task_type": tt,
                    "faculties": task.get("faculties", []),
                    "difficulty": task.get("difficulty", {}),
                    "question": task["prompt"],
                    "response": response,
                    "score": round(score, 4),
                    "scoring_method": "deterministic" if tt in DETERMINISTIC_SCORERS else "llm_judge",
                    "n_sub_items": task.get("n_steps", task.get("n_counterfactuals", 1)),
                    "repeat": rep,
                }
                if tt == "effect_recognition":
                    _result["gold"] = task["gold"]
                elif tt == "violation_identification":
                    _result["gold"] = task["gold"]
                elif tt == "belief_trace":
                    _result["gold_steps"] = task["gold_steps"]
                elif tt == "counterfactual":
                    _result["gold_counterfactuals"] = task["gold_counterfactuals"]
                elif tt == "calibration":
                    _result["gold"] = task["gold"]
                elif tt == "best_explanation":
                    _result["gold"] = task["gold"]

                results.append(_result)

    return results


def run_human_baseline(tasks: List[dict] = None) -> List[dict]:
    """Interactive human baseline collection."""
    tasks = tasks or build_all_tasks()
    results = []
    print("\n" + "="*60)
    print("MagicBench — Human Baseline Mode")
    print("="*60)

    for i, task in enumerate(tasks):
        tt = task["task_type"]
        print(f"\n--- Task {i+1}/{len(tasks)} [{tt}] ---")
        print(f"Scenario: {task['scenario_id']}\n")
        print(task["prompt"])
        print()
        response = input("Your answer: ").strip()

        # MCQ tasks: deterministic score
        if tt in DETERMINISTIC_SCORERS:
            score = DETERMINISTIC_SCORERS[tt](task, response)
        elif tt == TaskType.BELIEF_TRACE.value:
            # Show each step's gold and ask human to judge all
            gold_steps = task.get("gold_steps", [])
            all_correct = True
            for gs in gold_steps:
                print(f"\n  Step {gs['step']} ({gs['observable_event']}):")
                print(f"    Gold belief: {gs['audience_belief']}")
                s = input(f"    Did you get step {gs['step']} right? (1=yes, 0=no): ").strip()
                if s != "1":
                    all_correct = False
            score = 1.0 if all_correct else 0.0
        elif tt == TaskType.COUNTERFACTUAL.value:
            # Show each counterfactual's gold and ask human to judge all
            gold_cfs = task.get("gold_counterfactuals", [])
            all_correct = True
            for j, cf in enumerate(gold_cfs):
                print(f"\n  Scenario {j+1}: {cf['condition']}")
                print(f"    Gold answer: {cf['correct_answer']}")
                print(f"    Gold explanation: {cf['explanation']}")
                s = input(f"    Did you get scenario {j+1} right? (1=yes, 0=no): ").strip()
                if s != "1":
                    all_correct = False
            score = 1.0 if all_correct else 0.0
        else:
            # Other open-ended: show gold and let human self-judge
            print(f"\n  Gold answer: {task.get('gold', task.get('gold_answer', 'N/A'))}")
            self_score = input("  Does your answer match? (1=yes, 0=no): ").strip()
            score = 1.0 if self_score == "1" else 0.0


        _result = {
            "scenario_id": task["scenario_id"],
            "task_type": tt,
            "faculties": task.get("faculties", []),
            "difficulty": task.get("difficulty", {}),
            "question": task["prompt"],
            "response": response,
            "score": round(score, 4),
            "scoring_method": "deterministic" if tt in DETERMINISTIC_SCORERS else "llm_judge",
            "n_sub_items": task.get("n_steps", task.get("n_counterfactuals", 1)),
            "repeat": 0,
        }
        if tt == "effect_recognition":
            _result["gold"] = task["gold"]
        elif tt == "violation_identification":
            _result["gold"] = task["gold"]
        elif tt == "belief_trace":
            _result["gold_steps"] = task["gold_steps"]
        elif tt == "counterfactual":
            _result["gold_counterfactuals"] = task["gold_counterfactuals"]
        elif tt == "calibration":
            _result["gold"] = task["gold"]
        elif tt == "best_explanation":
            _result["gold"] = task["gold"]

        results.append(_result)
        
        print(f"  → Score: {score:.2f}")


# ════════════════════════════════════════════════════════════════
# §7  REPORTING
# ════════════════════════════════════════════════════════════════

def generate_report(results: List[dict], model_name: str) -> str:
    """Generate a human-readable analysis report."""
    profile = compute_profile(results)
    faculty_prof = compute_faculty_profile(results)
    diff_analysis = compute_difficulty_analysis(results)

    lines = [
        f"{'='*60}",
        f"MagicBench v1.0 — Evaluation Report",
        f"{'='*60}",
        f"Model: {model_name}",
        f"Total tasks: {len(results)}",
        f"Overall score: {profile['overall']['mean']:.1%}",
        "",
        "── 5-Dimension Cognitive Profile ──",
    ]
    for dim, stats in profile.items():
        if dim == "overall":
            continue
        bar = "█" * int(stats["mean"] * 20) + "░" * (20 - int(stats["mean"] * 20))
        lines.append(f"  {dim:<30} {bar} {stats['mean']:.1%}  (n={stats['n']})")

    lines += [
        "",
        "── DeepMind Faculty Mapping ──",
    ]
    for fac, stats in sorted(faculty_prof.items(), key=lambda x: -x[1]["mean"]):
        bar = "█" * int(stats["mean"] * 20) + "░" * (20 - int(stats["mean"] * 20))
        lines.append(f"  {fac:<26} {bar} {stats['mean']:.1%}  (n={stats['n']})")

    lines += ["", "── Performance by Difficulty Axis ──"]
    for axis, levels in diff_analysis.items():
        lines.append(f"  {axis}:")
        for level, stats in levels.items():
            bar = "█" * int(stats["mean"] * 10)
            lines.append(f"    Level {level}: {bar} {stats['mean']:.1%} (n={stats['n']})")

    # Per-task-type breakdown
    lines += ["", "── Per Task Type ──"]
    by_type = defaultdict(list)
    for r in results:
        by_type[r["task_type"]].append(r["score"])
    for tt, scores in sorted(by_type.items()):
        mean = sum(scores) / len(scores)
        lines.append(f"  {tt:<30} {mean:.1%}  (n={len(scores)})")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# §8  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MagicBench v1.0 — Deception-Sensitive Cognitive Benchmark"
    )
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--human", action="store_true",
                        help="Run interactive human baseline")
    parser.add_argument("--judge-model", default=None,
                        help="Model used as LLM judge for open-ended tasks")
    parser.add_argument("--judge-provider", default=None,
                        choices=["anthropic", "openai"])
    parser.add_argument("--judge-api-key", default=None,
                        help="API key for judge (defaults to --api-key)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if args.judge_model is None and args.judge_provider is None:
        # Use the same model and provider as the main model
        args.judge_provider = args.provider
        args.judge_model = args.model
    if args.judge_api_key is None:
        args.judge_api_key = args.api_key

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = build_all_tasks(seed=args.seed)
    print(f"Generated {len(tasks)} tasks from {len(SCENARIOS)} scenarios")

    if args.human:
        results = run_human_baseline(tasks)
        model_name = "human"
    elif args.dry_run:
        results = evaluate_model(
            args.model, args.provider, args.api_key, tasks, dry_run=True
        )
        model_name = f"{args.model}_dryrun"
    else:
        jkey = args.judge_api_key or args.api_key
        results = evaluate_model(
                args.model, args.provider, args.api_key, tasks,
                n_repeats=args.n_repeats, delay=args.delay,
                judge_model=args.judge_model,
                judge_provider=args.judge_provider,
                judge_api_key=jkey,
            )
        model_name = args.model

    # Report
    report = generate_report(results, model_name)
    print("\n" + report)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.output_dir}/{model_name.replace('/', '_')}_{ts}"
    with open(f"{base}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{base}_profile.json", "w") as f:
        json.dump({
            "model": model_name,
            "timestamp": ts,
            "profile": compute_profile(results),
            "faculty_profile": compute_faculty_profile(results),
            "difficulty_analysis": compute_difficulty_analysis(results),
        }, f, indent=2)
    with open(f"{base}_report.txt", "w") as f:
        f.write(report)
    print(f"\nResults saved to {base}_*.json/txt")


if __name__ == "__main__":
    main()
