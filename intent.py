# intent.py
import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class IntentStatus(str, Enum):
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass
class IntentResult:
    status: IntentStatus
    score: int
    reasons: List[str]


# =========================
# CONFIG – DETERMINISTIC
# =========================

DOMAIN_TERMS = {
    # approval / decisions
    "approve", "approval", "authorized", "authority",
    # processes
    "policy", "procedure", "process", "rule", "requirement",
    # change / incident
    "change", "emergency", "incident", "sev",
    # roles
    "ops", "operations", "lead", "owner",
    # conditions
    "allowed", "required", "exception", "review",
}

BLACKLIST_TERMS = {
    "joke", "meme", "fun","sport",
    "pizza", "weather", "movie", "song", "stock", "crypto",
}

PATTERNS = [
    re.compile(r"\bwho\s+(approves|is responsible)\b"),
    re.compile(r"\bwhat happens if\b"),
    re.compile(r"\bis\s+.*\s+allowed\b"),
    re.compile(r"\bdoes\s+.*\s+require\b"),
    re.compile(r"\bhow to\s+.*\s+process\b"),
]

QUESTION_STARTERS = ("who", "what", "how", "is", "does", "when", "can")


# =========================
# CORE LOGIC
# =========================

def check_intent(query: str) -> IntentResult:
    q = query.lower().strip()
    score = 0
    reasons: List[str] = []

    tokens = re.findall(r"\b\w+\b", q)

    # --- BLACKLIST (hard negative signal)
    for term in BLACKLIST_TERMS:
        if term in q:
            score -= 3
            reasons.append(f"blacklist:{term}")

    # --- DOMAIN TERMS (strong positive signal)
    domain_hits = [t for t in DOMAIN_TERMS if t in q]
    if domain_hits:
        score += 2
        reasons.append(f"domain_term:{domain_hits[0]}")

    # --- PATTERNS (structure-based signal)
    for pattern in PATTERNS:
        if pattern.search(q):
            score += 1
            reasons.append(f"pattern:{pattern.pattern}")
            break

    # --- QUESTION FORM
    if q.endswith("?") or q.startswith(QUESTION_STARTERS):
        score += 1
        reasons.append("question_form")

    # --- TOO SHORT HEURISTIC
    if len(tokens) < 3 and not domain_hits:
        score -= 1
        reasons.append("too_short")

    status = (
        IntentStatus.IN_SCOPE
        if score >= 2
        else IntentStatus.OUT_OF_SCOPE
    )

    return IntentResult(
        status=status,
        score=score,
        reasons=reasons,
    )
