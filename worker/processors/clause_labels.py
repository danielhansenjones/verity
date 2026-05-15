"""Single source of truth for the v1 clause taxonomy.

CLAUSE_WEIGHTS maps each clause type to its weight in the scoring formula.
CLAUSE_LABELS is the candidate set passed to the zero-shot classifier (every
key except the "general" catch-all). Deriving CLAUSE_LABELS from the weights
dict prevents the two lists from drifting.
"""

CLAUSE_WEIGHTS: dict[str, float] = {
    "indemnification": 1.0,
    "liability limitation": 0.9,
    "intellectual property assignment": 0.8,
    "termination": 0.7,
    "dispute resolution": 0.6,
    "confidentiality": 0.5,
    "payment terms": 0.5,
    "warranty": 0.4,
    "force majeure": 0.3,
    "governing law": 0.2,
    "general": 0.1,
}

CLAUSE_LABELS: list[str] = [
    label for label in CLAUSE_WEIGHTS if label != "general"
]
