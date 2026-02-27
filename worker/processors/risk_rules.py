"""Rule-based risk detection via spaCy Matcher.

Rules are declared as YAML in worker/processors/rules/*.yaml. Each rule compiles
into one or more spaCy Matcher patterns. Hits are returned with character offsets
so the assembler can quote the exact matched text rather than the whole chunk.

We use spacy.blank("en") (tokenizer only, no trained model) so the dependency
footprint stays small. Patterns use LOWER-based matching, which is token-aware
but does not handle lemma variants. That is an explicit trade-off; the upgrade
path to a trained model + DependencyMatcher is tracked in docs/DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy
import yaml
from spacy.matcher import Matcher

_RULES_DIR = Path(__file__).parent / "rules"


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    reason: str


@dataclass(frozen=True)
class RiskHit:
    id: str
    severity: str
    reason: str
    matched_text: str
    start: int
    end: int


def _load_rule_files(rules_dir: Path) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for path in sorted(rules_dir.glob("*.yaml")):
        with path.open() as fh:
            rules.extend(yaml.safe_load(fh) or [])
    return rules


class RiskMatcher:
    """Applies the compiled rule set to a piece of text and returns RiskHits."""

    def __init__(self, rules_dir: Path = _RULES_DIR):
        self._nlp = spacy.blank("en")
        self._matcher: Matcher = Matcher(self._nlp.vocab)
        self._rules: dict[str, Rule] = {}

        for rule_dict in _load_rule_files(rules_dir):
            rule = Rule(
                id=rule_dict["id"],
                severity=rule_dict["severity"],
                reason=rule_dict["reason"],
            )
            self._rules[rule.id] = rule
            self._matcher.add(rule.id, rule_dict["patterns"])

    @property
    def rule_ids(self) -> list[str]:
        return list(self._rules.keys())

    def match(self, text: str) -> list[RiskHit]:
        doc = self._nlp(text)
        hits: list[RiskHit] = []
        seen: set[tuple[str, int, int]] = set()

        for match_id, start, end in self._matcher(doc):
            rule_id = self._nlp.vocab.strings[match_id]
            span = doc[start:end]
            char_start = span.start_char
            char_end = span.end_char
            key = (rule_id, char_start, char_end)
            if key in seen:
                continue
            seen.add(key)

            rule = self._rules[rule_id]
            hits.append(
                RiskHit(
                    id=rule.id,
                    severity=rule.severity,
                    reason=rule.reason,
                    matched_text=span.text,
                    start=char_start,
                    end=char_end,
                )
            )
        return hits


# Module-level singleton: compile rules once per process.
_default_matcher: RiskMatcher | None = None


def default_matcher() -> RiskMatcher:
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = RiskMatcher()
    return _default_matcher
