"""Auditable cross-environment strategy features and canonical lessons.

Free-form reflections from a small actor tend to retain API/entity names even
when prompted not to.  This module deliberately uses a closed vocabulary so a
cross-family arm cannot accidentally leak environment-local instructions.
"""
from __future__ import annotations

import re


TRANSFER_FEATURES = {
    "lookup": ("find", "get", "list", "lookup", "search", "inspect", "fetch", "locate"),
    "create": ("create", "add", "register", "insert", "onboard"),
    "update": ("update", "edit", "change", "correct", "adjust", "set", "rename"),
    "delete": ("delete", "remove", "purge"),
    "restore": ("restore", "recover", "undelete", "reopen"),
    "assign": ("assign", "reassign", "allocate", "reserve"),
    "transfer": ("transfer", "move", "migrate", "relocate", "cross-post"),
    "cancel": ("cancel", "release", "revoke"),
    "verify": ("verify", "check", "confirm", "ensure", "validate"),
    "disambiguate": ("disambiguate", "duplicate", "canonical", "ambiguous", "matching"),
    "exact_match": ("exact", "exactly", "specific", "correct"),
    "preserve": ("preserve", "retain", "unchanged", "existing", "keep"),
    "order": ("before", "after", "first", "then", "prior", "prerequisite", "sequence"),
    "unique": ("unique", "conflict", "collision"),
    "authorization": ("authoriz", "permission", "role", "access", "forbidden", "denied"),
    "availability": ("available", "availability", "occupied", "vacant", "active", "pending"),
    "time": ("timestamp", "date", "time", "timezone", "epoch", "schedule"),
    "capacity": ("capacity", "limit", "quota", "slot", "bed", "venue"),
    "bulk": ("each", "every", "all", "bulk", "only"),
    "fallback": ("fallback", "otherwise", "if", "unless"),
    "error_recovery": ("error", "fail", "retry", "instead", "avoid"),
}


CANONICAL_RULES = {
    "lookup": "Inspect current state before relying on or mutating a record.",
    "create": "Validate required references and uniqueness constraints before creation.",
    "update": "Read current state before updating, then verify the intended field changed.",
    "delete": "Identify the exact target and dependent relationships before deletion.",
    "restore": "Confirm that a record exists in a restorable state before restoration.",
    "assign": "Check resource availability and relationship constraints before assignment.",
    "transfer": "Validate source and destination state before moving a resource or relationship.",
    "cancel": "Confirm an active record or relationship before cancellation or release.",
    "verify": "Verify current state before performing a dependent mutation.",
    "disambiguate": "Resolve duplicate or ambiguous records using stable attributes before mutation.",
    "exact_match": "Match the complete target criteria instead of relying on a partial identifier.",
    "preserve": "Preserve every unspecified state field during a targeted update.",
    "order": "Execute a dependent operation only after its prerequisite succeeds.",
    "unique": "Check uniqueness constraints before creating or renaming a record.",
    "authorization": "Verify the acting role or permission before a protected mutation.",
    "availability": "Re-check current availability immediately before allocating a resource.",
    "time": "Normalize and validate time representations before time-dependent operations.",
    "capacity": "Check capacity and occupancy before assigning a constrained resource.",
    "bulk": "Filter the target set precisely before applying a bulk operation.",
    "fallback": "Use a fallback path only after the primary precondition is shown to fail.",
    "error_recovery": "Inspect the returned error and current state before retrying a mutation.",
}

_PRIORITY = (
    "disambiguate", "exact_match", "preserve", "order", "unique", "authorization",
    "availability", "capacity", "time", "bulk", "fallback", "error_recovery", "verify",
    "lookup", "assign", "transfer", "cancel", "restore", "delete", "create", "update",
)


def transfer_features(text: str) -> set[str]:
    """Extract canonical workflow features intended to survive domain changes."""
    words = re.findall(r"[a-z]+", text.lower())
    return {
        feature
        for feature, stems in TRANSFER_FEATURES.items()
        if any(any(word.startswith(stem) for stem in stems) for word in words)
    }


def canonical_transfer_reflection(task_text: str, evidence_text: str, max_rules: int = 4) -> str:
    """Compile domain-free rules, preferring features present in task and evidence."""
    task = transfer_features(task_text)
    evidence = transfer_features(evidence_text)
    supported = task & evidence
    ordered = [feature for feature in _PRIORITY if feature in supported]
    ordered.extend(
        feature for feature in _PRIORITY if feature in evidence and feature not in supported
    )
    if not ordered:
        ordered = ["verify"]
    return "\n".join(
        f"- [{feature}] {CANONICAL_RULES[feature]}" for feature in ordered[:max_rules]
    )
