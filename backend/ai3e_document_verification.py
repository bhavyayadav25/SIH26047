"""AI-3E: safe human verification layer for AI-3D document extraction.

AI-3E does not diagnose or silently alter extracted values. It prepares a
review queue and accepts an explicit, item-level verification payload so a
human reviewer can confirm or correct the structured extraction.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List


def build_verification_summary(extraction: Dict[str, Any]) -> Dict[str, Any]:
    items = extraction.get("items") or []
    return {
        "schema_version": "AI-3E.1",
        "total_items": len(items),
        "items_needing_review": sum(1 for x in items if x.get("needs_review")),
        "high_confidence_items": sum(1 for x in items if float(x.get("confidence", 0)) >= 0.80),
        "review_required": bool(extraction.get("needs_review", True)),
        "instruction": "Compare each extracted value with the original document before confirming it for clinical use.",
        "items": [
            {
                "index": i,
                "category": x.get("category"),
                "label": x.get("label"),
                "value": x.get("value"),
                "confidence": x.get("confidence", 0),
                "needs_review": bool(x.get("needs_review", True)),
                "evidence": x.get("evidence", ""),
            }
            for i, x in enumerate(items)
        ],
    }


def apply_document_verification(extraction: Dict[str, Any], verified_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a verified copy. Each payload item must identify an extraction index.

    Payload shape: {"index": 0, "verified": true, "value": "..."}
    A corrected value is allowed only when verified=true and value is supplied.
    """
    source = extraction.get("items") or []
    if not verified_items:
        raise ValueError("Provide at least one item to verify.")
    result = deepcopy(extraction)
    result["schema_version"] = "AI-3E.1"
    result["items"] = deepcopy(source)
    seen = set()
    for entry in verified_items:
        if not isinstance(entry, dict):
            raise ValueError("Each verification item must be an object.")
        idx = entry.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(result["items"]):
            raise ValueError(f"Invalid extraction index: {idx}")
        if idx in seen:
            raise ValueError(f"Duplicate extraction index: {idx}")
        seen.add(idx)
        if entry.get("verified") is not True:
            raise ValueError(f"Item {idx} must be explicitly verified=true.")
        item = result["items"][idx]
        if "value" in entry:
            value = entry["value"]
            if value is None or not str(value).strip():
                raise ValueError(f"Corrected value for item {idx} cannot be empty.")
            item["value"] = str(value).strip()
            item["corrected"] = True
        item["verified"] = True
        item["verified_confidence"] = 1.0
        item["needs_review"] = False
        item["verification_source"] = "Human reviewer"
    result["verified_item_count"] = sum(1 for x in result["items"] if x.get("verified"))
    result["unverified_item_count"] = len(result["items"]) - result["verified_item_count"]
    result["needs_review"] = result["unverified_item_count"] > 0
    result["verification_status"] = "Verified" if not result["needs_review"] else "Partially Verified"
    return result
