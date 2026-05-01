#!/usr/bin/env python3
"""
Evaluate post-measurement LLM reporting outputs.

This script scores model outputs without calling any model API. It expects JSONL
records containing a structured measurement input and the model's JSON response.
The aim is to compare LLMs as decision-support reporters, not as reconstruction
or measurement engines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "recommendation",
    "evidence",
    "uncertainties",
    "do_not_claim",
    "needs_human_review",
}


FORBIDDEN_UNSUPPORTED_TERMS = {
    "disease",
    "fungal",
    "pest",
    "yield loss",
    "nitrogen",
    "irrigation deficit",
    "fiber quality",
}


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def parse_response(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "response is not a JSON object"
    return parsed, None


def schema_valid(response: dict[str, Any]) -> bool:
    if set(response) != REQUIRED_KEYS:
        return False
    return (
        isinstance(response["recommendation"], str)
        and isinstance(response["evidence"], list)
        and isinstance(response["uncertainties"], list)
        and isinstance(response["do_not_claim"], list)
        and isinstance(response["needs_human_review"], bool)
    )


def hallucination_hits(response: dict[str, Any]) -> list[str]:
    claim_fields = {
        "recommendation": response.get("recommendation", ""),
        "evidence": response.get("evidence", []),
        "uncertainties": response.get("uncertainties", []),
    }
    text = as_text(claim_fields).lower()
    return sorted(term for term in FORBIDDEN_UNSUPPORTED_TERMS if term in text)


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    valid = 0
    hallucinated = 0
    parse_failures = 0
    details = []

    for idx, record in enumerate(records):
        model = record.get("model", "unknown")
        raw = record.get("response", "")
        if isinstance(raw, dict):
            response = raw
            parse_error = None
        else:
            response, parse_error = parse_response(str(raw))

        if response is None:
            parse_failures += 1
            details.append({"index": idx, "model": model, "schema_valid": False, "error": parse_error})
            continue

        is_valid = schema_valid(response)
        hits = hallucination_hits(response)
        valid += int(is_valid)
        hallucinated += int(bool(hits))
        details.append(
            {
                "index": idx,
                "model": model,
                "schema_valid": is_valid,
                "hallucination_terms": hits,
            }
        )

    return {
        "records": total,
        "schema_valid_rate": valid / total if total else 0.0,
        "parse_failure_rate": parse_failures / total if total else 0.0,
        "unsupported_claim_rate": hallucinated / total if total else 0.0,
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score LLM reporting outputs.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = score_records(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
