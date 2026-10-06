"""Evaluation: per-field accuracy, micro precision/recall/F1 for list fields,
note-level exact match, and agreement of the downstream policy decision."""
from __future__ import annotations

from .policy import adjudicate
from .schema import LIST_FIELDS, SCALAR_FIELDS, Extraction


def _items(ex: Extraction, name: str) -> set:
    if name == "medications":
        return {m.key() for m in ex.medications}
    return {a.strip().lower() for a in ex.allergies}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def field_errors(pred: Extraction, gold: Extraction) -> list[dict]:
    errors = []
    for name in SCALAR_FIELDS:
        if getattr(pred, name) != getattr(gold, name):
            errors.append({"field": name, "gold": getattr(gold, name), "pred": getattr(pred, name)})
    for name in LIST_FIELDS:
        g, p = _items(gold, name), _items(pred, name)
        if g != p:
            errors.append({"field": name, "gold": sorted(map(str, g)), "pred": sorted(map(str, p))})
    return errors


def score(preds: list[Extraction], golds: list[Extraction]) -> dict:
    if len(preds) != len(golds):
        raise ValueError("preds and golds must have the same length")
    n = len(golds)
    if n == 0:
        raise ValueError("nothing to score")

    result: dict = {"n_notes": n, "fields": {}}
    for name in SCALAR_FIELDS:
        correct = sum(getattr(p, name) == getattr(g, name) for p, g in zip(preds, golds))
        result["fields"][name] = {"accuracy": round(correct / n, 4)}
    for name in LIST_FIELDS:
        tp = fp = fn = 0
        for p, g in zip(preds, golds):
            ps, gs = _items(p, name), _items(g, name)
            tp += len(ps & gs)
            fp += len(ps - gs)
            fn += len(gs - ps)
        result["fields"][name] = _prf(tp, fp, fn)

    exact = sum(not field_errors(p, g) for p, g in zip(preds, golds))
    agree = sum(adjudicate(p).decision == adjudicate(g).decision for p, g in zip(preds, golds))
    result["note_exact_match"] = round(exact / n, 4)
    result["decision_agreement"] = round(agree / n, 4)
    return result
