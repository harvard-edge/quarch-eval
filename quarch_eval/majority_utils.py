from collections import Counter
from typing import Iterable, cast
from . import datatypes as dt


def normalize_label(s: str) -> dt.JudgeLabel:
    if s in ("CORRECT", "INCORRECT", "PARTIALLY-CORRECT"):
        return cast(dt.JudgeLabel, s)
    raise ValueError(f"Unknown label: {s!r}")


def filter_unsure(judgments: Iterable[dt.Judgment]) -> list[dt.Judgment]:
    return [j for j in judgments if j.judgment != "UNSURE"]


def rebucket(
    judgments: Iterable[dt.Judgment], rebucket_partial_to_incorrect: bool
) -> Iterable[dt.Judgment]:
    if not rebucket_partial_to_incorrect:
        return judgments
    new_judgments = []
    for j in judgments:
        if j.judgment == "PARTIALLY-CORRECT":
            new_j = j.model_copy()
            new_j.judgment = "INCORRECT"
            new_judgments.append(new_j)
        else:
            new_judgments.append(j)
    return new_judgments


def majority_reached_for_planned_k(
    judgments: Iterable[dt.Judgment],
    k_planned: int,
    judge_llm_name: str | None,
    rebucket_partial_to_incorrect: bool,
) -> dt.MajorityResult:
    """
    Check if any label has already clinched a majority
    given the planned total k (early stopping).
    """
    judgments = filter_unsure(judgments)
    if judge_llm_name is not None:
        judgments = [j for j in judgments if j.judge_llm_name == judge_llm_name]
    judgments = rebucket(judgments, rebucket_partial_to_incorrect)
    counts: Counter[dt.JudgeLabel] = Counter(
        normalize_label(j.judgment) for j in judgments
    )
    threshold = k_planned // 2 + 1

    if len(counts) == 0:
        return dt.MajorityResult(False, None, counts)

    label, c = counts.most_common(1)[0]
    return dt.MajorityResult(c >= threshold, label if c >= threshold else None, counts)


def majority_in_hand(
    judgments: Iterable[dt.Judgment],
    judge_llm_name: str | None,
    rebucket_partial_to_incorrect: bool,
) -> dt.MajorityResult:
    """
    Check if there's a strict majority among the collected judgments only.
    """
    judgments = filter_unsure(judgments)
    if judge_llm_name is not None:
        judgments = [j for j in judgments if j.judge_llm_name == judge_llm_name]
    judgments = rebucket(judgments, rebucket_partial_to_incorrect)
    counts: Counter[dt.JudgeLabel] = Counter(
        normalize_label(j.judgment) for j in judgments
    )
    n = sum(counts.values())
    if n == 0:
        return dt.MajorityResult(False, None, counts)

    label, c = counts.most_common(1)[0]
    return dt.MajorityResult(c > n // 2, label if c > n // 2 else None, counts)


def unanimous_in_hand(
    judgments: Iterable[dt.Judgment],
    judge_llm_name: str | None,
    rebucket_partial_to_incorrect: bool,
) -> dt.MajorityResult:
    """
    Check if there's a unanimous agreement among the collected judgments only.
    """
    judgments = filter_unsure(judgments)
    if judge_llm_name is not None:
        judgments = [j for j in judgments if j.judge_llm_name == judge_llm_name]
    judgments = rebucket(judgments, rebucket_partial_to_incorrect)
    counts: Counter[dt.JudgeLabel] = Counter(
        normalize_label(j.judgment) for j in judgments
    )
    n = sum(counts.values())
    if n == 0:
        return dt.MajorityResult(False, None, counts)

    label, c = counts.most_common(1)[0]
    return dt.MajorityResult(c == n, label if c == n else None, counts)

def majority_vote_correct(
    resp: dt.Response,
    j: int | None,
    judge_llm_name: str | None
) -> bool | None:
    """
    Simpler check if an attempt is voted as correct based on LLM judgements.
    Returns None if j not reached or no majority found.
    """
    judgments = resp.judgments
    if j is None:
        maj = majority_in_hand(
            judgments,
            judge_llm_name=judge_llm_name,
            rebucket_partial_to_incorrect=True,
        )
    else:
        maj = majority_reached_for_planned_k(
            judgments,
            k_planned=j,
            judge_llm_name=judge_llm_name,
            rebucket_partial_to_incorrect=True,
        )

    if maj.label == 'CORRECT':
        return True
    if maj.label == 'INCORRECT':
        return False
    return None