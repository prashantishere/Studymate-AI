"""Deterministic mastery tracking, grading, and weak-topic selection for StudyMate AI.

Implements pure deterministic grading without LLM reliance, calculates concept-level
accuracy, tie-breaks systematically, and updates learning progress upon retesting.
"""

from typing import Any
from studymate.schemas import GradingError


def grade_answers(
    questions: list[dict[str, Any]],
    answers: dict[str, Any]
) -> dict[str, Any]:
    """Grade submitted answers against the stored answer key deterministically.

    Args:
        questions: Full question dictionaries containing 'id', 'concept', 'correct_index', etc.
        answers: Map of question_id -> selected_option_index (0-3).
                 Unanswered questions may be omitted or mapped to None/-1.

    Returns:
        Grading summary dictionary with counts, accuracy, concept mastery, and per-question results.

    Raises:
        GradingError: If question list is invalid or answer indexes are out of bounds.
    """
    if not isinstance(questions, list) or not questions:
        raise GradingError("Cannot grade an empty or invalid questions list.")

    if not isinstance(answers, dict):
        raise GradingError(f"Expected answers dictionary, got {type(answers).__name__}")

    q_map: dict[str, dict[str, Any]] = {}
    for q in questions:
        qid = q.get("id")
        if not qid:
            raise GradingError("Found question without a valid 'id'.")
        q_map[qid] = q

    correct_count = 0
    attempted_count = 0
    concept_stats: dict[str, dict[str, Any]] = {}
    question_results: list[dict[str, Any]] = []

    # Initialize all concepts seen in the quiz as untested (0/0)
    for q in questions:
        concept = q.get("concept", "General").strip()
        if concept not in concept_stats:
            concept_stats[concept] = {
                "correct": 0,
                "attempted": 0,
                "total": 0,
                "accuracy": None,  # None indicates unattempted/untested
            }
        concept_stats[concept]["total"] += 1

    for q in questions:
        qid = q["id"]
        concept = q.get("concept", "General").strip()
        correct_idx = q.get("correct_index")

        if correct_idx is None or not (0 <= correct_idx <= 3):
            raise GradingError(f"Question '{qid}' has an invalid correct_index: {correct_idx}")

        selected_idx = answers.get(qid)

        # Check if question was attempted
        is_attempted = selected_idx is not None and selected_idx != -1
        is_correct = False

        if is_attempted:
            if not isinstance(selected_idx, int) or not (0 <= selected_idx <= 3):
                raise GradingError(
                    f"Invalid answer index {selected_idx} for question '{qid}'. Must be 0, 1, 2, or 3."
                )
            attempted_count += 1
            concept_stats[concept]["attempted"] += 1

            if selected_idx == correct_idx:
                is_correct = True
                correct_count += 1
                concept_stats[concept]["correct"] += 1

        question_results.append({
            "id": qid,
            "concept": concept,
            "question": q.get("question", ""),
            "options": q.get("options", []),
            "selected_index": selected_idx if is_attempted else None,
            "correct_index": correct_idx,
            "is_correct": is_correct,
            "is_attempted": is_attempted,
            "explanation": q.get("explanation", ""),
            "source_pages": q.get("source_pages", []),
        })

    # Compute percentage accuracies
    mastery: dict[str, dict[str, Any]] = {}
    missed_concepts: list[str] = []

    for concept, stats in concept_stats.items():
        att = stats["attempted"]
        cor = stats["correct"]
        tot = stats["total"]

        if att > 0:
            acc = round((cor / att) * 100.0, 1)
            mastery[concept] = {
                "correct": cor,
                "attempted": att,
                "total": tot,
                "accuracy": acc,
            }
            if acc < 100.0:
                missed_concepts.append(concept)
        else:
            mastery[concept] = {
                "correct": 0,
                "attempted": 0,
                "total": tot,
                "accuracy": None,  # Untested
            }

    overall_accuracy = (
        round((correct_count / attempted_count) * 100.0, 1)
        if attempted_count > 0
        else 0.0
    )

    weakest = choose_weakest_concept(mastery)

    return {
        "correct_answers": correct_count,
        "attempted_answers": attempted_count,
        "total_questions": len(questions),
        "overall_accuracy": overall_accuracy,
        "mastery": mastery,
        "missed_concepts": sorted(missed_concepts),
        "weakest_concept": weakest,
        "question_results": question_results,
    }


def choose_weakest_concept(mastery: dict[str, dict[str, Any]]) -> str | None:
    """Select the weakest attempted concept using strict deterministic tie-breaking.

    Tie-breaking hierarchy:
      1. Lowest accuracy percentage (e.g. 0.0% < 50.0% < 100.0%).
      2. Highest number of attempted questions (more evidence of difficulty).
      3. Lexicographical order of concept name.

    Concepts with 100% accuracy or 0 attempted questions are not considered weak.

    Args:
        mastery: Dictionary of concept -> {correct, attempted, accuracy, ...}.

    Returns:
        The concept name to focus on, or None if no weak concept exists.
    """
    if not mastery or not isinstance(mastery, dict):
        return None

    candidates: list[tuple[float, int, str]] = []

    for concept, stats in mastery.items():
        if not isinstance(stats, dict):
            continue
        accuracy = stats.get("accuracy")
        attempted = stats.get("attempted", 0)

        # Only attempted concepts with accuracy < 100% are eligible weaknesses
        if accuracy is not None and attempted > 0 and accuracy < 100.0:
            candidates.append((accuracy, attempted, concept))

    if not candidates:
        return None

    # Sort key:
    # 1. accuracy ascending (lowest first)
    # 2. attempted descending (-attempted for higher attempts first)
    # 3. concept name ascending
    candidates.sort(key=lambda x: (x[0], -x[1], x[2]))

    return candidates[0][2]


def grade_retest_submission(
    question: dict[str, Any],
    answer_index: int,
    previous_mastery: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Grade a follow-up retest question and compute updated concept mastery.

    Preserves the initial diagnostic scores so before/after progress can be displayed.

    Args:
        question: Retest question dictionary with 'id', 'concept', 'correct_index', etc.
        answer_index: Selected option index (0 to 3).
        previous_mastery: The concept mastery dictionary from the initial diagnostic quiz.

    Returns:
        Dictionary with retest results, updated mastery, and next steps.

    Raises:
        GradingError: If answer_index is invalid or question lacks required fields.
    """
    if not isinstance(question, dict):
        raise GradingError("Invalid retest question structure.")

    correct_idx = question.get("correct_index")
    if correct_idx is None or not (0 <= correct_idx <= 3):
        raise GradingError(f"Retest question has invalid correct_index: {correct_idx}")

    if not isinstance(answer_index, int) or not (0 <= answer_index <= 3):
        raise GradingError(f"Selected answer index must be between 0 and 3, got: {answer_index}")

    concept = question.get("concept", "General").strip()
    is_correct = (answer_index == correct_idx)

    # Deep copy previous mastery to avoid mutating callers' state
    import copy
    updated_mastery = copy.deepcopy(previous_mastery) if previous_mastery else {}

    # Update or insert concept stats
    if concept not in updated_mastery:
        updated_mastery[concept] = {
            "correct": 1 if is_correct else 0,
            "attempted": 1,
            "total": 1,
            "accuracy": 100.0 if is_correct else 0.0,
        }
    else:
        stats = updated_mastery[concept]
        prev_cor = stats.get("correct", 0)
        prev_att = stats.get("attempted", 0)
        prev_tot = stats.get("total", 0)

        new_cor = prev_cor + (1 if is_correct else 0)
        new_att = prev_att + 1
        new_tot = prev_tot + 1
        new_acc = round((new_cor / new_att) * 100.0, 1)

        updated_mastery[concept] = {
            "correct": new_cor,
            "attempted": new_att,
            "total": new_tot,
            "accuracy": new_acc,
        }

    next_topic = choose_weakest_concept(updated_mastery)

    return {
        "question_id": question.get("id", "retest_q1"),
        "concept": concept,
        "selected_index": answer_index,
        "correct_index": correct_idx,
        "is_correct": is_correct,
        "explanation": question.get("explanation", ""),
        "source_pages": question.get("source_pages", []),
        "previous_mastery": previous_mastery,
        "updated_mastery": updated_mastery,
        "suggested_next_topic": next_topic,
    }
