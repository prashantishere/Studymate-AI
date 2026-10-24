"""StudyMate AI - Main Backend Interface.

This module provides the primary public API consumed by the Streamlit frontend (`app.py`).
It coordinates document text validation, Gemma model inference, deterministic grading,
evidence-grounded retrieval, and adaptive mastery state updates.
"""

from typing import Any
import json

from studymate.ai_client import GemmaAIClient
from studymate.mastery import (
    choose_weakest_concept,
    grade_answers,
    grade_retest_submission,
)
from studymate.retrieval import (
    format_passages_for_prompt,
    retrieve_passages,
)
from studymate.schemas import (
    DocumentContentError,
    InvalidModelResponseError,
    validate_question_dict,
    validate_quiz_list,
    validate_teaching_response,
)


# Global default client instance (instantiated on first use or overridden)
_DEFAULT_CLIENT: GemmaAIClient | None = None


def get_ai_client() -> GemmaAIClient:
    """Retrieve or initialize the singleton Gemma AI client."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = GemmaAIClient()
    return _DEFAULT_CLIENT


def set_ai_client(client: GemmaAIClient) -> None:
    """Inject a custom or mock AI client (useful for unit testing)."""
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = client


def _prepare_document_context(pages: list[dict[str, Any]], max_chars: int = 12000) -> tuple[str, set[int]]:
    """Validate and compile pages into a readable reference context string.

    Returns:
        tuple of (context_string, set_of_valid_page_numbers)
    """
    if not pages or not isinstance(pages, list):
        raise DocumentContentError("Document pages list is empty or invalid.")

    valid_pages: set[int] = set()
    context_parts: list[str] = []
    current_chars = 0
    all_words: list[str] = []

    for p in pages:
        page_num = p.get("page")
        text = p.get("text", "")
        if isinstance(page_num, int) and isinstance(text, str) and text.strip():
            valid_pages.add(page_num)
            clean_text = text.strip()
            words = [w.lower() for w in clean_text.split() if len(w) > 1]
            all_words.extend(words)
            block = f"--- [PAGE {page_num}] ---\n{clean_text}\n"
            if current_chars + len(block) <= max_chars:
                context_parts.append(block)
                current_chars += len(block)
            else:
                remaining = max_chars - current_chars
                if remaining > 100:
                    context_parts.append(block[:remaining] + "\n...[truncated]")
                break

    if not valid_pages:
        raise DocumentContentError("The provided document does not contain any usable text.")

    # Guard against scanned PDFs containing only repetitive watermarks (e.g. 'arnav notes' 16 times)
    unique_words = set(all_words)
    if len(all_words) < 25 or len(unique_words) < 15:
        sample_snippet = " ".join(all_words[:8])
        raise DocumentContentError(
            f"The uploaded PDF does not contain sufficient readable text to generate questions (found only {len(unique_words)} unique words, e.g. '{sample_snippet}'). "
            "This happens when a PDF consists of scanned images or photos of notes where only a watermark/header is detectable as digital text. "
            "Please upload a document with digital, selectable course text."
        )

    return "\n".join(context_parts), valid_pages


# =====================================================================
# Function A: Generate diagnostic quiz
# =====================================================================

def generate_quiz(
    pages: list[dict[str, Any]],
    language: str = "English"
) -> list[dict[str, Any]]:
    """Generate five multiple-choice questions from the uploaded material.

    Args:
        pages: List of extracted page dictionaries [{"page": 1, "text": "..."}].
        language: Language for the generated questions (default: English).

    Returns:
        List of 5 validated question dictionaries matching the QuestionSchema.

    Raises:
        DocumentContentError: If pages contain no usable text.
        InvalidModelResponseError: If the model cannot produce 5 valid questions.
        ModelRequestError: If network/provider call fails.
    """
    doc_context, valid_pages = _prepare_document_context(pages)
    client = get_ai_client()

    prompt = f"""You are a master educator creating a diagnostic quiz from course notes.
Study Material:
{doc_context}

Task:
Create exactly 5 diagnostic multiple-choice questions covering key concepts in the study material above.
Language: {language}

CRITICAL RULES:
1. Every question must test a substantive, academic concept, theory, formula, or principle taught in the material.
2. STRICTLY FORBIDDEN: NEVER ask meta-questions about the document itself (such as author names, watermarks, total page counts, text casing, font formatting, or layout).
3. Provide exactly 4 options per question. Exactly one option must be correct.
4. 'correct_index' must be an integer from 0 to 3 corresponding to the correct option.
5. 'source_pages' must be a list containing the integer page numbers from the document where this concept is discussed (e.g. [1] or [2]). Only use pages present in the text above: {sorted(list(valid_pages))}.
6. 'explanation' must explain why the correct option is right based on the text.
7. IDs must be 'q1', 'q2', 'q3', 'q4', 'q5'.

Output strictly a JSON list of 5 objects matching this schema:
[
  {{
    "id": "q1",
    "concept": "Name of concept",
    "question": "Question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 0,
    "explanation": "Why this answer is correct...",
    "source_pages": [1]
  }}
]
"""

    system_instruction = "You are an expert tutor. Output strictly valid RFC-8259 JSON without markdown fences."

    # Attempt 1
    raw_response = client.generate_json(prompt, system_instruction=system_instruction)
    try:
        return validate_quiz_list(raw_response, valid_page_numbers=valid_pages, expected_count=5)
    except Exception as first_error:
        # Retry 1 time with error feedback
        retry_prompt = f"""The previous output was invalid due to: {first_error}
Please regenerate the 5 diagnostic questions strictly following the required JSON schema.
Valid document page numbers: {sorted(list(valid_pages))}

{prompt}
"""
        retry_response = client.generate_json(retry_prompt, system_instruction=system_instruction)
        try:
            return validate_quiz_list(retry_response, valid_page_numbers=valid_pages, expected_count=5)
        except Exception as second_error:
            raise InvalidModelResponseError(
                f"Failed to generate valid 5-question diagnostic quiz after retry: {second_error}"
            ) from second_error


# =====================================================================
# Function B: Evaluate diagnostic quiz
# =====================================================================

def evaluate_quiz(
    questions: list[dict[str, Any]],
    answers: dict[str, Any]
) -> dict[str, Any]:
    """Grade submitted answers and calculate concept-level performance.

    Args:
        questions: Stored full question dictionaries.
        answers: Map of question_id -> selected_option_index (0-3).

    Returns:
        Grading summary including overall accuracy, concept mastery, missed concepts,
        and weakest concept for adaptive follow-up.
    """
    return grade_answers(questions, answers)


# =====================================================================
# Function C: Select the next learning topic
# =====================================================================

def choose_next_topic(mastery: dict[str, Any]) -> str | None:
    """Choose the weakest attempted concept using deterministic rules.

    Args:
        mastery: Concept-level mastery dictionary.

    Returns:
        Concept name string, or None if no weak concept exists.
    """
    return choose_weakest_concept(mastery)


# =====================================================================
# Function D: Teach the selected concept (Conversational Tutor)
# =====================================================================

def teach_concept(
    concept: str,
    pages: list[dict[str, Any]],
    action: str = "initial_explanation",
    student_query: str | None = None,
    language: str = "English",
) -> dict[str, Any]:
    """Retrieve relevant passages and generate a conversational explanation grounded in the document.

    Follows Socratic tutoring principles:
    - Intuition before formal definitions
    - Relatable real-world analogies
    - Step-by-step formula breakdown
    - Visible citations strictly tied to retrieved evidence
    - Formative check questions to diagnose comprehension

    Actions supported:
    - 'initial_explanation': Hook -> intuition -> source connection -> quick check
    - 'simplify': Explain more simply (plain English, zero jargon)
    - 'example': Practical, scenario-based walkthrough
    - 'steps': Step-by-step problem solving recipe
    - 'quiz_me': Formative check question

    Args:
        concept: The concept to teach.
        pages: List of extracted page dictionaries.
        action: Tutor action/mode.
        student_query: Optional custom follow-up question or clarification.
        language: Explanation language (default English).

    Returns:
        Dictionary with conversational tutoring fields, evidence, and citations.
    """
    if not concept or not concept.strip():
        return {
            "concept": "Unknown",
            "action": action,
            "intuitive_intro": "",
            "explanation": "No concept was specified for tutoring.",
            "example": "",
            "formal_connection": "",
            "evidence": "",
            "source_pages": [],
            "evidence_found": False,
            "check_question": None,
        }

    # Retrieve matching passages from the document (augment query with student request if present)
    retrieval_query = f"{concept} {student_query}" if student_query else concept
    passages = retrieve_passages(pages, retrieval_query, limit=3)

    if not passages:
        # Fallback to concept-only retrieval if augmented query returned nothing
        passages = retrieve_passages(pages, concept, limit=3)

    if not passages:
        return {
            "concept": concept,
            "action": action,
            "intuitive_intro": f"Let's focus on **{concept}**.",
            "explanation": (
                f"We identified '{concept}' as an area for improvement. However, our search could not "
                "find a direct, explicit passage defining this concept in your uploaded document. "
                "Please review your class syllabus or general textbook notes on this topic."
            ),
            "example": "",
            "formal_connection": "Note: General principles applied since explicit text was not found in the uploaded PDF.",
            "evidence": "",
            "source_pages": [],
            "evidence_found": False,
            "check_question": None,
        }

    evidence_text = passages[0]["text"]
    source_pages = sorted(list({p["page"] for p in passages}))
    passages_context = format_passages_for_prompt(passages)

    # Action-specific guidance
    action_guidance = {
        "initial_explanation": (
            "Mode: Initial Concept Tutoring.\n"
            "1. 'intuitive_intro': A warm, engaging conversational hook (e.g. 'Let's untangle this...'). "
            "Introduce an intuitive scenario BEFORE throwing formal terms.\n"
            "2. 'explanation': Explain the core intuition in 2 short, conversational paragraphs. Focus on the 'why'.\n"
            "3. 'example': A concrete real-world scenario illustrating the idea.\n"
            "4. 'formal_connection': Connect the intuition to the formal definition, formula, or rules in the notes, referencing the source.\n"
            "5. 'check_question': One quick multi-choice check question (question, 3 options, correct_index, explanation) to test their intuition."
        ),
        "simplify": (
            "Mode: Explain More Simply (ELI5 / Plain English).\n"
            "1. 'intuitive_intro': Reassure the student ('No worries, let's break this down even simpler...').\n"
            "2. 'explanation': Strip away academic jargon completely. Use simple everyday language.\n"
            "3. 'example': A vivid, everyday metaphor.\n"
            "4. 'formal_connection': Relate back to the essential takeaway from the notes.\n"
            "5. 'check_question': A simple check question to confirm they got the core intuition."
        ),
        "example": (
            "Mode: Give Me an Example.\n"
            "1. 'intuitive_intro': 'Let's see this in action with a concrete case.'\n"
            "2. 'explanation': Describe the background and setup of the example.\n"
            "3. 'example': Detailed, numerical or scenario-driven walk-through from start to finish.\n"
            "4. 'formal_connection': Show how the formula/rule from the notes was applied.\n"
            "5. 'check_question': A question asking what would happen if one factor in the example changed."
        ),
        "steps": (
            "Mode: Show the Steps.\n"
            "1. 'intuitive_intro': 'Here is the step-by-step blueprint to solve this:'\n"
            "2. 'explanation': Walk through Step 1, Step 2, and Step 3 systematically.\n"
            "3. 'example': Brief application of these steps.\n"
            "4. 'formal_connection': Note key traps or edge cases mentioned in the notes.\n"
            "5. 'check_question': A question checking if they know which step comes first."
        ),
        "quiz_me": (
            "Mode: Test My Understanding.\n"
            "1. 'intuitive_intro': 'Let's put your intuition to the test!'\n"
            "2. 'explanation': Quick 1-sentence refresher of the key rule.\n"
            "3. 'example': None needed.\n"
            "4. 'formal_connection': Citing notes.\n"
            "5. 'check_question': A thoughtful concept question testing their understanding."
        ),
    }

    selected_mode = action_guidance.get(action, action_guidance["initial_explanation"])

    client = get_ai_client()
    prompt = f"""You are StudyMate AI, a patient, encouraging 1-on-1 personal tutor who has carefully studied the student's notes.

Target Concept: {concept}
Action Requested: {action}
Language: {language}

Verified passages from student's study notes:
{passages_context}

Tutor Teaching Contract:
- Speak like a friendly, expert tutor sitting right next to the student.
- Prefer short paragraphs, direct language, and one main idea at a time.
- Explain intuition BEFORE formal definitions.
- When formulas appear, explain what each symbol stands for.
- Factual claims about course material must be strictly grounded in the verified passages.
- NEVER invent page numbers. Only cite from valid pages: {source_pages}.
- Clearly distinguish everyday illustrative analogies from what the document explicitly states.

{selected_mode}

Output strictly valid RFC-8259 JSON matching this structure:
{{
  "concept": "{concept}",
  "intuitive_intro": "Conversational hook...",
  "explanation": "Core intuitive breakdown...",
  "example": "Real-world or scenario example...",
  "formal_connection": "How this connects to the notes on page(s) {source_pages}...",
  "source_pages": {source_pages},
  "check_question": {{
    "question": "Quick check question?",
    "options": ["Option 1", "Option 2", "Option 3"],
    "correct_index": 0,
    "explanation": "Why this answer is right..."
  }}
}}
"""
    system_instruction = (
        "You are an expert Socratic tutor. Teach intuitively, cite accurately, and respond strictly with valid RFC-8259 JSON."
    )

    try:
        raw_json = client.generate_json(prompt, system_instruction=system_instruction)
        validated = validate_teaching_response(raw_json, valid_page_numbers=set(source_pages))
        validated["action"] = action
        validated["evidence"] = evidence_text
        validated["evidence_found"] = True
        return validated
    except Exception:
        # Fallback to robust dictionary structure if LLM missed schema fields
        raw_text = client.generate_text(prompt, system_instruction=system_instruction)
        return {
            "concept": concept,
            "action": action,
            "intuitive_intro": f"Let's explore **{concept}** together.",
            "explanation": raw_text.strip() or f"Key principles of {concept} grounded in your notes.",
            "example": "",
            "formal_connection": f"Grounded in verified passages from page {source_pages}.",
            "evidence": evidence_text,
            "source_pages": source_pages,
            "evidence_found": True,
            "check_question": None,
        }


# =====================================================================
# Function E: Generate a targeted retest
# =====================================================================

def generate_retest(
    concept: str,
    pages: list[dict[str, Any]],
    previous_questions: list[dict[str, Any]] | None = None,
    language: str = "English"
) -> dict[str, Any]:
    """Generate one new multiple-choice question testing the selected concept.

    Args:
        concept: The concept to test.
        pages: List of extracted page dictionaries.
        previous_questions: Optional list of previously generated questions to avoid duplication.
        language: Target language.

    Returns:
        Validated question dictionary testing the target concept.
    """
    passages = retrieve_passages(pages, concept, limit=3)
    valid_pages = {p["page"] for p in passages} if passages else {p["page"] for p in pages if p.get("text")}

    passages_block = format_passages_for_prompt(passages) if passages else "No specific passages found."

    prev_questions_text = ""
    if previous_questions:
        q_texts = [f"- {q.get('question', '')}" for q in previous_questions if q.get("concept") == concept]
        if q_texts:
            prev_questions_text = "Previous questions to NOT repeat:\n" + "\n".join(q_texts)

    client = get_ai_client()
    prompt = f"""Create a single new multiple-choice retest question to check if the student has understood the concept: '{concept}'.

Target Concept: {concept}
Language: {language}

Reference Material:
{passages_block}

{prev_questions_text}

Rules:
1. The question must test the same underlying concept: '{concept}'.
2. It must be fresh and different from previous questions.
3. Provide exactly 4 options. Exactly one option must be correct.
4. 'correct_index' must be an integer between 0 and 3.
5. 'source_pages' must be a list containing valid page numbers from: {sorted(list(valid_pages))}.
6. 'id' must be 'retest_q1'.

Output strictly valid JSON with this schema:
{{
  "id": "retest_q1",
  "concept": "{concept}",
  "question": "Question text?",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_index": 0,
  "explanation": "Why this answer is correct...",
  "source_pages": {sorted(list(valid_pages))[:1]}
}}
"""
    system_instruction = "You are an expert exam creator. Output strictly valid RFC-8259 JSON."

    raw = client.generate_json(prompt, system_instruction=system_instruction)
    try:
        return validate_question_dict(raw, valid_page_numbers=valid_pages)
    except Exception as first_error:
        # Retry 1 time
        retry_prompt = f"The previous question was invalid: {first_error}. Please regenerate strictly:\n{prompt}"
        retry_raw = client.generate_json(retry_prompt, system_instruction=system_instruction)
        try:
            return validate_question_dict(retry_raw, valid_page_numbers=valid_pages)
        except Exception as second_error:
            raise InvalidModelResponseError(
                f"Failed to generate valid retest question for '{concept}': {second_error}"
            ) from second_error


# =====================================================================
# Function F: Evaluate the retest
# =====================================================================

def evaluate_retest(
    question: dict[str, Any],
    answer_index: int,
    previous_mastery: dict[str, Any]
) -> dict[str, Any]:
    """Grade the follow-up retest answer and compute updated learning progress.

    Args:
        question: Retest question dictionary.
        answer_index: Index of selected answer (0-3).
        previous_mastery: Concept mastery map from the diagnostic quiz.

    Returns:
        Retest evaluation result containing updated and previous mastery.
    """
    return grade_retest_submission(question, answer_index, previous_mastery)


# =====================================================================
# Optional Function: Topic overview
# =====================================================================

def extract_topics(
    pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Identify the main concepts covered by the document (optional helper).

    Returns:
        List of topic dictionaries: [{"topic": "...", "page": 1, "summary": "..."}]
    """
    doc_context, _ = _prepare_document_context(pages, max_chars=6000)
    client = get_ai_client()

    prompt = f"""Analyze the following course material and extract the 3 to 6 major topics/concepts covered.

Document Material:
{doc_context}

Output strictly a JSON list of objects:
[
  {{
    "topic": "Name of topic",
    "summary": "One sentence summary of this topic"
  }}
]
"""
    try:
        topics = client.generate_json(prompt, system_instruction="Output valid JSON list of topics.")
        if isinstance(topics, list):
            return topics
    except Exception:
        pass
    return []
