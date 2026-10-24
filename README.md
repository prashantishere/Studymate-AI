# 🎓 StudyMate AI — Adaptive Socratic AI Tutor

> **Hacktoberfest 2026** Open-Source AI Track Submission  
> *Stewarded by Major League Hacking (MLH) & DEV, presented by DigitalOcean.*  
> **Powered by Google Gemma 4** (`gemma-4-26b-a4b-it` / `gemma-4-31b-it`) via the Gemini API & `google-genai` SDK.

[![License: MIT](https://img.shields.io/badge/License-MIT-indigo.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model: Gemma 4](https://img.shields.io/badge/Model-Google%20Gemma%204-6366f1.svg)](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
[![Tests Passing](https://img.shields.io/badge/Tests-24%20Passing-10b981.svg)](tests/)

StudyMate AI is an adaptive, evidence-grounded AI tutor that turns static lecture notes into a personalized Socratic learning loop. Rather than acting as a generic PDF chatbot or a mindless quiz dumper, StudyMate assesses student understanding, deterministically calculates concept-level mastery, teaches weak concepts conversationally using direct evidence cited from the student's study notes, and conducts targeted retests to track real improvement.

---

## 🌟 The Core Differentiator

> **The student's demonstrated understanding directly dictates what the tutor teaches next.**

```
       [Upload Notes (PDF)]
                │
                ▼
    [Page-Aware Text Extraction]
                │
                ▼
     [5-Question Diagnostic Quiz]
                │
                ▼
      [Deterministic Grading] ────► (NO LLM grading hallucinations)
                │
                ▼
    [Concept-Level Mastery %]
                │
                ▼
    [Course Concept Navigator] ───► (Clickable, persistent per-topic progress)
                │
                ▼
   [Socratic 1-on-1 Tutoring] ───► (Intuition ➔ Example ➔ Notes citation ➔ Quick check)
                │
        ┌───────┴───────────────────────────────┐
        ▼                                       ▼
  [4 Tutor Actions]                       [Cited Source Evidence]
  • Explain simpler                       (Exact quotation + Page #)
  • Give me an example                          │
  • Show the steps                              ▼
  • Ready for retest                   [Targeted Retest Question]
                                                │
                                                ▼
                                   [Before vs. After Mastery Delta]
                                                │
                                                ▼
                                    [Proceed to Next Topic ➔]
```

---

## 🚀 Key Features

### 1. 🧑‍🏫 Socratic Conversational Tutoring (Not a Wall of Text)
- **Intuition Before Jargon:** Introduces everyday scenarios first before connecting to formal definitions or equations.
- **Strict Evidence Grounding:** Factual explanations are tied directly to verified passages extracted from the student's uploaded PDF with exact 1-indexed page citations.
- **Honest Limitations:** If the notes lack coverage on a topic, the tutor explicitly states so rather than hallucinating citations.

### 2. ⚡ The Four Interactive Tutor Actions
Students control their own learning pace with one-click actions:
- 🗣️ **Explain simpler:** Strips away jargon and explains the concept using plain English and an everyday metaphor.
- 💡 **Give me an example:** Walks through a concrete, scenario-based walkthrough step-by-step.
- 🔢 **Show the steps:** Outlines a systematic problem-solving recipe (Step 1, Step 2, Step 3).
- 🎯 **Test understanding:** Automatically generates a targeted retest question.

### 3. 🎯 Formative "Quick Check — Your Turn"
Every explanation includes an interactive micro-check question to test the student's intuition with immediate feedback before facing the formal retest.

### 4. 🗺️ Course Concept Navigator & Persistent Per-Topic Memory
- Interactive dashboard displays color-coded badges for every concept:
  - 🟢 **Mastered** (100% accuracy)
  - 🟡 **Reviewing** (Partial accuracy)
  - 🔴 **Focus Area** (Needs attention)
  - 🟣 **✓ Retested** (Follow-up retest completed)
- **Persistent State:** Switching between topics never resets previous conversations, examples, or retest scores. Students can freely jump between concepts and their progress remains intact.

### 5. 🛡️ Scanned Image & Watermark Noise Guard
Detects and warns when a PDF contains only repetitive watermarks/headers (such as scanned handwritten photos) instead of digital selectable text, avoiding nonsensical quiz generation.

### 6. 📚 One-Click Instant Demo Notes
Includes pre-loaded lecture notes on **Operating Systems (Concurrency, Semaphores & Deadlocks)** so judges and teammates can test the complete learning loop in 5 seconds with zero setup.

---

## 🏗️ Project Architecture & Team Boundaries

The codebase follows a modular design with strict separation of concerns across a 4-person hackathon team:

```
studymate-ai/
├── app.py                     # Streamlit frontend (Owned by Frontend Teammate)
├── backend.py                 # Primary public interface for all backend functions
├── pdf_extractor.py           # Page-aware PDF text extraction (Owned by PDF Teammate)
├── requirements.txt           # Minimal dependencies (google-genai, openai, pypdf, streamlit, pytest)
├── .env.example               # Config template for hosted Gemma inference
├── .gitignore                 # Secrets and environment cache exclusions
├── LICENSE                    # MIT License
├── README.md                  # Project documentation & integration guide
├── studymate/                 # Backend core package
│   ├── __init__.py
│   ├── ai_client.py           # Hosted Gemma 4 client (Google GenAI & OpenAI-compatible)
│   ├── retrieval.py           # Deterministic lexical passage retrieval with page numbers
│   ├── mastery.py             # Deterministic grading & tie-breaking rules
│   └── schemas.py             # Pydantic schemas, validation, and error hierarchy
└── tests/                     # 24 automated unit & integration tests
    ├── test_backend.py        # Schema validation, error cases & end-to-end learning loop
    ├── test_mastery.py        # Grading accuracy, tie-breaking & retest score updates
    ├── test_pdf_extractor.py  # PDF parsing boundaries & error tests
    └── test_retrieval.py      # Keyword retrieval & page preservation tests
```

---

## ⚡ Quickstart

### 1. Prerequisites
- **Python 3.10+**
- A free API key from [Google AI Studio](https://aistudio.google.com/).

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/your-org/studymate-ai.git
cd studymate-ai

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Open `.env` and configure:
```env
# Google AI Studio API Key
GEMINI_API_KEY=your_actual_api_key_here

# Supported Gemma 4 models:
# - gemma-4-26b-a4b-it
# - gemma-4-31b-it
GEMMA_MODEL=gemma-4-26b-a4b-it

# Set to 1 for offline testing without external API calls, or 0 for live model calls
STUDYMATE_MOCK_AI=0
```

### 4. Run the Application
```powershell
streamlit run app.py
```

---

## 🧪 Automated Testing

Run the full test suite with `pytest`:

```powershell
pytest tests/ -v
```

All **24 tests** execute in under 0.3 seconds:
- Diagnostic quiz & retest schema validation
- 4-option enforcement and non-empty choice validation
- Duplicate question ID rejection
- Page citation verification (rejects hallucinated page numbers)
- Deterministic answer grading (correct, incorrect, unanswered)
- Untested concepts marked appropriately (not failed or 0%)
- Deterministic tie-breaking hierarchy (accuracy ➔ attempts ➔ alphabetical)
- Socratic tutor actions (`simplify`, `example`, `steps`, `quiz_me`)
- Evidence honesty when PDF lacks concept coverage
- PDF corruption and empty bytes handling
- Lexical passage retrieval preserving 1-indexed page citations

---

## 🤝 Module Integration Contracts

### PDF Extraction Contract (`pdf_extractor.py`)
```python
def extract_pdf_text(pdf_bytes: bytes) -> list[dict]:
    """
    Returns:
    [
        {"page": 1, "text": "Extracted text from page 1..."},
        {"page": 2, "text": "Extracted text from page 2..."}
    ]
    """
```

### Public Backend API (`backend.py`)
- `generate_quiz(pages, language="English") -> list[dict]`: Generates 5 validated diagnostic MCQs.
- `evaluate_quiz(questions, answers) -> dict`: Deterministic grading, concept mastery, and weakest topic.
- `choose_next_topic(mastery) -> str | None`: Chooses the weakest concept using deterministic tie-breaking.
- `teach_concept(concept, pages, action="initial_explanation", student_query=None, language="English") -> dict`: Socratic tutoring grounded in notes.
- `generate_retest(concept, pages, previous_questions=None, language="English") -> dict`: Generates 1 fresh MCQ on the weak concept.
- `evaluate_retest(question, answer_index, previous_mastery) -> dict`: Evaluates retest and returns updated mastery progress.

---

## 📜 License
MIT License. Open-source for everyone. Built with ❤️ for **Hacktoberfest 2026**.
