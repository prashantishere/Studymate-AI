# 🎓 StudyMate AI — Adaptive AI Tutor

> **Hacktoberfest 2026** Open-Source AI Track Submission  
> *Stewarded by Major League Hacking (MLH) & DEV, presented by DigitalOcean.*

StudyMate AI is an adaptive, evidence-grounded AI tutor that turns course material into a personalized mastery-based learning loop. Rather than acting as a generic chatbot, StudyMate assesses student understanding, deterministically calculates concept-level mastery, teaches weak concepts using direct evidence cited from the student's study notes, and conducts targeted retests to track improvement.

---

## 🔄 The Adaptive Learning Loop

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
     [Deterministic Grading]
               │
               ▼
   [Concept Mastery Calculation]
               │
               ▼
      [Weak Concept Selected]
               │
               ▼
   [Evidence-Grounded Teaching] ──► (Cites exact text & page numbers)
               │
               ▼
     [Targeted Follow-Up MCQ]
               │
               ▼
  [Updated Progress & Mastery]
```

**The core differentiator:** *The student's demonstrated understanding determines what the tutor teaches next.*

---

## 🏗️ Project Architecture

```
studymate-ai/
├── app.py                     # Streamlit frontend (Owned by Frontend Teammate)
├── backend.py                 # Core backend API consumed by app.py
├── pdf_extractor.py           # Page-aware PDF text extraction (Owned by PDF Teammate)
├── requirements.txt           # Python dependencies
├── .env.example               # Configuration template for hosted Gemma inference
├── .gitignore                 # Secrets and environment cache exclusions
├── LICENSE                    # MIT License
├── README.md                  # Project documentation & integration guide
├── studymate/                 # Backend core modules
│   ├── __init__.py
│   ├── ai_client.py           # Hosted Gemma inference client (OpenAI-compatible)
│   ├── retrieval.py           # Deterministic passage retrieval with page numbers
│   ├── mastery.py             # Deterministic grading & tie-breaking rules
│   └── schemas.py             # Pydantic schemas, validation, and error hierarchy
└── tests/                     # Comprehensive automated unit & integration tests
    ├── test_backend.py        # Schema validation, error cases & end-to-end learning loop
    ├── test_mastery.py        # Grading accuracy, tie-breaking & retest score updates
    ├── test_pdf_extractor.py  # PDF parsing & boundary tests
    └── test_retrieval.py      # Keyword retrieval & page preservation tests
```

---

## ⚡ Quickstart

### 1. Prerequisites
- Python 3.10+
- A hosted Gemma inference API key (e.g., from [Google AI Studio](https://aistudio.google.com/) or any OpenAI-compatible provider like Groq/OpenRouter).

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-org/studymate-ai.git
cd studymate-ai

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set your configuration:
```env
GEMMA_API_KEY=your_actual_api_key_here
GEMMA_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMMA_MODEL=gemma-2-9b-it
STUDYMATE_MOCK_AI=0
```

> **Note for Offline Testing:** Setting `STUDYMATE_MOCK_AI=1` enables headless mock inference without needing an active API key or internet connection.

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:

```bash
pytest tests/ -v
```

All 22 unit tests test:
- Diagnostic quiz and retest schema validation
- 4-option enforcement and non-empty choice checks
- Duplicate question ID rejection
- Page reference integrity against input documents
- Deterministic answer grading (correct, incorrect, unanswered)
- Untested concepts marked appropriately (not failed or 0%)
- Deterministic tie-breaking for equal-accuracy concepts
- Retest mastery updates preserving initial baseline
- PDF extraction and corruption handling
- Keyword passage retrieval preserving 1-indexed page citations

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
- `generate_quiz(pages, language="English") -> list[dict]`
- `evaluate_quiz(questions, answers) -> dict`
- `choose_next_topic(mastery) -> str | None`
- `teach_concept(concept, pages, language="English") -> dict`
- `generate_retest(concept, pages, previous_questions=None, language="English") -> dict`
- `evaluate_retest(question, answer_index, previous_mastery) -> dict`

---

## 📜 License
MIT License. Open-source for everyone.
