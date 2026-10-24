"""StudyMate AI - Streamlit Web Application.

Modern, high-contrast, adaptive AI tutor interface powered by open-weight Gemma models.
Features personalized Socratic tutoring, concept-by-concept navigation, and persistent per-topic progress.
"""

import streamlit as st
from pdf_extractor import extract_pdf_text
from backend import (
    choose_next_topic,
    evaluate_quiz,
    evaluate_retest,
    generate_quiz,
    generate_retest,
    teach_concept,
)
from studymate.schemas import StudyMateError

# =====================================================================
# Page Configuration & Modern Theme Styling
# =====================================================================

st.set_page_config(
    page_title="StudyMate AI — Adaptive AI Tutor",
    page_icon="🎓",
    layout="wide",
)

# Custom High-Contrast, Vibrant CSS
CUSTOM_CSS = """
<style>
/* Main app canvas */
.stApp {
    background-color: #0b0f17;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Typography contrast */
h1, h2, h3, h4 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}
p, span, label {
    color: #cbd5e1;
}

/* Card components */
.study-card {
    background: #1e293b;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

.tutor-bubble {
    background: linear-gradient(135deg, #1e293b 0%, #172033 100%);
    border-left: 4px solid #6366f1;
    border-top: 1px solid rgba(99, 102, 241, 0.2);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 18px;
}

.example-box {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 8px;
    padding: 16px;
    margin: 14px 0;
    color: #e0e7ff;
}

.evidence-box {
    background: rgba(16, 185, 129, 0.1);
    border-left: 3px solid #10b981;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 14px 0;
    color: #a7f3d0;
}

/* Pill badges */
.pill-danger {
    background: rgba(244, 63, 94, 0.2);
    color: #fda4af !important;
    border: 1px solid rgba(244, 63, 94, 0.4);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
}

.pill-warning {
    background: rgba(245, 158, 11, 0.2);
    color: #fde68a !important;
    border: 1px solid rgba(245, 158, 11, 0.4);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
}

.pill-success {
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7 !important;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
}

.pill-neutral {
    background: rgba(148, 163, 184, 0.2);
    color: #cbd5e1 !important;
    border: 1px solid rgba(148, 163, 184, 0.4);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
}

/* Custom interactive Radio group */
div[role="radiogroup"] > label {
    background: #182234 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 12px 18px !important;
    margin-bottom: 10px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    color: #f1f5f9 !important;
}
div[role="radiogroup"] > label:hover {
    border-color: #6366f1 !important;
    background: #1e293b !important;
}

/* Primary Button Styling */
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.5) !important;
}

/* Secondary Buttons */
button[kind="secondary"] {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
button[kind="secondary"]:hover {
    background: #27354a !important;
    border-color: #6366f1 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Title & Header
st.title("🎓 StudyMate AI")
st.markdown(
    "<p style='color:#94a3b8; font-size:1.05rem; margin-top:-10px; margin-bottom:20px;'>"
    "Personalized Socratic AI Tutor powered by open-weight <b>Gemma 4</b>."
    "</p>",
    unsafe_allow_html=True,
)

# =====================================================================
# Session State Initialization (Persistent Per-Topic)
# =====================================================================

if "pages" not in st.session_state:
    st.session_state.pages = None
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None
if "live_mastery" not in st.session_state:
    st.session_state.live_mastery = {}
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None

# Dictionaries keyed by concept name to prevent resetting progress on topic switch
if "topic_tutor_turns" not in st.session_state:
    st.session_state.topic_tutor_turns = {}
if "topic_retest_question" not in st.session_state:
    st.session_state.topic_retest_question = {}
if "topic_retest_result" not in st.session_state:
    st.session_state.topic_retest_result = {}
if "check_answered" not in st.session_state:
    st.session_state.check_answered = {}


def reset_learning_state():
    """Reset all quiz and tutoring state while keeping uploaded pages."""
    st.session_state.quiz = None
    st.session_state.eval_result = None
    st.session_state.live_mastery = {}
    st.session_state.current_topic = None
    st.session_state.topic_tutor_turns = {}
    st.session_state.topic_retest_question = {}
    st.session_state.topic_retest_result = {}
    st.session_state.check_answered = {}


# =====================================================================
# Sidebar: Document Upload & Settings
# =====================================================================

with st.sidebar:
    st.header("📂 1. Study Notes")
    uploaded_file = st.file_uploader("Upload course notes (PDF)", type=["pdf"])
    language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Hindi"])

    if uploaded_file is not None:
        if st.button("Extract Course Notes", type="primary", use_container_width=True):
            with st.spinner("Extracting page-aware text..."):
                try:
                    pdf_bytes = uploaded_file.read()
                    pages = extract_pdf_text(pdf_bytes)
                    st.session_state.pages = pages
                    reset_learning_state()
                    st.success(f"Extracted {len(pages)} pages successfully!")
                except StudyMateError as e:
                    st.error(f"Extraction error: {e}")
                except Exception as ex:
                    st.error(f"Unexpected error: {ex}")

    st.markdown("---")
    st.write("**Or test instantly:**")
    if st.button("📚 Load Sample Notes (Operating Systems)", use_container_width=True):
        st.session_state.pages = [
            {
                "page": 1,
                "text": (
                    "Operating Systems: Concurrency & Synchronization.\n\n"
                    "A race condition occurs in concurrent programming when multiple threads access "
                    "shared memory concurrently and the final outcome depends on the order of execution. "
                    "The region of code accessing shared variables is called the critical section. "
                    "Mutual exclusion (mutex) guarantees that at most one thread can execute within "
                    "the critical section at any time, preventing data corruption."
                )
            },
            {
                "page": 2,
                "text": (
                    "Semaphores and Deadlocks.\n\n"
                    "A counting semaphore maintains an integer value representing available units of a resource. "
                    "The wait() operation decrements the count and blocks if the value is zero or less. "
                    "The signal() operation increments the count and wakes sleeping threads. "
                    "A deadlock occurs when processes are permanently blocked waiting for resources held by each other. "
                    "The four necessary conditions for deadlock are: Mutual Exclusion, Hold and Wait, "
                    "No Preemption, and Circular Wait."
                )
            },
            {
                "page": 3,
                "text": (
                    "Classic Synchronization Problems.\n\n"
                    "In the Producer-Consumer problem, a shared bounded buffer is coordinated using two counting semaphores: "
                    "'empty_slots' initialized to buffer size N, and 'full_slots' initialized to 0, alongside a mutex lock. "
                    "In the Dining Philosophers problem, circular deadlock can be prevented by breaking the circular wait condition, "
                    "for example by having odd philosophers pick up their left chopstick first and even philosophers pick up their right chopstick first."
                )
            }
        ]
        reset_learning_state()
        st.success("Loaded 3 pages of Operating Systems notes!")
        st.rerun()

    st.markdown("---")
    st.caption("StudyMate AI • Hacktoberfest 2026")

# =====================================================================
# Main Application Flow
# =====================================================================

if not st.session_state.pages:
    st.info("👈 Please upload your course notes (PDF) or click 'Load Sample Notes' in the sidebar to begin.")
else:
    pages = st.session_state.pages

    # -----------------------------------------------------------------
    # Step 1: Generate & Display Diagnostic Quiz
    # -----------------------------------------------------------------
    if st.session_state.quiz is None:
        st.subheader("📝 Diagnostic Assessment")
        st.write(
            "Take this 5-question diagnostic quiz so StudyMate can understand your current "
            "mastery and tailor 1-on-1 tutoring to the exact concepts you need."
        )

        if st.button("Generate Diagnostic Quiz", type="primary"):
            with st.spinner("Generating diagnostic quiz with Gemma 4..."):
                try:
                    quiz = generate_quiz(pages, language=language)
                    st.session_state.quiz = quiz
                    st.rerun()
                except StudyMateError as err:
                    st.error(f"Could not generate quiz: {err}")
                except Exception as ex:
                    st.error(f"Unexpected error: {ex}")

    # -----------------------------------------------------------------
    # Step 2: Diagnostic Submission & Grading
    # -----------------------------------------------------------------
    elif st.session_state.eval_result is None:
        st.subheader("📝 Diagnostic Assessment")
        st.write("Answer each question to test your baseline understanding:")

        quiz = st.session_state.quiz
        user_answers = {}

        with st.form("quiz_form"):
            for idx, q in enumerate(quiz, start=1):
                st.markdown(
                    f"<div class='study-card'>"
                    f"<h4 style='color:#f8fafc; margin-bottom:4px;'>Question {idx}: {q['question']}</h4>"
                    f"<span class='pill-neutral'>Concept: {q['concept']}</span> &nbsp; "
                    f"<span style='color:#94a3b8; font-size:0.8rem;'>Cited from Page(s): {q.get('source_pages', [])}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                selected = st.radio(
                    label=f"Options for {q['id']}",
                    options=list(range(len(q["options"]))),
                    format_func=lambda i, opts=q["options"]: opts[i],
                    key=f"radio_{q['id']}",
                    label_visibility="collapsed",
                )
                user_answers[q["id"]] = selected
                st.write("")

            submitted = st.form_submit_button("Submit Diagnostic Answers", type="primary", use_container_width=True)
            if submitted:
                eval_res = evaluate_quiz(quiz, user_answers)
                st.session_state.eval_result = eval_res
                st.session_state.live_mastery = eval_res["mastery"]
                st.session_state.current_topic = eval_res.get("weakest_concept")
                st.session_state.topic_tutor_turns = {}
                st.session_state.topic_retest_question = {}
                st.session_state.topic_retest_result = {}
                st.session_state.check_answered = {}
                st.rerun()

    # -----------------------------------------------------------------
    # Step 3: Mastery Dashboard & Topic Navigator
    # -----------------------------------------------------------------
    else:
        eval_res = st.session_state.eval_result
        mastery = st.session_state.live_mastery

        st.subheader("📊 Concept Mastery Dashboard")

        # Top Metric Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Accuracy", f"{eval_res['overall_accuracy']}%")
        with col2:
            st.metric("Score", f"{eval_res['correct_answers']} / {eval_res['attempted_answers']}")
        with col3:
            weak_target = choose_next_topic(mastery)
            st.metric("Next Recommended Focus", weak_target if weak_target else "All Mastered! 🎉")

        st.markdown("---")

        # Concept Navigator
        st.markdown("#### 🗺️ Course Concept Navigator")
        st.caption("Select any topic to view its persistent tutor conversation, notes, and retest progress:")

        concept_cols = st.columns(len(mastery))
        for col_idx, (concept_name, stats) in enumerate(mastery.items()):
            col = concept_cols[col_idx % len(concept_cols)]
            with col:
                acc = stats.get("accuracy")
                is_active = (st.session_state.current_topic == concept_name)

                # Badge & card styling
                if acc is None:
                    badge_html = "<span class='pill-neutral'>⚪ Untested</span>"
                elif acc == 100.0:
                    badge_html = "<span class='pill-success'>🟢 100% Mastered</span>"
                elif acc >= 50.0:
                    badge_html = f"<span class='pill-warning'>🟡 {acc}% Reviewing</span>"
                else:
                    badge_html = f"<span class='pill-danger'>🔴 {acc}% Focus Area</span>"

                border_style = "2px solid #6366f1" if is_active else "1px solid rgba(255,255,255,0.08)"
                bg_style = "#1e293b" if not is_active else "#25334d"

                # Check if this topic has already completed a retest
                has_retested = concept_name in st.session_state.topic_retest_result
                retest_tag = " <span style='font-size:0.75rem; color:#a5b4fc;'>✓ Retested</span>" if has_retested else ""

                st.markdown(
                    f"<div style='background:{bg_style}; border:{border_style}; border-radius:10px; padding:14px; margin-bottom:10px;'>"
                    f"<b style='color:#f8fafc; font-size:0.95rem;'>{concept_name}</b>{retest_tag}<br>"
                    f"<div style='margin-top:8px;'>{badge_html}</div>"
                    f"<div style='font-size:0.75rem; color:#94a3b8; margin-top:6px;'>"
                    f"Score: {stats['correct']}/{stats['attempted']}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Selected" if is_active else "Learn Topic",
                    key=f"nav_btn_{concept_name}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    # Switch active topic WITHOUT clearing any previous topic's progress
                    st.session_state.current_topic = concept_name
                    st.rerun()

        st.markdown("---")

        # -------------------------------------------------------------
        # Step 4: Socratic 1-on-1 Tutoring Session (Persistent Per-Topic)
        # -------------------------------------------------------------
        current_c = st.session_state.current_topic

        if current_c:
            st.markdown(
                f"<div class='tutor-bubble'>"
                f"<h3 style='margin:0; color:#f8fafc;'>🧑‍🏫 Personal Tutoring: <span style='color:#a5b4fc;'>{current_c}</span></h3>"
                f"<p style='color:#94a3b8; font-size:0.88rem; margin-top:4px;'>Grounded in verified passages from your study notes.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Retrieve current topic's preserved chat turns
            current_turns = st.session_state.topic_tutor_turns.get(current_c, [])

            if not current_turns:
                st.write(
                    f"Let's untangle **{current_c}** using intuition, real-world examples, "
                    "and the exact formulas from your notes."
                )
                if st.button("Start 1-on-1 Tutoring Session", type="primary", use_container_width=True):
                    with st.spinner("Preparing intuitive explanation grounded in your notes..."):
                        try:
                            initial_resp = teach_concept(current_c, pages, action="initial_explanation", language=language)
                            st.session_state.topic_tutor_turns.setdefault(current_c, []).append(initial_resp)
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Tutoring session could not start: {ex}")
            else:
                # Render conversation turns for current topic
                for turn_idx, turn in enumerate(current_turns, start=1):
                    with st.chat_message("assistant", avatar="🧑‍🏫"):
                        if turn.get("intuitive_intro"):
                            st.markdown(f"<p style='font-size:1.05rem; color:#cbd5e1; font-style:italic;'>{turn['intuitive_intro']}</p>", unsafe_allow_html=True)

                        st.markdown(f"<p style='font-size:1rem; color:#f1f5f9; line-height:1.6;'>{turn.get('explanation', '')}</p>", unsafe_allow_html=True)

                        if turn.get("example"):
                            st.markdown(
                                f"<div class='example-box'>"
                                f"<b style='color:#a5b4fc;'>💡 Intuitive Scenario:</b><br>{turn['example']}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        if turn.get("formal_connection"):
                            st.markdown(
                                f"<div style='color:#94a3b8; font-size:0.88rem; margin:8px 0;'>"
                                f"📘 <b>Connecting to your notes:</b> {turn['formal_connection']}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        if turn.get("evidence_found") and turn.get("evidence"):
                            with st.expander(f"🔍 Cited Source from Notes (Page {turn.get('source_pages', [])})"):
                                st.markdown(
                                    f"<div class='evidence-box'>"
                                    f"<i>\"{turn['evidence']}\"</i>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        # Quick Check Question
                        chk = turn.get("check_question")
                        if chk and isinstance(chk, dict):
                            st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
                            st.markdown(f"**🎯 Quick Check — Your Turn:** {chk['question']}")
                            chk_key = f"chk_turn_{current_c}_{turn_idx}"
                            selected_chk = st.radio(
                                "Options:",
                                options=list(range(len(chk["options"]))),
                                format_func=lambda i, opts=chk["options"]: opts[i],
                                key=f"radio_{chk_key}",
                                label_visibility="collapsed",
                            )
                            if st.button(f"Check Answer", key=f"btn_{chk_key}"):
                                st.session_state.check_answered[chk_key] = (selected_chk == chk["correct_index"])

                            if chk_key in st.session_state.check_answered:
                                if st.session_state.check_answered[chk_key]:
                                    st.success(f"🎉 Correct! {chk.get('explanation', '')}")
                                else:
                                    st.warning(f"Not quite. {chk.get('explanation', '')}")

                # Interactive Tutor Action Bar
                st.markdown("#### How would you like StudyMate to help next?")
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    if st.button("🗣️ Explain simpler", use_container_width=True):
                        with st.spinner("Breaking it down in plain English..."):
                            resp = teach_concept(current_c, pages, action="simplify", language=language)
                            st.session_state.topic_tutor_turns.setdefault(current_c, []).append(resp)
                            st.rerun()

                with col_b:
                    if st.button("💡 Give an example", use_container_width=True):
                        with st.spinner("Crafting a practical walkthrough..."):
                            resp = teach_concept(current_c, pages, action="example", language=language)
                            st.session_state.topic_tutor_turns.setdefault(current_c, []).append(resp)
                            st.rerun()

                with col_c:
                    if st.button("🔢 Show the steps", use_container_width=True):
                        with st.spinner("Outlining problem-solving steps..."):
                            resp = teach_concept(current_c, pages, action="steps", language=language)
                            st.session_state.topic_tutor_turns.setdefault(current_c, []).append(resp)
                            st.rerun()

                with col_d:
                    if st.button("🎯 Test understanding", type="primary", use_container_width=True):
                        with st.spinner("Generating targeted retest question..."):
                            retest_q = generate_retest(
                                current_c,
                                pages,
                                previous_questions=st.session_state.quiz,
                                language=language,
                            )
                            st.session_state.topic_retest_question[current_c] = retest_q
                            # Clear old retest result so they can answer the fresh question
                            st.session_state.topic_retest_result.pop(current_c, None)
                            st.rerun()

                # -----------------------------------------------------
                # Step 5: Follow-Up Retest (Persistent Per-Topic)
                # -----------------------------------------------------
                current_retest_q = st.session_state.topic_retest_question.get(current_c)
                current_retest_res = st.session_state.topic_retest_result.get(current_c)

                if current_retest_q is not None:
                    st.markdown("---")
                    st.subheader(f"🎯 Targeted Retest: {current_c}")
                    rq = current_retest_q

                    if current_retest_res is None:
                        st.markdown(
                            f"<div class='study-card'>"
                            f"<h4 style='color:#f8fafc; margin-bottom:8px;'>{rq['question']}</h4>"
                            f"<span style='color:#94a3b8; font-size:0.8rem;'>Testing: <b>{current_c}</b></span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        with st.form(f"retest_form_{current_c}"):
                            retest_ans = st.radio(
                                "Select your answer:",
                                options=list(range(len(rq["options"]))),
                                format_func=lambda i, opts=rq["options"]: opts[i],
                                key=f"retest_radio_{current_c}",
                            )
                            submitted_retest = st.form_submit_button("Submit Retest Answer", type="primary", use_container_width=True)

                            if submitted_retest:
                                retest_eval = evaluate_retest(
                                    question=rq,
                                    answer_index=retest_ans,
                                    previous_mastery=st.session_state.live_mastery,
                                )
                                # Save permanently under current_c
                                st.session_state.topic_retest_result[current_c] = retest_eval
                                st.session_state.live_mastery = retest_eval["updated_mastery"]
                                st.rerun()
                    else:
                        retest_res = current_retest_res
                        if retest_res["is_correct"]:
                            st.success("🎉 Correct! You demonstrated improved mastery on this concept!")
                        else:
                            st.warning("Keep practicing! Review the tutor explanations above and try again.")

                        st.markdown(f"**Explanation:** {retest_res['explanation']}")

                        # Before vs After Progress Comparison Card
                        st.markdown("#### 📈 Learning Progress")
                        prev_c_acc = retest_res["previous_mastery"].get(current_c, {}).get("accuracy", 0.0)
                        upd_c_acc = retest_res["updated_mastery"].get(current_c, {}).get("accuracy", 0.0)
                        delta_val = round(upd_c_acc - prev_c_acc, 1)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric(f"Initial '{current_c}'", f"{prev_c_acc}%")
                        with c2:
                            st.metric(f"Updated '{current_c}'", f"{upd_c_acc}%", delta=f"+{delta_val}%" if delta_val >= 0 else f"{delta_val}%")

                        # Next Topic Navigation Action
                        st.markdown("---")
                        next_weak = choose_next_topic(st.session_state.live_mastery)

                        col_next1, col_next2 = st.columns([2, 1])

                        with col_next1:
                            if next_weak and next_weak != current_c:
                                st.markdown(f"#### 🚀 Ready for the next topic?")
                                st.write(f"StudyMate suggests continuing to **{next_weak}** to build complete mastery.")
                                if st.button(f"👉 Proceed to Next Topic: {next_weak}", type="primary", use_container_width=True):
                                    st.session_state.current_topic = next_weak
                                    st.rerun()
                            elif not next_weak:
                                st.balloons()
                                st.success("🌟 Congratulations! You have mastered all concepts in this study material!")

                        with col_next2:
                            if st.button("🔄 Retest This Topic Again", use_container_width=True):
                                with st.spinner("Generating fresh retest question..."):
                                    new_rq = generate_retest(
                                        current_c,
                                        pages,
                                        previous_questions=st.session_state.quiz,
                                        language=language,
                                    )
                                    st.session_state.topic_retest_question[current_c] = new_rq
                                    st.session_state.topic_retest_result.pop(current_c, None)
                                    st.rerun()

                        if st.button("🔄 Reset & Start New Session", type="secondary"):
                            for key in list(st.session_state.keys()):
                                del st.session_state[key]
                            st.rerun()
        else:
            st.success("🌟 Outstanding! All concepts are at 100% mastery.")
            if st.button("🔄 Start New Session", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
