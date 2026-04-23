import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from gap_tracker import GapTracker, extract_topic
from guardrails import CRISIS_RESPONSE, REFUSAL_MESSAGE, detect_crisis, detect_injection
from logging_setup import get_logger
from rag_service import RAGService
from rate_limiter import RateLimiter, RateLimitError
from student_profile import EDUCATION_LEVELS, FAMILIARITY_LEVELS, StudentProfile
from teaching_assistant import TeachingAssistant
import web_search

load_dotenv()
log = get_logger(__name__)

st.set_page_config(page_title="AI Teaching Assistant", page_icon="🎓", layout="wide")


# ----------------------------------------------------------- provider config
# NOTE: `env` is used ONLY as a default pre-fill when TA_USE_ENV_KEYS=1 (dev
# convenience). We never write the key back to os.environ, so one user's key
# can't leak into another's session via shared process env.
PROVIDERS = {
    "OpenAI":        {"value": "openai", "env": "OPENAI_API_KEY", "label": "OpenAI API Key",         "help": "platform.openai.com → API keys"},
    "Google Gemini": {"value": "gemini", "env": "GOOGLE_API_KEY", "label": "Gemini API Key (free!)", "help": "aistudio.google.com → Get API key"},
}

# In single-user dev mode we pre-fill from env. Disabled by default so a
# multi-user deploy doesn't leak the operator's key to every visitor.
_USE_ENV_KEYS = os.getenv("TA_USE_ENV_KEYS", "0") == "1"


# ---------------------------------------------------------------- session init
def _ensure_state():
    ss = st.session_state
    # Stable per-session UUID. Streamlit gives each tab/user its own
    # session_state, so this naturally isolates vector stores per user.
    if "session_id" not in ss:
        ss.session_id = uuid.uuid4().hex
        log.info("new session started: %s", ss.session_id)
    if "rate_limiter" not in ss:
        ss.rate_limiter = RateLimiter()
    if "provider" not in ss:
        ss.provider = "openai"
    if "api_key" not in ss:
        ss.api_key = ""
    if "rag_service" not in ss:
        ss.rag_service = None          # built lazily once key+provider are known
    if "profile" not in ss:
        ss.profile = None
    if "gap_tracker" not in ss:
        ss.gap_tracker = GapTracker()
    if "assistant" not in ss:
        ss.assistant = None
    if "chat_history" not in ss:
        ss.chat_history = []
    if "processed_files" not in ss:
        ss.processed_files = set()
    if "active_quiz" not in ss:
        ss.active_quiz = None
    if "quiz_submitted" not in ss:
        ss.quiz_submitted = False
    if "index_loaded" not in ss:
        ss.index_loaded = False
    if "socratic_mode" not in ss:
        ss.socratic_mode = False
    if "web_search_on" not in ss:
        ss.web_search_on = False
    if "study_plan" not in ss:
        ss.study_plan = None


_ensure_state()


def _rebuild_rag(provider: str, api_key: str):
    """(Re)create the RAGService when the provider changes."""
    st.session_state.rag_service = RAGService(
        provider=provider,
        api_key=api_key,
        session_id=st.session_state.session_id,
        rate_limiter=st.session_state.rate_limiter,
    )
    st.session_state.processed_files = set()
    st.session_state.index_loaded = False
    st.session_state.assistant = None


def _rebuild_assistant(provider: str, api_key: str):
    if st.session_state.profile is None or st.session_state.rag_service is None:
        return
    st.session_state.assistant = TeachingAssistant(
        rag=st.session_state.rag_service,
        profile=st.session_state.profile,
        api_key=api_key,
        provider=provider,
        rate_limiter=st.session_state.rate_limiter,
    )


def _render_source(s: dict) -> None:
    """Render one source row. Web results get a clickable link + 🌐 tag."""
    if s.get("web") and s.get("url"):
        title = s.get("title") or s["source"]
        st.markdown(f"🌐 **[{s['id']}] [{title}]({s['url']})**")
    else:
        page = f" p.{s['page']}" if s.get("page") else ""
        st.markdown(f"**[{s['id']}] {s['source']}**{page}")
    if s.get("snippet"):
        st.caption(s["snippet"])


def _render_usage(status: dict) -> None:
    """Small sidebar widget showing session usage against caps."""
    minute = f"{status['llm_minute']}/{status['llm_minute_cap']}"
    session = f"{status['llm_session']}/{status['llm_session_cap']}"
    mb = status['uploaded_bytes'] / (1024 * 1024)
    max_mb = status['max_total_bytes'] / (1024 * 1024)
    st.caption(
        f"Requests: **{minute}/min** · session **{session}** · "
        f"uploads **{mb:.1f}/{max_mb:.0f} MB**"
    )


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Provider selector
    provider_name = st.selectbox(
        "LLM Provider",
        list(PROVIDERS.keys()),
        index=list(p["value"] for p in PROVIDERS.values()).index(
            st.session_state.provider
        ),
    )
    pinfo = PROVIDERS[provider_name]
    provider = pinfo["value"]

    # Reinitialise RAG when provider changes
    if provider != st.session_state.provider:
        st.session_state.provider = provider
        st.session_state.rag_service = None
        st.session_state.index_loaded = False
        st.session_state.assistant = None

    # --- API key input. Pre-fill from env ONLY in dev mode (single user).
    default_key = os.getenv(pinfo["env"], "") if _USE_ENV_KEYS else ""
    api_key = st.text_input(
        pinfo["label"],
        type="password",
        value=st.session_state.api_key or default_key,
        help=pinfo["help"],
    )
    # Persist in session state but NEVER in os.environ — that would leak to
    # other concurrent sessions running in the same process.
    st.session_state.api_key = api_key

    # Provider-specific hint shown below the key field
    if provider == "gemini":
        st.caption("🆓 Free tier available — [Get key](https://aistudio.google.com)")
    elif provider == "openai":
        st.caption("🔑 [Get key at platform.openai.com](https://platform.openai.com)")

    if api_key and st.session_state.rag_service is None:
        try:
            _rebuild_rag(provider, api_key)
        except Exception as e:
            log.exception("could not init embeddings: %s", e)
            st.error(f"Could not initialise embeddings: {e}")

    # Try to load a persisted index once the key is set
    if api_key and st.session_state.rag_service and not st.session_state.index_loaded:
        try:
            if st.session_state.rag_service.load():
                st.session_state.processed_files = set(
                    st.session_state.rag_service.processed_files
                )
                st.success(
                    f"Loaded persisted index ({len(st.session_state.processed_files)} files)"
                )
            st.session_state.index_loaded = True
        except Exception as e:
            log.warning("could not load persisted index: %s", e)
            st.warning(f"Could not load persisted index: {e}")

    st.divider()
    st.header("📁 Materials")
    max_mb = st.session_state.rate_limiter.max_file_bytes / (1024 * 1024)
    st.caption(f"Max **{max_mb:.0f} MB** per file · PDF or TXT")
    uploaded_files = st.file_uploader(
        "Upload course materials (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if api_key and st.session_state.rag_service:
        # Add any newly-uploaded files
        current_names = {f.name for f in (uploaded_files or [])}
        new_files = [
            f for f in (uploaded_files or [])
            if f.name not in st.session_state.processed_files
        ]
        if new_files:
            with st.spinner("Processing new materials..."):
                try:
                    st.session_state.rag_service.process_files(new_files)
                    for f in new_files:
                        st.session_state.processed_files.add(f.name)
                    if st.session_state.profile:
                        _rebuild_assistant(provider, api_key)
                    st.success(f"Processed & persisted {len(new_files)} file(s).")
                except RateLimitError as e:
                    # Show the rate-limit reason to the user verbatim — the
                    # messages are crafted to be user-readable.
                    st.error(str(e))
                except Exception as e:
                    log.exception("process_files failed: %s", e)
                    st.error(f"Error processing files: {e}")

        # Detect files that were removed from the uploader and drop their
        # chunks from the FAISS index so the assistant no longer cites them.
        if uploaded_files is not None:
            removed = [
                name
                for name in list(st.session_state.processed_files)
                if name not in current_names
            ]
            for name in removed:
                try:
                    st.session_state.rag_service.remove_file(name)
                    st.session_state.processed_files.discard(name)
                except Exception as e:
                    log.warning("remove_file failed for %s: %s", name, e)
                    st.warning(f"Could not remove {name}: {e}")
            if removed:
                st.info(f"Removed {len(removed)} file(s) from the index.")
                if st.session_state.profile:
                    _rebuild_assistant(provider, api_key)
    elif not api_key:
        st.warning("Enter your API key to proceed.")

    # Full wipe — useful when the persisted index is stale or the user wants
    # to start over. Removes this session's vector store from disk.
    if st.session_state.rag_service and st.session_state.processed_files:
        if st.button("🗑️ Clear index", help="Delete all persisted chunks"):
            try:
                st.session_state.rag_service.clear_index()
                st.session_state.processed_files = set()
                st.session_state.assistant = None
                st.session_state.chat_history = []
                st.session_state.active_quiz = None
                st.success("Index cleared. Upload files to rebuild.")
                st.rerun()
            except Exception as e:
                log.exception("clear_index failed: %s", e)
                st.error(f"Failed to clear index: {e}")

    st.divider()
    st.markdown("### 📊 Session usage")
    _render_usage(st.session_state.rate_limiter.status())

    if st.session_state.profile:
        st.divider()
        p = st.session_state.profile
        t = st.session_state.gap_tracker
        st.markdown("### 🧑 Student Profile")
        st.caption(f"{p.education_level} · {p.familiarity}")
        if p.learning_goal:
            st.caption(f"Goal: {p.learning_goal}")
        st.markdown(
            f"**Quiz difficulty:** {t.difficulty}/5 — {t.difficulty_label()}"
        )
        weak = t.weak_topics()
        if weak:
            st.markdown("**Weak topics:** " + ", ".join(weak[:5]))
        if st.button("Reset profile"):
            st.session_state.profile = None
            st.session_state.assistant = None
            st.session_state.gap_tracker = GapTracker()
            st.rerun()


# -------------------------------------------------------------------- header
st.title("🎓 AI Teaching Assistant")
st.caption("Grounded in your course materials. Adaptive to you.")
st.markdown("---")


# ------------------------------------------------------------ profile gate
if st.session_state.profile is None:
    st.subheader("👋 Quick interview")
    st.write("Tell us about yourself so explanations fit your level.")
    with st.form("profile_form"):
        edu = st.selectbox("Education level", EDUCATION_LEVELS, index=1)
        fam = st.selectbox("Subject familiarity", FAMILIARITY_LEVELS, index=0)
        goal = st.text_input(
            "What's your learning goal? (e.g. 'ace my midterm on data structures')"
        )
        submitted = st.form_submit_button("Start")
        if submitted:
            if not api_key:
                st.warning("Enter your API key in the sidebar first.")
            else:
                profile = StudentProfile(
                    education_level=edu, familiarity=fam, learning_goal=goal
                )
                st.session_state.profile = profile
                st.session_state.gap_tracker.set_baseline(profile.baseline_difficulty())
                _rebuild_assistant(provider, api_key)
                st.rerun()
    st.stop()


# -------------------------------------------------- content gate: need materials
if st.session_state.rag_service is None or not st.session_state.rag_service.has_index():
    st.info("Upload at least one course file in the sidebar to enable the assistant.")
    st.stop()

if st.session_state.assistant is None:
    _rebuild_assistant(provider, api_key)


tab_qa, tab_grade, tab_quiz, tab_plan = st.tabs(
    ["💬 Q&A", "📝 Rubric Grading", "🎯 Adaptive Quiz", "🗓️ Study Plan"]
)


# =========================================================================== QA
with tab_qa:
    st.subheader("Q&A with Assistant")
    # Toggles: Socratic coaching and optional DuckDuckGo web search. Both
    # default to off so existing flows behave exactly as before.
    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state.socratic_mode = st.toggle(
            "🧭 Socratic mode",
            value=st.session_state.socratic_mode,
            help="Guide the student with questions instead of giving the answer directly.",
        )
    with col_b:
        st.session_state.web_search_on = st.toggle(
            "🌐 Web search",
            value=st.session_state.web_search_on,
            help="Also search the web (DuckDuckGo) and mix results into the context.",
        )

    # Surface persistent web-search failures so users understand why a
    # "web search" toggle is on but no web sources appeared.
    if st.session_state.web_search_on and web_search.last_error:
        st.warning(
            f"🌐 Web search is currently unavailable "
            f"({web_search.last_error}). Answers will use course materials only."
        )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        _render_source(s)
            if msg.get("verified") is False:
                st.warning("⚠️ Answer could not be fully verified against sources.")

    suggestion = st.session_state.gap_tracker.suggestion()
    if suggestion:
        st.info(suggestion)

    if prompt := st.chat_input("Ask a question about the course..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Safety first: crisis content is intercepted before any retrieval /
        # LLM call. We reply with professional resources instead of grading
        # or answering the question.
        if detect_crisis(prompt):
            with st.chat_message("assistant"):
                st.error("Safety notice")
                st.markdown(CRISIS_RESPONSE)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": CRISIS_RESPONSE,
                    "sources": [],
                    "verified": True,
                }
            )
            st.rerun()

        if detect_injection(prompt):
            reply = REFUSAL_MESSAGE
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply, "sources": [], "verified": True}
            )
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Pass prior turns so follow-ups like "can you provide
                    # more information?" keep the conversation context.
                    prior_history = st.session_state.chat_history[:-1]
                    result = st.session_state.assistant.answer_question(
                        prompt,
                        chat_history=prior_history,
                        socratic=st.session_state.socratic_mode,
                        use_web=st.session_state.web_search_on,
                    )
                    st.markdown(result["answer"])
                    if result.get("sources"):
                        with st.expander(f"📚 Sources ({len(result['sources'])})"):
                            for s in result["sources"]:
                                _render_source(s)
                    if not result.get("verified", True):
                        st.warning(
                            "⚠️ Answer could not be fully verified against sources."
                        )
                    # Only record a topic if one meaningful non-stopword term
                    # could be extracted — avoids "what", "can", etc.
                    topic = result.get("topic") or extract_topic(prompt)
                    if topic:
                        st.session_state.gap_tracker.record_question_asked(topic)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", []),
                            "verified": result.get("verified", True),
                        }
                    )
                    st.rerun()
                except RateLimitError as e:
                    st.error(str(e))
                except Exception as e:
                    log.exception("answer_question failed: %s", e)
                    st.error(f"An error occurred: {e}")


# ====================================================================== Grading
with tab_grade:
    st.subheader("📝 Rubric-based Grading")
    grade_q = st.text_input("Question", key="grade_q")
    grade_a = st.text_area("Student's answer", key="grade_a")
    st.markdown("**Rubric criteria** (one per line)")
    default_rubric = (
        "Answer is factually correct\n"
        "Includes a concrete example\n"
        "Mentions time or space complexity where relevant"
    )
    rubric_text = st.text_area("Rubric", value=default_rubric, height=120)

    if st.button("Grade answer"):
        if not (grade_q and grade_a):
            st.warning("Provide both a question and an answer.")
        # Safety intercept before spinning up the grading LLM call.
        elif detect_crisis(grade_q) or detect_crisis(grade_a):
            st.error("Safety notice")
            st.markdown(CRISIS_RESPONSE)
        else:
            criteria = [c for c in rubric_text.splitlines() if c.strip()]
            with st.spinner("Grading..."):
                result = None
                try:
                    result = st.session_state.assistant.grade_with_rubric(
                        grade_q, grade_a, criteria
                    )
                except RateLimitError as e:
                    st.error(str(e))
                except Exception as e:
                    log.exception("grade_with_rubric failed: %s", e)
                    st.error(f"An error occurred: {e}")
                if result and result.get("crisis"):
                    st.error("Safety notice")
                    st.markdown(result.get("feedback", CRISIS_RESPONSE))
                elif result is not None:
                    st.metric("Overall score", f"{result['score']}/100")
                    st.markdown("#### Breakdown")
                    for row in result["breakdown"]:
                        icon = "✅" if row.get("met") else "❌"
                        st.markdown(
                            f"{icon} **{row.get('criterion','?')}** — "
                            f"{row.get('sub_score', 0)}/100"
                        )
                        st.caption(row.get("justification", ""))
                    st.markdown("#### Final feedback")
                    st.write(result["feedback"])
                    # Feed grade into gap tracker using a meaningful topic
                    # extracted from the question (skip common stopwords).
                    topic = extract_topic(grade_q)
                    if topic:
                        st.session_state.gap_tracker.record_result(
                            topic, result["score"] >= 70
                        )
                    if result.get("sources"):
                        with st.expander("📚 Sources used"):
                            for s in result["sources"]:
                                _render_source(s)


# ========================================================================= Quiz
with tab_quiz:
    st.subheader("🎯 Adaptive Quiz")
    tracker = st.session_state.gap_tracker
    st.caption(
        f"Current difficulty: **{tracker.difficulty}/5** "
        f"({tracker.difficulty_label()})"
    )
    weak = tracker.weak_topics()
    if weak:
        st.info("Focusing on weak topics: " + ", ".join(weak[:5]))

    num_q = st.slider("Number of questions", 1, 5, 3)
    focus_input = st.text_input(
        "Optional focus topics (comma-separated)", value=", ".join(weak[:3])
    )

    if st.button("Generate quiz"):
        focus = [t.strip() for t in focus_input.split(",") if t.strip()] or None
        with st.spinner("Generating..."):
            try:
                quiz = st.session_state.assistant.generate_quiz(
                    num_questions=num_q, tracker=tracker, focus_topics=focus
                )
                if not quiz["questions"]:
                    st.error("Could not generate a quiz from the materials.")
                else:
                    st.session_state.active_quiz = quiz
                    st.session_state.quiz_submitted = False
            except RateLimitError as e:
                st.error(str(e))
            except Exception as e:
                log.exception("generate_quiz failed: %s", e)
                st.error(f"An error occurred: {e}")

    quiz = st.session_state.active_quiz
    if quiz:
        st.markdown(
            f"### Quiz — difficulty {quiz['difficulty']}/5 · {quiz['difficulty_label']}"
        )
        with st.form("quiz_form"):
            answers = {}
            for i, q in enumerate(quiz["questions"], 1):
                st.markdown(f"**Q{i} ({q.get('topic','')}):** {q['question']}")
                opts = q.get("options", {})
                choice = st.radio(
                    f"Your answer (Q{i})",
                    options=list(opts.keys()),
                    format_func=lambda k, opts=opts: f"{k}. {opts[k]}",
                    key=f"quiz_q_{i}",
                    index=None,
                )
                answers[i] = choice
            submit = st.form_submit_button("Submit quiz")
            if submit:
                correct_count = 0
                for i, q in enumerate(quiz["questions"], 1):
                    chosen = answers.get(i)
                    is_correct = chosen is not None and chosen == q.get("answer")
                    st.session_state.gap_tracker.record_result(
                        q.get("topic", "general"), is_correct
                    )
                    if is_correct:
                        correct_count += 1
                    icon = "✅" if is_correct else "❌"
                    st.markdown(
                        f"{icon} **Q{i}** — correct answer: **{q.get('answer')}**. "
                        f"{q.get('explanation','')}"
                    )
                st.success(
                    f"Score: {correct_count}/{len(quiz['questions'])}. "
                    f"New difficulty: {st.session_state.gap_tracker.difficulty}/5."
                )
                suggestion = st.session_state.gap_tracker.suggestion()
                if suggestion:
                    st.info(suggestion)
                st.session_state.quiz_submitted = True

        if quiz.get("sources"):
            with st.expander("📚 Sources used to generate this quiz"):
                for s in quiz["sources"]:
                    _render_source(s)


# =================================================================== Study plan
with tab_plan:
    st.subheader("🗓️ Personalized Study Plan")
    tracker = st.session_state.gap_tracker
    weak = tracker.weak_topics()
    if weak:
        st.info("Focus: " + ", ".join(weak[:5]))
    else:
        st.caption("No weak topics tracked yet — the plan will cover general material.")

    num_days = st.slider("Plan length (days)", 3, 7, 5)
    if st.button("Generate Study Plan"):
        with st.spinner("Building your plan..."):
            try:
                st.session_state.study_plan = (
                    st.session_state.assistant.generate_study_plan(
                        tracker=tracker, num_days=num_days
                    )
                )
            except RateLimitError as e:
                st.error(str(e))
            except Exception as e:
                log.exception("generate_study_plan failed: %s", e)
                st.error(f"An error occurred: {e}")

    plan = st.session_state.study_plan
    if plan and plan.get("days"):
        st.caption(
            f"Difficulty baseline: {plan['difficulty']}/5 · "
            f"{plan['difficulty_label']}"
        )
        for day in plan["days"]:
            with st.container(border=True):
                st.markdown(
                    f"### Day {day.get('day','?')} — {day.get('topic','(topic)')}"
                )
                st.markdown(f"**Goal:** {day.get('goal','')}")
                st.markdown(f"**Practice:** {day.get('practice','')}")
        if plan.get("sources"):
            with st.expander("📚 Sources used to build this plan"):
                for s in plan["sources"]:
                    _render_source(s)
    elif plan is not None:
        st.warning("Could not build a plan from the current materials.")


st.markdown("---")
st.caption(
    "Grounded QA · Rubric grading · Adaptive quizzing · Study plan · "
    "Socratic mode · Injection-guarded · Rate-limited."
)
