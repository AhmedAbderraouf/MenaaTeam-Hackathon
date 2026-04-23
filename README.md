# AI Teaching Assistant

A multi-provider, RAG-grounded, safety-hardened Streamlit app that turns a pile of course PDFs or notes into an adaptive tutor. Answers are cited against source material, quizzes rescale as the student learns, and every turn is filtered through an injection + crisis-safety layer before it ever reaches the LLM.

Built as a production-minded study of how to wire a real LLM feature set — grounded QA, rubric grading, assessment, study planning, Socratic coaching — on top of a RAG backbone without handing users a model that hallucinates, leaks prompts, or costs you arbitrary money.

---

## What it does

Four tabs, one consistent per-session state (profile → gap tracker → vector store):

| Tab | Feature |
|---|---|
| 💬 **Q&A** | Grounded retrieve-then-generate with citation, follow-up support via chat history, optional **Socratic mode** (guides with questions instead of answering), optional **web search** (DuckDuckGo) mixed into context. |
| 📝 **Rubric Grading** | Grade a student answer against a free-form rubric. Returns per-criterion sub-scores with justifications and citations, plus a final 0–100. |
| 🎯 **Adaptive Quiz** | Generates MCQs at one of 5 difficulty tiers. Difficulty auto-adjusts based on recent answer history; can focus on weak topics. |
| 🗓️ **Study Plan** | 3–7 day personalized plan grounded in uploaded material, prioritizing weak topics and matching the student's familiarity level. |

## Options surfaced in the UI

- **LLM provider** — OpenAI (`gpt-4o`) or Google Gemini (`gemini-2.5-flash`), hot-swappable mid-session.
- **Per-user API keys** — each visitor pastes their own key into the sidebar. Keys live only in `st.session_state`, never in process env. There's a `TA_USE_ENV_KEYS=1` escape hatch for single-user dev.
- **File uploads** — PDF or TXT, up to 200 MB per file and 200 MB per session total, tunable via env.
- **Socratic toggle** — replaces direct answers with 1–2 guiding questions.
- **Web search toggle** — augments RAG context with DuckDuckGo results; mixed in with the same bracket-citation format as course chunks.
- **Profile** — education level (HS / undergrad / grad) × familiarity (beginner / intermediate / advanced) × free-text learning goal. Drives answer style and baseline quiz difficulty.
- **Per-session usage widget** — live LLM-per-minute, LLM-per-session, and MB-uploaded against caps.
- **Per-file index management** — remove a single file from the FAISS index (chunks get deleted, not just hidden) or wipe the entire index.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Streamlit UI (app.py)                  │
│  sidebar: provider + key + uploads │ tabs: QA / Grade / Quiz /  │
│                                      Plan                       │
└─────────────┬───────────────────────┬───────────────────────────┘
              │                       │
              ▼                       ▼
   ┌────────────────────┐   ┌────────────────────────┐
   │   Guardrails       │   │   RateLimiter           │
   │ (regex, pre-LLM):  │   │ (per-session):          │
   │  • crisis          │   │  • N calls/min          │
   │  • prompt-inj      │   │  • N calls/session      │
   │  • sanitize        │   │  • MB/file, MB total    │
   └──────────┬─────────┘   └────────────┬────────────┘
              │                          │
              ▼                          │
   ┌────────────────────────────────────────────────────────┐
   │                  TeachingAssistant                      │
   │  answer_question / grade_with_rubric / generate_quiz /  │
   │  generate_study_plan                                    │
   │                                                         │
   │  Pipeline (QA):                                         │
   │   1. guardrails.detect_crisis → CRISIS_RESPONSE         │
   │   2. guardrails.detect_injection → REFUSAL_MESSAGE      │
   │   3. correct_typos (spellcheck on retrieval query only) │
   │   4. _augment_query_with_history (chat-aware retrieval) │
   │   5. RAGService.retrieve (FAISS, k=4)                   │
   │   6. + optional web_search.search_web                   │
   │   7. format_context → LLM #1 (generator, T=0.2)         │
   │   8. LLM #2 (verifier, T=0.0) grounds-check             │
   │   9. if unverified → strict-mode regen → re-verify      │
   │   10. return answer + sources + verified flag + topic   │
   └─────────┬──────────────────┬─────────────┬──────────────┘
             │                  │             │
             ▼                  ▼             ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  RAGService  │  │ StudentProfile│ │  GapTracker  │
    │  (FAISS,     │  │ (edu × fam × │  │ (per-topic   │
    │   per-user,  │  │  goal →      │  │  correct/    │
    │   marker-    │  │  style guide)│  │  incorrect,  │
    │   verified)  │  │              │  │  adaptive    │
    └──────┬───────┘  └──────────────┘  │  difficulty) │
           │                             └──────────────┘
           ▼
    ┌──────────────┐
    │ vector_store │
    │  /<sid>/     │
    │   faiss_     │
    │   index/     │
    │   manifest   │
    │   .session_  │
    │   marker     │
    └──────────────┘
```

### Module map

| File | Responsibility |
|---|---|
| [`app.py`](app.py) | Streamlit UI, session wiring, tab layout, pre-LLM safety interception, state rebuilds when provider/profile/uploads change. |
| [`teaching_assistant.py`](teaching_assistant.py) | Provider factory (`build_llm`), the four public methods, two-pass **generate → verify → strict-regen** loop, JSON-extracting parsers for quiz / grade / plan. |
| [`rag_service.py`](rag_service.py) | FAISS wrapper. Per-session directories, per-file chunk-ID tracking (so individual files can be **removed**, not just re-indexed), marker file to make `allow_dangerous_deserialization=True` safe, back-compat for older manifests. |
| [`guardrails.py`](guardrails.py) | 15-pattern crisis detector with a caring, resource-rich response; 12-pattern injection detector with a `teaching/ta/tutor/professor` allow-list; input sanitizer that strips embedded `system:` / `assistant:` role markers. |
| [`rate_limiter.py`](rate_limiter.py) | Three independent caps per session: sliding-window requests/min, session-total requests, file + session upload MB. Refundable on file removal. |
| [`student_profile.py`](student_profile.py) | Dataclass mapping education × familiarity × goal → an LLM style-guidance snippet and a baseline quiz difficulty. |
| [`gap_tracker.py`](gap_tracker.py) | Weak-topic detection (topics with more errors than corrects, + topics repeatedly asked about) and adaptive 1–5 difficulty driven by the last 3 answers. Stopword list keeps `what` / `how` / `explain` out of the tracker. |
| [`web_search.py`](web_search.py) | DuckDuckGo client with 3-attempt exponential-backoff retry, library rename compat (`ddgs` / `duckduckgo_search`), and a module-level `last_error` the UI surfaces as a banner. |
| [`logging_setup.py`](logging_setup.py) | Idempotent logger factory with an env-controlled `TA_LOG_LEVEL`. |
| [`tests/`](tests/) | pytest suite — 54 tests covering guardrails, rate limiter, RAG (real FAISS round-trips), web search retries, and teaching-assistant safety dispatch with a mocked LLM. |

---

## What makes it non-trivial

This isn't a "stick LangChain on a Streamlit app" demo. A few things got carefully engineered:

### 1. Two-pass grounding verifier

Every non-Socratic answer goes through a second, temperature-0 LLM with a strict system prompt asking only: "is this answer fully supported by CONTEXT? JSON-only." If unverified, the generator is re-run in **strict mode** (`"only use sentences directly supported…"`). The UI flags answers that fail verification twice with a ⚠️ banner instead of silently trusting them. This is what the tests cover in `TestVerifier`.

### 2. Safe FAISS deserialization

FAISS persistence requires `allow_dangerous_deserialization=True` because it pickles. To keep that safe, the app writes a `.session_marker` file on save and **refuses to load** a directory whose marker is missing, unreadable, or belongs to another session. Cross-session isolation is verified by a test that ingests in session A then tries to open it as session B — `load()` returns `False`.

### 3. Per-file index surgery

Most RAG demos treat the index as write-once. This one tracks chunk IDs per filename and calls `FAISS.delete(ids)` when the user removes a file from the uploader, plus refunds the MB quota back to the rate limiter. If the last file is removed, the persist directory is wiped entirely. Load/save handles older manifests without `file_chunks` gracefully.

### 4. Defense-in-depth safety

Three layers run **before** any retrieval or LLM spend:

- **Crisis detection** — 15 regex patterns (kill self, better off dead, overdose, self-harm variants…). On hit, the user gets 988 / Samaritans / findahelpline.com resources and the grading/quiz/QA paths are short-circuited. The grader specifically checks both the question *and* the student answer so nobody gets a 0/100 for a self-harm message.
- **Prompt-injection detection** — 12 patterns covering "ignore previous instructions", "jailbreak", "developer mode", "reveal the system prompt", plus a lookahead that lets "act as a TA/tutor/professor" pass while blocking "act as a hacker". The refusal message is deliberately short and topic-specific.
- **Role-marker sanitization** — `sanitize_user_input` strips inline `system:` / `assistant:` tokens so a user can't impersonate the system role when their text gets concatenated into a prompt.

Both the injection refusal and the OOD refusal are grounded in the same principle: **if the course materials can't justify the answer, we refuse rather than hallucinate.**

### 5. Rate limits that actually map to cost

The `RateLimiter` caps three separate things: requests-per-minute (burst abuse), requests-per-session (cost ceiling), and MB uploaded (embedding-cost ceiling). Upload quota is refunded when files are removed. Limits are configurable per-deployment via env vars:

```
TA_LLM_RPM, TA_LLM_SESSION_CAP,
TA_MAX_FILE_BYTES, TA_MAX_TOTAL_BYTES,
TA_USE_ENV_KEYS, TA_VECTOR_ROOT, TA_LOG_LEVEL
```

### 6. Adaptive quiz difficulty

Difficulty starts at the profile's baseline (beginner=2, intermediate=3, advanced=4). After each answer, the last-3-answer window drives a ±1 adjustment. Weak topics are topics where errors > corrects **and** errors ≥ 2, plus topics repeatedly asked about. The quiz generator seeds its retrieval query with those weak topics so questions are targeted, not random.

### 7. History-aware retrieval

Follow-ups like "can you explain more?" are retrieval-dead on their own — no content words to match against FAISS. `_augment_query_with_history` appends the last user turn to the current query so the retriever still finds the right chunks, while the LLM prompt still receives the original question plus full history messages as context.

### 8. Provider parity

`build_llm` and `build_embeddings` are tiny factories that return LangChain objects for either OpenAI or Gemini. Everything downstream — the verifier, grading, quiz, plan — is provider-agnostic. Swapping providers in the UI rebuilds the RAG service and assistant lazily; the vector store and chat history persist across swaps until you clear them.

### 9. Real test coverage

`tests/` exercises production code with real FAISS round-trips and mocked LLMs:

- crisis + injection matrices (positive, negative, case-insensitive, benign false-positive traps)
- rate-limit sliding-window math + upload accounting
- RAG: ingest → persist → reload → retrieve, plus marker-mismatch rejection
- teaching assistant: crisis intercepted before LLM, injection intercepted before LLM, empty-index returns OOD refusal, verifier loop
- web search: retry-on-failure, no-library graceful degradation, empty-query short-circuit

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then, in the sidebar: pick a provider, paste your key, upload a PDF or TXT, fill the student-profile form, and start asking questions.

### Env vars

| Var | Default | Purpose |
|---|---|---|
| `TA_USE_ENV_KEYS` | `0` | If `1`, pre-fills the API-key field from env (single-user dev only). Off in multi-user deploys so one visitor's session can't inherit the operator's key. |
| `TA_VECTOR_ROOT` | `vector_store` | Where per-session FAISS indexes live. |
| `TA_LLM_RPM` | `20` | LLM calls allowed per minute per session. |
| `TA_LLM_SESSION_CAP` | `200` | Total LLM calls per session. |
| `TA_MAX_FILE_BYTES` | `200 MB` | Per-file upload cap. |
| `TA_MAX_TOTAL_BYTES` | `200 MB` | Per-session total upload cap. |
| `TA_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / … |

### Tests

```bash
pytest -v
```

---

## License & credits

Internal demo project. Lecture notes under [`materials/`](materials/) are hand-written sample content used for dev and tests.
