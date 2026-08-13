I have an existing Streamlit application called `intellicomply_app.py`
for my project IntelliComply — an RBI Compliance Assistant.

IMPORTANT:
DO NOT rebuild the application.
DO NOT redesign the application.
DO NOT remove, rename, or alter any existing tabs, benchmark results,
cached questions, routing logic, audit trail, session metrics, or styling.

I only want to improve the existing "Ask Compliance" tab so that the
demo has a clean, explicit LIVE QUESTION capability.

CURRENT ARCHITECTURE:

1. Cached/demo questions:
   - Stored in CACHE
   - Have expected answers
   - Have Base LLM / Basic RAG / Ensemble answers
   - Have risk level
   - Have tier
   - Have audit chunks
   - Use existing routing logic
   - T4 and CRITICAL questions can use Ensemble

2. Custom questions:
   - The app already has "Type my own question"
   - These currently call the `/query` endpoint
   - They currently use Basic RAG only
   - This is intentional because Ensemble takes much longer

The relevant existing logic is already present:

mode = st.radio("", ["Use demo question", "Type my own question"], ...)

and custom questions currently call:

call_rag_live(question)

with:

top_k=6
hybrid_weight=0.0

DO NOT change this backend behaviour.

===========================================================
OBJECTIVE
===========================================================

Add a polished "LIVE QUESTION" demo experience.

The professor specifically wants us to demonstrate that the system
can answer a question that is NOT already present in the cached demo
questions.

The live mode should therefore:

- Accept a question typed by the user
- Send it to the actual RAG API
- Retrieve from the RBI knowledge base
- Generate an answer using the existing answer model
- Display the retrieved RBI sources
- Display retrieval confidence
- Clearly indicate that this is LIVE BASIC RAG
- NOT invoke Ensemble
- NOT perform query decomposition
- NOT change the existing Ensemble implementation
- Remain fast enough for a classroom presentation

===========================================================
IMPORTANT DEMO DESIGN PRINCIPLE
===========================================================

We intentionally DO NOT run Ensemble on arbitrary live questions.

Reason:

Ensemble requires approximately 6 API calls per query and is
significantly slower.

Existing architecture already defines:

Basic RAG:
2 API calls
1 retrieval + 1 generation

Ensemble:
6 API calls
3 retrievals + decomposition + generation

Therefore:

CACHED / BENCHMARK QUESTIONS
→ Existing tier/risk router
→ Basic RAG or Ensemble as currently implemented

LIVE QUESTIONS
→ ALWAYS BASIC RAG
→ 2 API calls
→ Fast interactive demonstration

Do not change this logic.

===========================================================
UI CHANGE
===========================================================

Inside the existing "Ask Compliance" tab, keep:

"Use demo question"
"Type my own question"

Do not remove this radio selector.

When the user selects:

TYPE MY OWN QUESTION

show a clearly labelled live-demo area.

Use a heading:

"🔴 LIVE QUESTION"

and a short explanatory note:

"Test a question outside the benchmark using the live RBI knowledge
base. Live testing uses Basic RAG for fast response."

Do not use the word "production".

Add a text area:

Placeholder:

"Ask a short RBI compliance question...
e.g. What is the SMA-1 overdue period under RBI prudential norms?"

Keep the text area compact.

Below it add the existing primary button, but in custom-question
mode label it:

"⚡ Test Live Question"

Do not create a second competing button.

===========================================================
LIVE QUESTION SAFETY / SCOPE
===========================================================

Because this is intended for a live classroom demonstration,
add a small informational note:

"Best for factual or single-document RBI questions.
Complex multi-step questions are intentionally not routed to Ensemble
in Live Demo mode."

This makes the architectural trade-off explicit rather than making
the live mode look incomplete.

Do NOT block the user from entering a complex question.
Just display the note.

===========================================================
LIVE RESPONSE
===========================================================

When the user clicks:

"⚡ Test Live Question"

and the question is non-empty:

1. Check whether the RAG server is online using the existing
   `check_server()` function.

2. If offline:
   Show the existing error:
   "RAG server offline. Please start uvicorn to use live mode."

3. If online:
   Call the existing `call_rag_live(question)` function.

4. Keep:
   - answer
   - raw chunks
   - source filenames

5. Do NOT call:
   - get_routing_decision()
   - Ensemble
   - query decomposition
   - cached benchmark logic

6. Set:

method = "LIVE BASIC RAG"
api_calls = 2
calls_saved = 4

===========================================================
LIVE RESPONSE CARD
===========================================================

After the API returns, display a response card using the EXISTING
answer-card styling.

Header:

"⚡ Live IntelliComply Answer"

Beside it display a small badge:

"LIVE • BASIC RAG"

Do NOT display "Verified" for a live question.

This is important.

For cached benchmark questions, retain the existing "Verified" badge.

For live questions, use:

"Retrieved from RBI corpus"

or

"Live retrieval"

instead.

Reason:
We have not manually benchmarked or validated an arbitrary live
question against a gold-standard expected answer.

Therefore we should never imply that every live answer is benchmark-
verified.

===========================================================
ANSWER CARD CONTENT
===========================================================

Display:

1. Generated answer

2. Model / method:

"Basic RAG + gpt-oss-20b | 2 API calls"

3. Retrieved sources:

Show the filenames of the top retrieved chunks using the existing
citation chips.

Example:

📄 F1-Master Direction-KYC

📄 F5-Master Circular-Prudential Norms

4. Retrieval confidence

Keep the existing confidence card.

For live questions, if the API provides a meaningful retrieval score,
derive/display it appropriately.

If the current API does not expose a reliable aggregate confidence,
retain the current fallback value of 0.85.

DO NOT invent a more precise confidence score.

===========================================================
LIVE AUDIT PREVIEW
===========================================================

After the live answer, add a small expandable section:

"🔍 View Retrieved RBI Evidence"

Inside it, show the actual chunks returned by the live API.

For each chunk display:

- document filename
- similarity/retrieval score if available
- retrieved text

Reuse the existing audit-row styling.

Do not modify the existing Audit Trail tab.

This is only a compact evidence preview for the live question.

===========================================================
SESSION METRICS
===========================================================

Preserve the existing session metrics.

When a live question is answered:

increment:

session_questions

increment:

session_api_saved += 4

because Live Basic RAG uses 2 calls versus the 6-call
always-Ensemble baseline.

Do NOT attempt to calculate:

- regulatory exposure
- risk level
- fine exposure
- critical questions

for arbitrary live questions because those fields are only available
for cached benchmark questions.

Therefore:

For live questions:
Questions Asked → increment
API Calls Saved → +4
Regulatory Exposure → unchanged
Critical Risk → unchanged

Do not display a fake risk classification.

===========================================================
VERY IMPORTANT: PRESERVE EXISTING CACHED DEMO BEHAVIOUR
===========================================================

When:

"Use demo question"

is selected:

DO NOT CHANGE ANYTHING.

The existing behaviour must remain exactly as it is:

- Question selector
- [Recovered] labels
- Expected answer
- Tier
- Risk
- Routing Decision
- Basic RAG vs Ensemble selection
- API call count
- Business Impact
- Confidence
- Violation
- Fine reference
- Existing session metrics

All of this must continue working.

===========================================================
DO NOT CHANGE OTHER TABS
===========================================================

Do NOT modify:

1. Compare Methods
2. Audit Trail
3. Benchmark Results
4. How It Works

The existing Compare Methods tab must continue using cached
benchmark questions only.

The existing Audit Trail tab must continue using cached audit data.

The existing benchmark numbers must remain untouched.

The existing routing matrix must remain untouched.

The existing RISK_CONFIG must remain untouched.

The existing CACHE must remain untouched.

===========================================================
DO NOT CHANGE THE RAG API
===========================================================

Do not modify the backend API.

Continue using:

RAG_API_URL = "http://localhost:7860"

EMBEDDING_MODEL = "MiniLM-L6 (fast, 384d)"

ANSWER_MODEL = "gpt-oss-20b (Together AI)"

Continue calling:

POST /query

using the existing:

call_rag_live()

function.

Do not create a new endpoint.

===========================================================
OPTIONAL DEMO ENHANCEMENT
===========================================================

Add 3 small suggested-question buttons/chips below the live text box:

"Small Account limits"

"SMA-1 / SMA-2 trigger"

"DLG 5% cap"

When clicked, populate the text area with the corresponding question.

These are only suggestions.

They must STILL go through the live API.

IMPORTANT:
Do not use CACHE for these questions.

The point is to demonstrate that even a familiar RBI question can be
retrieved LIVE from the actual RBI knowledge base.

Use:

"Use as live test"

or similar subtle wording.

===========================================================
ERROR HANDLING
===========================================================

Handle:

- server offline
- timeout
- empty answer
- empty retrieved chunks
- API error

gracefully.

If no chunks are retrieved, do NOT display:

"Verified"

Instead display:

"⚠️ No supporting RBI evidence retrieved"

and make it clear that the response should not be treated as verified.

If the API fails, show a clean Streamlit error rather than a Python
traceback.

===========================================================
IMPORTANT PRODUCT PRINCIPLE
===========================================================

The application should communicate this architecture:

                 USER QUESTION
                       │
          ┌────────────┴────────────┐
          │                         │
     DEMO QUESTION             LIVE QUESTION
          │                         │
   Existing Router             Basic RAG
          │                         │
   ┌──────┴──────┐              2 API calls
   │             │
Basic RAG    Ensemble
2 calls       6 calls

Do NOT add this diagram to the UI unless necessary.

The behaviour should simply reflect this architecture.

===========================================================
FINAL CHECK
===========================================================

After making the modification, verify:

✓ Existing cached demo questions still work
✓ Existing recovered questions still route to Ensemble where required
✓ Existing Compare Methods tab unchanged
✓ Existing Audit Trail unchanged
✓ Existing Benchmark Results unchanged
✓ Existing How It Works unchanged
✓ Existing session metrics still work
✓ "Type my own question" now clearly looks like LIVE DEMO MODE
✓ Live questions actually hit the RAG API
✓ Live questions use Basic RAG only
✓ Live questions do NOT trigger Ensemble
✓ Live answer displays actual retrieved RBI sources
✓ Live answer does NOT falsely say "Verified"
✓ Live mode remains fast
✓ No existing features are removed
✓ No existing benchmark numbers are modified

Make the smallest possible code change necessary.
Do not refactor unrelated code.
Do not change the visual theme.
Do not change the existing cached-question behaviour.
