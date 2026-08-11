# IntelliComply — RBI Compliance AI

PGPBA Term 4 | P&GAI Group 5 | IIM Bangalore

Live Demo: https://pgai-end-term-project-group-5.streamlit.app

---

## What We Built

A compliance AI benchmark and RAG system for Indian Banking regulations.
Tested 3 LLMs on 50 RBI compliance questions, then built a RAG pipeline
on 15 RBI documents. Ran parameter sensitivity analysis and ensemble retrieval
to recover failing questions.



## Scripts

| File | Purpose |
|---|---|
| `run_rag_benchmark.py` | Run all 50 questions through RAG |
| `evaluate_rag.py` | Score RAG responses (E1/E2/E3 profiles) |
| `hybrid_sensitivity.py` | Test hybrid_weight = 0.0 / 0.5 / 1.0 |
| `topk_sensitivity.py` | Test Top-K = 2 / 4 / 6 / 8 |
| `query_decomposition.py` | Decompose T4 questions into sub-questions |
| `real_ensemble.py` | Ensemble on 12 failing questions |
| `ensemble_all50.py` | Ensemble on all 50 for generalisation |
| `intellicomply_app.py` | Streamlit demo app |

---

## Setup

Add API keys at the top of each script before running.
RAG server: `python -m uvicorn backend.main:app --port 7860`
Demo: `streamlit run intellicomply_app.py`

---

Group 5: Mohammed Safi Ur Rahman, Malladi Sai Anudeep, Vetcha Nikhilesh,
Banoth Tejender, Damala Susanth, Duppada Sai Kaushik
