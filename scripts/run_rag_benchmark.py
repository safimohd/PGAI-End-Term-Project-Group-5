"""
PGAI Group 5 — Phase 2 RAG Benchmark Runner
---------------------------------------------
Calls the local RAG API for all 50 questions
Saves responses with retrieval info to CSV after every question
Run from: rag-skeleton directory
Ensure: uvicorn is running on http://localhost:7860
"""

import requests
import pandas as pd
import json
import time
import os

import re

def fix_response(val):
    if not isinstance(val, str):
        return val
    # Fix encoding
    try:
        val = val.encode('latin-1').decode('utf-8')
    except:
        pass
    # Fix special characters
    val = val.replace('\u202f', ' ')
    val = val.replace('\xa0', ' ')
    val = val.replace('\xad', '-')
    val = val.replace('\u2011', '-')
    val = val.replace('【', '[').replace('】', ']')
    # Strip markdown formatting
    val = re.sub(r'\*\*(.*?)\*\*', r'\1', val)
    val = re.sub(r'\*(.*?)\*', r'\1', val)
    return val

# ============================================================
# CONFIGURATION
# ============================================================
RAG_API_URL     = "http://localhost:7860"
QUESTIONS_PATH  = r"C:\Users\DELL\OneDrive - Indian Institute of Management\Desktop\Term 4\P&GAI\End Term Project\Questions_PGAI_Group_5_FINAL.xlsx"   # update path if needed
INPUT_SHEET     = "All Questions"
OUTPUT_CSV      = "rag_phase2_responses.csv"

# RAG parameters (as validated in your test)
EMBEDDING_MODEL = "MiniLM-L6 (fast, 384d)"
ANSWER_MODEL    = "gpt-oss-20b (Together AI)"
TOP_K           = 4
HYBRID_WEIGHT   = 0.5

DELAY_SECONDS   = 2   # pause between calls

# ============================================================
# STEP 1: Verify RAG server is running
# ============================================================
print("🔍 Checking RAG server...")
try:
    r = requests.get(f"{RAG_API_URL}/answer-models", timeout=5)
    models = r.json()
    print(f"✅ Server running. Available models: {list(models['models'])}")
    if not models["available"].get(ANSWER_MODEL):
        print(f"❌ {ANSWER_MODEL} not configured — check your .env file")
        exit(1)
    print(f"✅ {ANSWER_MODEL} is ready\n")
except Exception as e:
    print(f"❌ Server not reachable: {e}")
    print("   Make sure uvicorn is running: python -m uvicorn backend.main:app --port 7860")
    exit(1)

# ============================================================
# STEP 2: Load questions
# ============================================================
print(f"📂 Loading questions from: {QUESTIONS_PATH}")
try:
    df = pd.read_excel(QUESTIONS_PATH, sheet_name=INPUT_SHEET)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["question_text"].notna()].reset_index(drop=True)
    total = len(df)
    print(f"✅ {total} questions loaded\n")
except Exception as e:
    print(f"❌ Error loading questions: {e}")
    exit(1)

# ============================================================
# STEP 3: Resume from checkpoint if exists
# ============================================================
if os.path.exists(OUTPUT_CSV):
    existing  = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig')
    done_ids  = set(existing["question_id"].tolist())
    results   = existing.to_dict("records")
    print(f"♻️  Resuming — {len(done_ids)}/{total} already done\n")
else:
    done_ids = set()
    results  = []
    print(f"🚀 Fresh start — running {total} questions\n")

# ============================================================
# STEP 4: Run all questions through RAG
# ============================================================
start_time = time.time()

for i, row in df.iterrows():
    q_id   = row.get("question_id", f"Q{i+1}")
    tier   = row.get("tier", "")
    q_text = str(row["question_text"]).strip()
    exp    = str(row.get("expected_answer", "")).strip()

    if q_id in done_ids:
        print(f"[{i+1}/{total}] {q_id} ⏭ skipped")
        continue

    print(f"[{i+1}/{total}] {q_id} | Tier {tier}")

    payload = {
        "question":        q_text,
        "embedding_model": EMBEDDING_MODEL,
        "answer_model":    ANSWER_MODEL,
        "top_k":           TOP_K,
        "hybrid_weight":   HYBRID_WEIGHT,
    }

    try:
        response = requests.post(
            f"{RAG_API_URL}/query",
            json=payload,
            timeout=60
        )
        data = response.json()

        rag_answer      = fix_response(data.get("answer", ""))
        retrieval_ms    = round(data.get("retrieval_latency_ms", 0), 1)
        generation_ms   = round(data.get("generation_latency_ms", 0), 1)

        # Extract retrieved chunk info
        chunks = data.get("chunks", [])
        top_sources = "; ".join([
            f"{c.get('filename','?')} (score:{c.get('score',0):.3f})"
            for c in chunks[:TOP_K]
        ]) if chunks else ""
        top_score = chunks[0].get("score", 0) if chunks else 0

        status = "OK" if rag_answer else "EMPTY"
        print(f"   ✅ Retrieved {len(chunks)} chunks | Top: {chunks[0].get('filename','?') if chunks else 'none'} ({top_score:.3f})")
        print(f"   ⏱ Retrieval: {retrieval_ms}ms | Generation: {generation_ms}ms")

    except Exception as e:
        rag_answer    = ""
        retrieval_ms  = 0
        generation_ms = 0
        top_sources   = ""
        top_score     = 0
        status        = f"ERROR: {str(e)[:100]}"
        print(f"   ❌ {status}")

    results.append({
        "question_id":        q_id,
        "tier":               tier,
        "question_text":      q_text,
        "expected_answer":    exp,
        "rag_response":       rag_answer,
        "top_sources":        top_sources,
        "top_similarity":     round(top_score, 3),
        "retrieval_ms":       retrieval_ms,
        "generation_ms":      generation_ms,
        "chunks_retrieved":   len(chunks) if status == "OK" else 0,
        "status":             status,
        # Scoring columns — fill after evaluation
        "E1_score":           "",
        "E2_score":           "",
        "E3_score":           "",
        "final_avg":          "",
    })

    # Save after every question
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    elapsed   = time.time() - start_time
    remaining = (elapsed / len(results)) * (total - len(results))
    print(f"   💾 Saved | ⏱ {elapsed/60:.1f}m elapsed | ~{remaining/60:.1f}m remaining\n")

    time.sleep(DELAY_SECONDS)

# ============================================================
# STEP 5: Summary
# ============================================================
results_df = pd.DataFrame(results)
ok    = (results_df["status"] == "OK").sum()
empty = (results_df["status"] == "EMPTY").sum()
err   = total - ok - empty


print("=" * 55)
print("  PHASE 2 RAG RUN COMPLETE")
print("=" * 55)
print(f"  ✅ Successful  : {ok}/{total}")
print(f"  ⚠️  Empty      : {empty}")
print(f"  ❌ Errors      : {err}")
print(f"\n  Avg retrieval time : {results_df['retrieval_ms'].mean():.0f}ms")
print(f"  Avg generation time: {results_df['generation_ms'].mean():.0f}ms")
print(f"\n  Output: {OUTPUT_CSV}")
print(f"\nNext step: Run evaluation script on rag_phase2_responses.csv")
print("=" * 55)

