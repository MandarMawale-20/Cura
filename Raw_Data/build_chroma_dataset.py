"""
build_chroma_dataset.py - Converts 9 medical datasets into unified JSONL for ChromaDB.
"""

import ast
import json
import os
import random
import re
from collections import defaultdict
from typing import Any

import pandas as pd

# Cap on HealthCareMagic rows (~100k) to prevent dominating retrieval. None = keep all.
HEALTHCAREMAGIC_SAMPLE_SIZE: int | None = 3000
RANDOM_SEED: int = 42

# Boilerplate sign-offs in HealthCareMagic answers - stripped without LLM.
SIGNOFF_PATTERNS: list[str] = [
    r"chat\s*doctor\.?",
    r"i hope (it|this) helps?\.?",
    r"best wishes,?",
    r"regards,?",
    r"thank you for (posting|consulting|your query)[^.]*\.",
    r"feel free to (ask|contact|consult)[^.]*\.",
    r"consult (your |a )?(doctor|physician) (for|if|in case)[^.]*\.",
]


def clean_hcm_answer(text: str, max_len: int = 800) -> str:
    for pat in SIGNOFF_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        truncated = text[:max_len].rsplit(".", 1)[0]
        text = (truncated + ".") if truncated else text[:max_len]
    return text


# ----------------------------- CONFIG ---------------------------------------
# Edit paths to your actual files. Missing files are skipped with a warning.

FILES = {
    "medquad": "medquad.csv",
    "intent": "intents.json",
    "healthcaremagic": "HealthCareMagic-100k.json",
    "description": "description.csv",
    "diet": "diets.csv",
    "dis_symp_dict": "dis_symp_dict.txt",
    "og_dataset": "Original_Dataset.csv",
    "precaution": "precautions_df.csv",
    "workout": "workout_df.csv",
}

OUTPUT_FILE = "rag_dataset.jsonl"

# Build consolidated "disease profile" docs. Better for RAG than separate chunks.
ALSO_BUILD_DISEASE_PROFILES = True


# --------------------------- HELPERS ----------------------------------------

def make_record(_id: str, document: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"id": _id, "document": document.strip(), "metadata": metadata}


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def nz(row: pd.Series, col: str) -> str:
    val = row.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


# --------------------------- PARSERS ----------------------------------------

def parse_medquad(path: str) -> list[dict[str, Any]]:
    df = clean_cols(pd.read_csv(path))
    records = []
    for i, row in df.iterrows():
        q, a = nz(row, "question"), nz(row, "answer")
        if not q or not a:
            continue
        document = f"Question: {q}\nAnswer: {a}"
        metadata = {
            "source": "MedQuAD",
            "type": "qa",
            "focus_area": nz(row, "focus_area"),
            "original_source": nz(row, "source"),
        }
        records.append(make_record(f"medquad_{i+1:05d}", document, metadata))
    return records


def parse_intent(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for i, intent in enumerate(data.get("intents", []), start=1):
        tag = intent.get("tag", "")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        document = f"Topic: {tag}\nCommon patient questions: {'; '.join(patterns)}\nAnswer: {' '.join(responses)}"
        metadata = {"source": "Intent", "type": "intent", "tag": tag}
        records.append(make_record(f"intent_{i:05d}", document, metadata))
    return records


def parse_healthcaremagic(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if HEALTHCAREMAGIC_SAMPLE_SIZE is not None and len(data) > HEALTHCAREMAGIC_SAMPLE_SIZE:
        random.seed(RANDOM_SEED)
        data = random.sample(data, HEALTHCAREMAGIC_SAMPLE_SIZE)

    records = []
    for i, item in enumerate(data, start=1):
        q = (item.get("input") or "").strip()
        a = clean_hcm_answer((item.get("output") or "").strip())
        if not q or not a:
            continue
        document = f"Patient query: {q}\nDoctor response: {a}"
        metadata = {"source": "HealthCareMagic100k", "type": "qa"}
        records.append(make_record(f"hcm_{i:05d}", document, metadata))
    return records


def parse_description(path: str) -> list[dict[str, Any]]:
    df = clean_cols(pd.read_csv(path))
    records = []
    for i, row in df.iterrows():
        disease, desc = nz(row, "Disease"), nz(row, "Description")
        if not disease:
            continue
        document = f"Disease: {disease}\nDescription: {desc}"
        metadata = {"source": "description", "type": "description", "disease": disease}
        records.append(make_record(f"description_{i+1:05d}", document, metadata))
    return records


def parse_diet(path: str) -> list[dict[str, Any]]:
    df = clean_cols(pd.read_csv(path))
    diet_cols = [c for c in df.columns if c.lower().startswith("diet")]
    records = []
    for i, row in df.iterrows():
        disease = nz(row, "Disease")
        if not disease:
            continue
        diets = [nz(row, c) for c in diet_cols if nz(row, c)]
        document = f"Disease: {disease}\nRecommended diet: {', '.join(diets)}"
        metadata = {"source": "diet", "type": "diet", "disease": disease}
        records.append(make_record(f"diet_{i+1:05d}", document, metadata))
    return records


def parse_precaution(path: str) -> list[dict[str, Any]]:
    df = clean_cols(pd.read_csv(path))
    prec_cols = [c for c in df.columns if c.lower().startswith("precaution")]
    records = []
    for i, row in df.iterrows():
        disease = nz(row, "Disease")
        if not disease:
            continue
        precs = [nz(row, c) for c in prec_cols if nz(row, c)]
        document = f"Disease: {disease}\nPrecautions to take: {', '.join(precs)}"
        metadata = {"source": "precaution", "type": "precaution", "disease": disease}
        records.append(make_record(f"precaution_{i+1:05d}", document, metadata))
    return records


def parse_workout(path: str) -> list[dict[str, Any]]:
    df = clean_cols(pd.read_csv(path))
    if "Disease" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Disease"})
    workout_cols = [c for c in df.columns if c.lower().startswith("workout")]
    records = []
    for i, row in df.iterrows():
        disease = nz(row, "Disease")
        if not disease:
            continue
        workouts = [nz(row, c) for c in workout_cols if nz(row, c)]
        document = f"Disease: {disease}\nRecommended lifestyle/workout guidance: {', '.join(workouts)}"
        metadata = {"source": "workout", "type": "workout", "disease": disease}
        records.append(make_record(f"workout_{i+1:05d}", document, metadata))
    return records


def parse_dis_symp_dict(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            disease = str(items[0]).strip()
            symptoms = [str(s).strip() for s in items[1:] if str(s).strip()]
            document = f"Disease: {disease}\nSymptoms: {', '.join(symptoms)}"
            metadata = {"source": "dis_symp_dict", "type": "symptom_dict", "disease": disease}
            records.append(make_record(f"dissympdict_{i:05d}", document, metadata))
    return records


def parse_og_dataset(path: str) -> list[dict[str, Any]]:
    df = clean_cols(pd.read_csv(path))
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]

    disease_symptoms: dict[str, set[str]] = defaultdict(set)
    disease_patterns: dict[str, set[tuple[str, ...]]] = defaultdict(set)

    for _, row in df.iterrows():
        disease = nz(row, "Disease")
        if not disease:
            continue
        symptoms = tuple(sorted({nz(row, c) for c in symptom_cols if nz(row, c)}))
        if not symptoms:
            continue
        disease_symptoms[disease].update(symptoms)
        disease_patterns[disease].add(symptoms)

    records = []
    for i, disease in enumerate(sorted(disease_symptoms), start=1):
        all_symptoms = sorted(disease_symptoms[disease])
        document = f"Disease: {disease}\nCommonly reported symptoms: {', '.join(all_symptoms)}"
        metadata = {
            "source": "Og_Dataset",
            "type": "symptoms",
            "disease": disease,
            "num_symptom_patterns_seen": len(disease_patterns[disease]),
        }
        records.append(make_record(f"ogdataset_{i:05d}", document, metadata))
    return records


# ------------------- OPTIONAL: consolidated disease profiles ----------------

def build_disease_profiles(
    description_recs: list[dict[str, Any]],
    diet_recs: list[dict[str, Any]],
    precaution_recs: list[dict[str, Any]],
    workout_recs: list[dict[str, Any]],
    symptom_recs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def index_by_disease(records: list[dict[str, Any]]) -> dict[str, str]:
        out = {}
        for r in records:
            disease = r["metadata"]["disease"]
            body = r["document"].split("\n", 1)
            out[disease] = body[1] if len(body) > 1 else ""
        return out

    desc_map = index_by_disease(description_recs)
    diet_map = index_by_disease(diet_recs)
    prec_map = index_by_disease(precaution_recs)
    work_map = index_by_disease(workout_recs)
    symp_map = index_by_disease(symptom_recs)

    all_diseases = sorted(set(desc_map) | set(diet_map) | set(prec_map)
                           | set(work_map) | set(symp_map))

    records = []
    for i, disease in enumerate(all_diseases, start=1):
        parts = [
            f"What are the symptoms, precautions, diet, and treatment for {disease}?",
            f"Disease: {disease}",
        ]
        if disease in desc_map:
            parts.append(desc_map[disease])
        if disease in symp_map:
            parts.append(symp_map[disease])
        if disease in prec_map:
            parts.append(prec_map[disease])
        if disease in diet_map:
            parts.append(diet_map[disease])
        if disease in work_map:
            parts.append(work_map[disease])
        document = "\n".join(parts)
        metadata = {
            "source": "disease_profile",
            "type": "disease_profile",
            "disease": disease,
            "has_description": disease in desc_map,
            "has_symptoms": disease in symp_map,
            "has_precaution": disease in prec_map,
            "has_diet": disease in diet_map,
            "has_workout": disease in work_map,
        }
        records.append(make_record(f"diseaseprofile_{i:05d}", document, metadata))
    return records


# ------------------------------ MAIN ----------------------------------------

PARSERS: dict[str, callable] = {
    "medquad": parse_medquad,
    "intent": parse_intent,
    "healthcaremagic": parse_healthcaremagic,
    "description": parse_description,
    "diet": parse_diet,
    "dis_symp_dict": parse_dis_symp_dict,
    "og_dataset": parse_og_dataset,
    "precaution": parse_precaution,
    "workout": parse_workout,
}


def main() -> None:
    all_records: list[dict[str, Any]] = []
    parsed: dict[str, list[dict[str, Any]]] = {}

    for key, path in FILES.items():
        if not path or not os.path.exists(path):
            print(f"[skip] {key}: file not found at '{path}'")
            continue
        try:
            recs = PARSERS[key](path)
            parsed[key] = recs
            all_records.extend(recs)
            print(f"[ok]   {key}: {len(recs)} records from {path}")
        except Exception as e:
            print(f"[error] {key} ({path}): {e}")

    if ALSO_BUILD_DISEASE_PROFILES:
        needed = ["description", "diet", "precaution", "workout", "og_dataset"]
        if any(k in parsed for k in needed):
            profile_recs = build_disease_profiles(
                parsed.get("description", []),
                parsed.get("diet", []),
                parsed.get("precaution", []),
                parsed.get("workout", []),
                parsed.get("og_dataset", []),
            )
            all_records.extend(profile_recs)
            print(f"[ok]   disease_profiles: {len(profile_recs)} consolidated records")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    all_diseases = sorted({
        rec["metadata"]["disease"]
        for rec in all_records
        if "disease" in rec["metadata"]
    })
    with open("diseases.json", "w", encoding="utf-8") as f:
        json.dump(all_diseases, f, ensure_ascii=False, indent=2)

    print(f"\nTotal records written: {len(all_records)} -> {OUTPUT_FILE}")
    print(f"Known disease list written: {len(all_diseases)} -> diseases.json")


if __name__ == "__main__":
    main()