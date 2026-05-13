"""
Drum Sample Classifier
======================
Classifies drum sample filenames into: kick, snare, hi-hat, percussion

Strategy:
  1. Keyword regex matching (fast, no ML — covers ~90% of filenames)
  2. Embedding cosine-similarity via sentence-transformers (fallback only,
     batch-processed to avoid spawning extra worker processes)
"""

from __future__ import annotations

import os
import re
import textwrap
import warnings
import json
from dataclasses import dataclass
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

CATEGORIES = ["kick", "snare", "hi-hat", "percussion", "undefined"]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_FLOOR = 0.47  # below this, the name is too ambiguous to classify

# Keyword patterns — checked in order, first match wins
_RULES: list[tuple[str, re.Pattern]] = [
    ("kick", re.compile(
        r"\b(kick|bd|bass[\s_\-]?drum|bassdrum|sub[\s_\-]?kick|kik|bds"
        r"|BasDr|Base|BASS_DRUM|KCK)\b",
        re.IGNORECASE,
    )),

    ("snare", re.compile(
        # Full words with strict boundaries
        r"\b(snare|rim[\s_\-]?shot|rimshot|side[\s_\-]?stick|snr"
        r"|RIM|RIMS|Crosstie|SIDESTIK|GOODSN)\b"
        # 2-letter codes: right boundary only so they match when embedded ("shtsd" → sd at end)
        r"|sd(?![a-zA-Z])"
        r"|(?<![a-zA-Z])sn(?![a-zA-Z])",  # "sn" still needs left boundary (too short otherwise)
        re.IGNORECASE,
    )),

    ("hi-hat", re.compile(
        r"\b(hi[\s_\-]?hat|hihat|hat|chh|ohh|phh"
        r"|open[\s_\-]?hat|closed[\s_\-]?hat|pedal[\s_\-]?hat"
        r"|HHCL|808OPEN|HHOP|Short|Longhi)\b"
        r"|hh(?![a-zA-Z])",  # right boundary only: matches "pdhh", "chhh" etc.
        re.IGNORECASE,
    )),

    ("percussion", re.compile(
        r"\b(perc|conga|bongo|clap|cowbell|cow|Cowb|COW|Cowbell|Block"
        r"|TimHi|TimbLo|Timpani|Tympani|Tria|SHAKE|CHIME|CONG|Claves"
        r"|GUIRA|CG-H[\s_\-]?bell"
        r"|crash|ride|cymbal|cy|tom|floor[\s_\-]?tom|shaker|shakr"
        r"|tambourine|tamb|agogo|clave|woodblock|triangle|djembe"
        r"|cajon|timbale|cabasa|maracas|guiro|bell|gong|chime"
        r"|snap|handclap|clav|crsh|cym|rim(?!shot))\b"
        # Short abbreviations: right-boundary only
        r"|cla(?![a-zA-Z])"   # clap / clave abbreviation
        r"|per(?![a-zA-Z])",  # percussion abbreviation
        re.IGNORECASE,
    )),
    ("kick", re.compile(
        r"\b(kick|bd|bass[\s_\-]?drum|bassdrum|sub[\s_\-]?kick|kik|bds)\b",
        re.IGNORECASE,
    )),
    ("snare", re.compile(
        # Full words with strict boundaries
        r"\b(snare|rim[\s_\-]?shot|rimshot|side[\s_\-]?stick|snr)\b"
        # 2-letter codes: right boundary only so they match when embedded ("shtsd" → sd at end)
        r"|sd(?![a-zA-Z])"
        r"|(?<![a-zA-Z])sn(?![a-zA-Z])",  # "sn" still needs left boundary (too short otherwise)
        re.IGNORECASE,
    )),
    ("hi-hat", re.compile(
        r"\b(hi[\s_\-]?hat|hihat|hat|chh|ohh|phh"
        r"|open[\s_\-]?hat|closed[\s_\-]?hat|pedal[\s_\-]?hat)\b"
        r"|hh(?![a-zA-Z])",  # right boundary only: matches "pdhh", "chhh" etc.
        re.IGNORECASE,
    )),
    ("percussion", re.compile(
        r"\b(perc|conga|bongo|clap|cowbell|cow|Cowb|COW|Cowbell|Block|TimHi|TimbLo|Timpani|Tympani|Tria|SHAKE|CHIME|CONG|Claves|GUIRA|CG-H[\s_\-]?bell"
        r"|crash|ride|cymbal|cy|tom|floor[\s_\-]?tom|shaker|shakr|tambourine|tamb"
        r"|agogo|clave|woodblock|triangle|djembe|cajon|timbale"
        r"|cabasa|maracas|guiro|bell|gong|chime|snap|handclap"
        r"|clav|crsh|cym|rim(?!shot))\b"
        # Short abbreviations: right-boundary only (no left \b) so they match
        # even when stuck to other letters, e.g. "202cla01" → "202 cla 01"
        r"|cla(?![a-zA-Z])"   # clap / clave abbreviation
        r"|per(?![a-zA-Z])",  # percussion abbreviation
        re.IGNORECASE,
    )),
]

CATEGORY_ANCHORS: dict[str, list[str]] = {
    "kick":       ["kick drum", "bass drum", "bd", "808 kick", "sub kick", "low drum hit"],
    "snare":      ["snare drum", "rim shot", "side stick", "sd", "back beat snare"],
    "hi-hat":     ["hi hat", "closed hat", "open hat", "chh", "ohh", "cymbal choke"],
    "percussion": ["clap", "cowbell", "conga", "tom", "crash cymbal", "ride cymbal",
                   "shaker", "tambourine", "percussion sound"],
}

_STRIP_EXT    = re.compile(r"\.[a-zA-Z0-9]{1,5}$")
_SEPARATORS   = re.compile(r"[_\-\./\\|]+")
_CAMEL_1      = re.compile(r'([a-z0-9])([A-Z])')    # digit/lower → UPPER: "909Ride" → "909 Ride"
_CAMEL_2      = re.compile(r'([A-Z]+)([A-Z][a-z])') # ACRONYM→Title: "LTambourine" → "L Tambourine"
_LETTER_DIGIT = re.compile(r'([a-zA-Z])(\d)')        # letter → digit: "Perc095" → "Perc 095"
_DIGIT_ALPHA  = re.compile(r'(\d)([a-zA-Z])')        # digit → letter: "202snr" → "202 snr"

# Applied FIRST to isolate drum keywords buried in all-caps codes like "RTOMLENH" or "RAPCLAP2".
# cy(?=\d) is Roland's 2-letter cymbal prefix (e.g. "CY0000"); must be matched before
# _LETTER_DIGIT would split "CY0000" into "C Y 0000" and destroy it.
_CAPS_DRUM_SPLIT = re.compile(
    r'(kick|snare|clap|crash|cym(?:bal)?|ride|conga|bongo|tom|hihat|perc|shaker|tamb|cowbell|cy(?=\d))',
    re.IGNORECASE,
)


def preprocess(raw: str) -> str:
    text = _STRIP_EXT.sub("", raw)
    text = _CAPS_DRUM_SPLIT.sub(r' \1 ', text)  # isolate drum words before any splitting
    text = _CAMEL_1.sub(r'\1 \2', text)
    text = _CAMEL_2.sub(r'\1 \2', text)
    text = _LETTER_DIGIT.sub(r'\1 \2', text)
    text = _DIGIT_ALPHA.sub(r'\1 \2', text)
    text = _SEPARATORS.sub(" ", text)
    return text.strip()


def _keyword_classify(text: str) -> Optional[str]:
    # No alphabetic content (pure numbers/punctuation) → unclassifiable
    if not re.search(r'[a-zA-Z]', text):
        return "undefined"
    for category, pattern in _RULES:
        if pattern.search(text):
            return category
    return None  # ambiguous — try embedding


@dataclass
class Result:
    raw:        str
    cleaned:    str
    category:   str
    confidence: float
    method:     str

    def __str__(self) -> str:
        return (
            f"  [{self.category.upper():10s}]  {self.raw!r:<45s}"
            f"  conf={self.confidence:.2f}  via={self.method}"
        )


def classify_samples(samples: list[str], verbose: bool = False) -> list[Result]:
    """
    Public entry-point. Returns a list of Result objects, one per sample.
    """
    cleaned = [preprocess(s) for s in samples]
    results: list[Optional[Result]] = [None] * len(samples)
    ambiguous: list[int] = []

    # Pass 1: keyword matching — instant, no models
    for i, (raw, clean) in enumerate(zip(samples, cleaned)):
        cat = _keyword_classify(clean)
        if cat:
            results[i] = Result(raw=raw, cleaned=clean, category=cat,
                                confidence=1.0, method="keyword")
        else:
            ambiguous.append(i)

    keyword_count = len(samples) - len(ambiguous)
    print(f"  Keyword match: {keyword_count}/{len(samples)} samples")

    if ambiguous:
        print(f"  Embedding model for {len(ambiguous)} ambiguous filename(s)…")
        _embedding_classify(samples, cleaned, results, ambiguous)

    print_results(results)
    return results

def save_filenames_json(results: list[Result], path: str = "drum_classes.json") -> None:
    grouped = {cat: [] for cat in CATEGORIES}

    for r in results:
        grouped[r.category].append(r.raw)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)

    print(f"Saved JSON results to: {path}")

def _embedding_classify(
    samples: list[str],
    cleaned: list[str],
    results: list,
    indices: list[int],
) -> None:
    from sentence_transformers import SentenceTransformer, util

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer(EMBEDDING_MODEL)

    # Pre-compute anchor embeddings (one batch)
    anchor_embs: dict[str, object] = {}
    for cat, phrases in CATEGORY_ANCHORS.items():
        embs = model.encode(phrases, convert_to_tensor=True, show_progress_bar=False)
        anchor_embs[cat] = embs.mean(dim=0)

    # Encode all ambiguous samples in a single batch
    texts = [cleaned[i] for i in indices]
    query_embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    for idx, i in enumerate(indices):
        best_cat, best_score = "undefined", -1.0
        for cat, anchor_emb in anchor_embs.items():
            score = float(util.cos_sim(query_embs[idx], anchor_emb))
            if score > best_score:
                best_score, best_cat = score, cat
        if best_score < SIMILARITY_FLOOR:
            best_cat = "undefined"
        results[i] = Result(
            raw=samples[i], cleaned=cleaned[i],
            category=best_cat, confidence=best_score, method="embedding",
        )


def print_results(results: list[Result]) -> None:
    grouped: dict[str, list[Result]] = {cat: [] for cat in CATEGORIES}
    for r in results:
        grouped[r.category].append(r)

    print("\n" + "═" * 72)
    print("  DRUM SAMPLE CLASSIFICATION RESULTS")
    print("═" * 72)
    for cat in CATEGORIES:
        items = grouped[cat]
        print(f"\n  ▸ {cat.upper()} ({len(items)} samples)")
        print("  " + "─" * 68)
        for r in items:
            print(r)

    print("\n" + "═" * 72)
    totals = {c: len(v) for c, v in grouped.items()}
    print("  SUMMARY: " + "  |  ".join(f"{k}: {v}" for k, v in totals.items()))
    print("═" * 72 + "\n")


# ── Example / standalone run ──────────────────────────────────────────────────

EXAMPLE_SAMPLES: list[str] = [
    "Kick_808_Deep.wav", "BD_Tight_v3.wav", "bass_drum_acoustic_01.aif",
    "SubKick-Heavy.wav", "kick_punchy_hard.wav",
    "Snare_Crack_Dry.wav", "SD_RimShot_01.wav", "snare-tight-vintage.aiff",
    "SideStick_Lo.wav", "Clack_Snare_08.wav",
    "CHH_16th_closed.wav", "OpenHat_Sizzle.wav", "HiHat_Pedal_v2.wav",
    "hi_hat_tight.wav", "OHH_Jazz_Sweep.wav",
    "Clap_Analog_808.wav", "Conga_High_Slap.wav", "Cowbell_Metallic.wav",
    "Tambourine_shake_01.wav", "CrashCymbal_Room.wav", "Tom_Floor_Heavy.wav",
    "Perc_FX_001.wav", "909_BD.wav", "606_SD.wav", "chh.wav", "ohh.wav",
    "KK01.wav", "SN_02.wav",
]

if __name__ == "__main__":
    file_names = []
    with open("file_registry.txt") as file:
        for line in file.readlines():
            file_names.append(line.strip())
    
    for line in file_names:
        line = line.replace("\n", "")
    print(file_names)
    results = classify_samples(file_names, verbose=False)

    save_filenames_json(results)

   #kicks = [r for r in results if r.category == "kick"]
    #print(f"Kick samples found: {[r.raw for r in kicks]}")
