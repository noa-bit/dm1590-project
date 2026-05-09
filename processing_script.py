"""
Drum Sample Classifier
======================
Classifies drum sample filenames/strings into:
  - kick
  - snare
  - hi-hat
  - percussion

Uses two complementary strategies:
  1. Zero-shot NLP classification via facebook/bart-large-mnli (primary)
  2. Embedding cosine-similarity via sentence-transformers (fallback / ensemble)

Both run on PyTorch.  GPU is used automatically if available.

Install dependencies:
    pip install torch transformers sentence-transformers
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import torch
torch.set_num_threads(1)
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CATEGORIES = ["kick", "snare", "hi-hat", "percussion"]

# Rich textual descriptions fed to the zero-shot model so it understands the
# musical domain, not just the raw label words.
HYPOTHESIS_TEMPLATES = {
    "kick":       "This is a kick drum, bass drum, or low-frequency drum hit.",
    "snare":      "This is a snare drum, rim shot, side stick, or back-beat drum hit.",
    "hi-hat":     "This is a hi-hat, closed hat, open hat, pedal hat, or cymbal choke.",
    "percussion": (
        "This is a percussion sound such as a clap, clave, cowbell, conga, bongo, "
        "tom, floor tom, tambourine, shaker, rim, crash cymbal, ride cymbal, "
        "woodblock, cabasa, agogo, or any other non-kick/snare/hi-hat hit."
    ),
}

# Anchor phrases used by the embedding similarity fallback.
CATEGORY_ANCHORS: dict[str, list[str]] = {
    "kick": [
        "kick drum", "bass drum", "bd", "kick", "808 kick", "sub kick",
        "punch kick", "low kick", "acoustic kick",
    ],
    "snare": [
        "snare drum", "snare", "rim shot", "rimshot", "side stick",
        "sd", "back beat", "snare hit", "cracking snare", "tight snare",
    ],
    "hi-hat": [
        "hi hat", "hihat", "hi-hat", "closed hat", "open hat",
        "pedal hat", "chh", "ohh", "phh", "cymbal choke", "tight hat",
    ],
    "percussion": [
        "clap", "clave", "cowbell", "conga", "bongo", "tom", "floor tom",
        "tambourine", "shaker", "rim", "crash", "ride", "woodblock",
        "cabasa", "agogo", "triangle", "cymbal", "hand percussion",
        "timbale", "djembe", "cajon",
    ],
}

ZERO_SHOT_MODEL  = "facebook/bart-large-mnli"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
CONFIDENCE_FLOOR = 0.40   # below this, fall back to embedding similarity


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PRE-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

_STRIP_EXT = re.compile(r"\.[a-zA-Z0-9]{1,5}$")
_SEPARATORS = re.compile(r"[_\-\./\\|]+")
_DIGITS_ALONE = re.compile(r"\b\d+\b")


def preprocess(raw: str) -> str:
    """
    Turn a raw filename / sample string into a clean, readable phrase that
    an NLP model can reason about.

    Examples
    --------
    'Kick_Drum_808_v2.wav'  →  'Kick Drum 808 v2'
    'SNARE-Tight-01.aiff'   →  'SNARE Tight 01'
    'chh_16th.wav'          →  'chh 16th'
    """
    text = _STRIP_EXT.sub("", raw)          # drop file extension
    text = _SEPARATORS.sub(" ", text)       # normalise separators
    text = text.strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MODELS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DrumClassifier:
    """
    Wraps both the zero-shot NLI pipeline and the embedding similarity model.
    """
    device: int = field(default_factory=lambda: 0 if torch.cuda.is_available() else -1)

    # lazily initialised
    _zs_pipeline: object = field(default=None, init=False, repr=False)
    _embed_model:  object = field(default=None, init=False, repr=False)
    _anchor_embeddings: dict = field(default_factory=dict, init=False, repr=False)

    # ── loaders ──────────────────────────────────────────────────────────────

    def _load_zero_shot(self) -> None:
        if self._zs_pipeline is None:
            print(f"  Loading zero-shot model '{ZERO_SHOT_MODEL}' …")
            self._zs_pipeline = pipeline(
                "zero-shot-classification",
                model=ZERO_SHOT_MODEL,
                device=self.device,
            )

    def _load_embedder(self) -> None:
        if self._embed_model is None:
            print(f"  Loading embedding model '{EMBEDDING_MODEL}' …")
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
            # Pre-compute anchor embeddings (mean-pooled per category)
            for cat, phrases in CATEGORY_ANCHORS.items():
                embs = self._embed_model.encode(phrases, convert_to_tensor=True)
                self._anchor_embeddings[cat] = embs.mean(dim=0)

    # ── inference ────────────────────────────────────────────────────────────

    def _zero_shot_classify(self, text: str) -> tuple[str, float]:
        """Returns (category, confidence) using the NLI zero-shot model."""
        result = self._zs_pipeline(
            text,
            candidate_labels=list(HYPOTHESIS_TEMPLATES.values()),
            hypothesis_template="{}",
            multi_label=False,
        )
        # Map the winning hypothesis text back to a category label
        winning_hypothesis = result["labels"][0]
        confidence         = result["scores"][0]
        label_map = {v: k for k, v in HYPOTHESIS_TEMPLATES.items()}
        category  = label_map[winning_hypothesis]
        return category, confidence

    def _embedding_classify(self, text: str) -> tuple[str, float]:
        """Returns (category, similarity_score) via cosine similarity."""
        query_emb = self._embed_model.encode(text, convert_to_tensor=True)
        best_cat, best_score = "percussion", -1.0
        for cat, anchor_emb in self._anchor_embeddings.items():
            score = float(util.cos_sim(query_emb, anchor_emb))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    # ── public API ───────────────────────────────────────────────────────────

    def classify(self, sample: str) -> "Result":
        """Classify a single drum sample string."""
        clean = preprocess(sample)

        # ── primary: zero-shot NLI ──
        self._load_zero_shot()
        zs_cat, zs_conf = self._zero_shot_classify(clean)

        # ── fallback: embedding similarity ──
        self._load_embedder()
        emb_cat, emb_score = self._embedding_classify(clean)

        # Decision logic
        if zs_conf >= CONFIDENCE_FLOOR:
            # Trust zero-shot if confident
            final_cat   = zs_cat
            method_used = "zero-shot NLI"
            confidence  = zs_conf
        elif emb_score > 0.35:
            # Embedding similarity is more reliable for short/abbreviated names
            final_cat   = emb_cat
            method_used = "embedding similarity"
            confidence  = emb_score
        else:
            # Ensemble: pick whichever of the two agrees; else default to zs
            final_cat   = zs_cat if zs_cat == emb_cat else zs_cat
            method_used = "ensemble"
            confidence  = (zs_conf + emb_score) / 2

        return Result(
            raw=sample,
            cleaned=clean,
            category=final_cat,
            confidence=confidence,
            method=method_used,
            zs_category=zs_cat,
            zs_confidence=zs_conf,
            emb_category=emb_cat,
            emb_similarity=emb_score,
        )

    def classify_batch(self, samples: list[str]) -> list["Result"]:
        """Classify a list of drum sample strings."""
        self._load_zero_shot()
        self._load_embedder()
        return [self.classify(s) for s in samples]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  RESULT DATACLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    raw:            str
    cleaned:        str
    category:       str
    confidence:     float
    method:         str
    zs_category:    str
    zs_confidence:  float
    emb_category:   str
    emb_similarity: float

    def __str__(self) -> str:
        return (
            f"  [{self.category.upper():10s}]  {self.raw!r:<45s}"
            f"  conf={self.confidence:.2f}  via={self.method}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SAMPLE DATA
# ──────────────────────────────────────────────────────────────────────────────

EXAMPLE_SAMPLES: list[str] = [
    # Kicks
    "Kick_808_Deep.wav",
    "BD_Tight_v3.wav",
    "bass_drum_acoustic_01.aif",
    "SubKick-Heavy.wav",
    "kick_punchy_hard.wav",

    # Snares
    "Snare_Crack_Dry.wav",
    "SD_RimShot_01.wav",
    "snare-tight-vintage.aiff",
    "SideStick_Lo.wav",
    "Clack_Snare_08.wav",

    # Hi-hats
    "CHH_16th_closed.wav",
    "OpenHat_Sizzle.wav",
    "HiHat_Pedal_v2.wav",
    "hi_hat_tight.wav",
    "OHH_Jazz_Sweep.wav",

    # Percussion (everything else)
    "Clap_Analog_808.wav",
    "Conga_High_Slap.wav",
    "Cowbell_Metallic.wav",
    "Tambourine_shake_01.wav",
    "CrashCymbal_Room.wav",
    "RideCymbal_Dry.wav",
    "Tom_Floor_Heavy.wav",
    "Clave_Wooden.wav",
    "Shaker_Plastic_02.wav",
    "Rimshot_dry.wav",

    # Tricky / ambiguous names
    "HH_Open_Crunch.wav",
    "Perc_FX_001.wav",
    "909_BD.wav",
    "606_SD.wav",
    "chh.wav",
    "ohh.wav",
    "KK01.wav",
    "SN_02.wav",
]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def print_results(results: list[Result]) -> None:
    # Group by category
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


def classify_samples(
    samples: list[str],
    verbose: bool = False,
) -> list[Result]:
    """
    Public entry-point.  Pass your own list of sample strings.
    Returns a list of Result objects.
    """
    device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"\nDrum Sample Classifier  |  device: {device_label}")
    print(f"Classifying {len(samples)} sample(s) …\n")

    classifier = DrumClassifier()
    results    = classifier.classify_batch(samples)

    if verbose:
        print("\nDetailed per-sample breakdown:")
        for r in results:
            print(textwrap.dedent(f"""
              raw        : {r.raw}
              cleaned    : {r.cleaned}
              → category : {r.category}  (conf={r.confidence:.3f}, via {r.method})
                zero-shot : {r.zs_category} ({r.zs_confidence:.3f})
                embedding : {r.emb_category} ({r.emb_similarity:.3f})
            """))

    print_results(results)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── swap in your own sample list here ────────────────────────────────────
    my_samples = EXAMPLE_SAMPLES

    results = classify_samples(my_samples, verbose=False)

    # ── programmatic access example ──────────────────────────────────────────
    kicks = [r for r in results if r.category == "kick"]
    print(f"Kick samples found: {[r.raw for r in kicks]}")