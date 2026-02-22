"""
Quran ASR — Full Evaluation Script  (6,236 verses × N reciters)
================================================================
Designed to run on Kaggle (free T4 GPU, 30 hrs/week) or any GPU machine.

Features
--------
• Downloads the full Quran text (Tanzil Uthmani) automatically via alquran.cloud API
• Downloads MP3s from EveryAyah.com in parallel (polite rate limiting)
• Transcribes in GPU batches for maximum throughput (~300-500 verses/min on T4)
• Atomic checkpoints after every batch — safe to kill and resume at any time
• Evaluates multiple reciters back to back using the same loaded model
• Reports corpus-level WER and CER (industry standard — not per-verse average)
• Saves a full JSON report + a summary CSV you can open in Excel/Sheets

Time estimates on Kaggle free T4
----------------------------------
  Startup check     :  instant (ffmpeg / pydub verified before anything runs)
  Model load        :  ~3 min  (one time, reused across all reciters)
  Per reciter       :  ~20-35 min  (download + transcribe 6,236 verses)
  5 reciters total  :  ~2-3 hours

Usage
-----
    # Full run — all 5 reciters
    python evaluate_full.py

    # Single reciter
    python evaluate_full.py --reciters Abdul_Basit_Murattal_64kbps

    # Multiple specific reciters
    python evaluate_full.py --reciters Alafasy_128kbps Husary_128kbps

    # Resume an interrupted run (uses existing checkpoints automatically)
    python evaluate_full.py

    # Lower batch size if you get GPU out-of-memory errors
    python evaluate_full.py --batch-size 8

Install
-------
    pip install requests nemo_toolkit[asr] torch pydub tqdm
    sudo apt install -y ffmpeg libsndfile1   # both required — ffmpeg for MP3, libsndfile for WAV reading in NeMo

Output files
------------
    checkpoints/          — per-verse results per reciter (safe to delete after final report)
    audio/                — downloaded MP3s (kept so resume does not re-download)
    results/
      full_report.json    — every verse, every reciter: reference, hypothesis, WER, CER
      summary.csv         — one row per reciter: corpus WER, CER, verse counts

⚠️  IMPORTANT — DATA CONTAMINATION WARNING
-------------------------------------------
The model (nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) was trained on
the TarteelAI/EveryAyah dataset (390h), which is audio downloaded from
EveryAyah.com — the same source this script uses.

What this means for your evaluation:
  - WER will be LOWER than it would be on truly unseen data because the model
    has already learned these specific audio recordings.
  - NVIDIA's own reported WER of 6.55% (RNNT) / 7.61% (CTC) on EveryAyah
    is from a held-out TEST SPLIT of that dataset, not from the full site.
  - Your results will be comparable to NVIDIA's numbers, but do not mistake
    them for performance on real-world unseen recitation.

For a truly unbiased baseline before fine-tuning, you should also evaluate
against data the model has never seen (e.g. your own recordings, or a
different Arabic speech dataset not in the training set).

This script is still valuable because:
  1. It tells you the model's starting WER on the EveryAyah domain specifically.
  2. After fine-tuning on your data, running the same script shows whether
     fine-tuning improved or degraded EveryAyah performance.
  3. NVIDIA's 6.55% gives you a concrete comparison point.
"""

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING  — writes to both console and file
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME       = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
SAMPLE_RATE      = 16000
BATCH_SIZE       = 16          # verses per GPU batch — conservative default for T4
                               # increase to 32 if you have plenty of VRAM
DOWNLOAD_WORKERS = 8           # parallel MP3 download threads
DOWNLOAD_DELAY   = 0.15        # seconds between requests PER WORKER
                               # 8 workers x 0.15s = max ~53 req/s — polite
MAX_RETRIES      = 4           # retry attempts per failed download
EXPECTED_VERSES  = 6236        # validated against after Quran text download

EVERYAYAH_BASE = "https://everyayah.com/data"
TANZIL_API     = "https://api.alquran.cloud/v1/quran/quran-uthmani"

ALL_RECITERS = [
    "Abdul_Basit_Murattal_64kbps",
    "Alafasy_128kbps",
    "Husary_128kbps",
    "Mohammad_al_Tablaway_128kbps",
    "Minshawy_Murattal_128kbps",
]

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR    = Path("results")
AUDIO_DIR      = Path("audio")
QURAN_CACHE    = Path("quran_uthmani.json")


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP CHECKS  — fail loudly before wasting time downloading or loading model
# ─────────────────────────────────────────────────────────────────────────────
def check_dependencies() -> None:
    """
    Verify ffmpeg, libsndfile, and pydub are available.
    Exits immediately with a clear error if not — before any download/model load.
    """
    errors = []

    if shutil.which("ffmpeg") is None:
        errors.append(
            "ffmpeg not found.\n"
            "    Install with:  sudo apt install -y ffmpeg"
        )

    # libsndfile — required by NeMo to read WAV files
    try:
        import soundfile  # noqa: F401
    except ImportError:
        errors.append(
            "soundfile not installed (needed by NeMo for WAV reading).\n"
            "    Install with:  pip install soundfile\n"
            "    Also ensure:   sudo apt install -y libsndfile1"
        )

    try:
        import pydub  # noqa: F401
    except ImportError:
        errors.append("pydub not installed.  Install with:  pip install pydub")

    if errors:
        log.error("Missing dependencies — fix these before running:\n")
        for e in errors:
            log.error(f"  x {e}\n")
        sys.exit(1)

    log.info("Dependencies OK (ffmpeg, libsndfile/soundfile, pydub)")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALISATION
# We strip diacritics AND punctuation before WER/CER comparison.
#
# WHY strip diacritics:
#   We test whether the model recognised the correct WORD, not whether it
#   reproduced the correct vowel markings.
#
# WHY strip punctuation:
#   This model (pcd = punctuation + diacritics) outputs Arabic punctuation:
#   ، (Arabic comma \u060C), ؟ (Arabic question mark \u061F), and . (period).
#   Tanzil Uthmani reference text contains none of these. Leaving punctuation
#   in would penalise the model for correctly transcribing words but also
#   adding punctuation — an unfair penalty unrelated to ASR accuracy.
#
# Diacritics range notes:
#   Original code had two separate ranges (\u06D6-\u06DC and \u06DF-\u06E4)
#   that missed:
#     \u06DD  Arabic End of Ayah ۝  — appears in Uthmani Quran text
#     \u06DE  Arabic Start of Rub El Hizb ۞ — appears in Uthmani Quran text
#     \u06E9  Arabic Place of Sajdah ۩  — appears in Uthmani Quran text
#   Fixed by merging into \u06D6-\u06E4 and \u06E7-\u06ED.
#
# Character normalization notes:
#   alef variants (إأآٱ) → bare alef (ا):
#     The pcd model and Tanzil may disagree on which alef variant to use for
#     hamzated alefs. Normalizing all to bare alef ensures these are not
#     penalised as word errors.
#
#   teh marbuta (ة) → heh (ه):
#     At end of words, some models output ه where text has ة. Normalizing
#     prevents spurious character errors on feminine nouns.
#
#   alef maqsura (ى) → yeh (ي):
#     Words like على، إلى، هدى end in ى (U+0649) in the Tanzil reference.
#     The pcd model may output ي (U+064A) in these positions. Both represent
#     the same sound in context. Normalizing avoids penalising the model for
#     a character variant that carries no phonemic distinction here.
#     Convention consistent with standard Arabic NLP benchmarks.
# ─────────────────────────────────────────────────────────────────────────────
_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06E4\u06E7-\u06ED\u0640]"
)

# Arabic and ASCII punctuation that the model may output but reference lacks.
# \u060C = ،   \u061B = ؛   \u061F = ؟   \u06D4 = ۔ (Urdu full stop)
# ASCII . , ; : ! ? also stripped for completeness.
_PUNCTUATION = re.compile(r"[\u060C\u061B\u061F\u06D4.,;:!?]")

def normalize(text: str) -> str:
    """
    Strip diacritics, Quranic annotation marks, and punctuation.
    Then normalise common Arabic character variants and whitespace.
    Applied identically to both reference and hypothesis.
    """
    text = _DIACRITICS.sub("", text)
    text = _PUNCTUATION.sub("", text)
    text = re.sub(r"[إأآٱ]", "ا", text)  # alef variants -> bare alef
    text = re.sub(r"ة",      "ه", text)  # teh marbuta -> heh
    text = re.sub(r"ى",      "ي", text)  # alef maqsura -> yeh
    text = re.sub(r"\s+",    " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# METRICS  — CORPUS LEVEL  (industry standard for ASR evaluation)
#
# WHY corpus-level, not per-verse average?
# -----------------------------------------
# Per-verse average WER treats a 3-word verse the same as a 20-word verse.
# A single wrong word in a 3-word verse inflates WER to 33%.
# A single wrong word in a 20-word verse is only 5%.
# Averaging those gives 19% — a misleading number that changes just by
# picking different verse lengths to test.
#
# Corpus-level WER = total edit distance across ALL words / total reference words.
# This is what all ASR papers, competitions, and evaluation toolkits report.
# ─────────────────────────────────────────────────────────────────────────────
def _edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein distance between two sequences. Memory: O(n)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        curr_row = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(
                    prev_row[j],      # deletion
                    curr_row[j - 1],  # insertion
                    prev_row[j - 1],  # substitution
                )
        prev_row = curr_row
    return prev_row[n]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """Returns (word_edit_distance, reference_word_count)."""
    r = normalize(ref).split()
    h = normalize(hyp).split()
    return _edit_distance(r, h), len(r)


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    """Returns (char_edit_distance, reference_char_count)."""
    r = list(normalize(ref))
    h = list(normalize(hyp))
    return _edit_distance(r, h), len(r)


def corpus_wer(results: list[dict]) -> float:
    """
    Corpus WER = sum(word_errors) / sum(ref_words) across ALL verses.
    Failed verses (download_failed, conversion_failed, transcription_failed)
    contribute their full reference word count as errors, so the metric
    honestly reflects what fraction of the corpus the model got right.
    An empty results list returns 1.0 (worst case).
    """
    if not results:
        return 1.0
    total_errors = sum(r["word_errors"] for r in results)
    total_words  = sum(r["ref_words"]   for r in results)
    return total_errors / total_words if total_words > 0 else 0.0


def corpus_cer(results: list[dict]) -> float:
    """
    Corpus CER = sum(char_errors) / sum(ref_chars) across ALL verses.
    Same rationale as corpus_wer — failed verses are included at 100% error.
    An empty results list returns 1.0 (worst case).
    """
    if not results:
        return 1.0
    total_errors = sum(r["char_errors"] for r in results)
    total_chars  = sum(r["ref_chars"]   for r in results)
    return total_errors / total_chars if total_chars > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# QURAN TEXT  — download once from alquran.cloud, validate, cache locally
# ─────────────────────────────────────────────────────────────────────────────
def load_quran_text() -> list[dict]:
    """
    Returns list of dicts:  [{"surah": 1, "verse": 1, "text": "..."}, ...]
    Fetches from alquran.cloud on first run, then reads from local cache.
    Hard-validates that exactly EXPECTED_VERSES (6,236) verses were received.
    """
    if QURAN_CACHE.exists():
        log.info(f"Loading Quran text from cache: {QURAN_CACHE}")
        with open(QURAN_CACHE, encoding="utf-8") as f:
            verses = json.load(f)
        if len(verses) != EXPECTED_VERSES:
            log.warning(
                f"Cached file has {len(verses)} verses (expected {EXPECTED_VERSES}). "
                f"Deleting and re-downloading."
            )
            QURAN_CACHE.unlink()
            return load_quran_text()
        log.info(f"  {len(verses)} verses loaded from cache")
        return verses

    log.info("Downloading full Quran text from alquran.cloud (one-time) ...")
    try:
        r = requests.get(TANZIL_API, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Failed to download Quran text: {e}")
        sys.exit(1)

    try:
        data   = r.json()
        surahs = data["data"]["surahs"]
    except (ValueError, KeyError) as e:
        log.error(
            f"Unexpected response from alquran.cloud — could not parse verse data: {e}\n"
            f"  Response preview: {r.text[:200]}"
        )
        sys.exit(1)

    verses = []
    for surah in surahs:
        surah_num = surah["number"]
        for ayah in surah["ayahs"]:
            verses.append({
                "surah": surah_num,
                "verse": ayah["numberInSurah"],
                "text":  ayah["text"],
            })

    # Hard validation — never cache partial data
    if len(verses) != EXPECTED_VERSES:
        log.error(
            f"API returned {len(verses)} verses, expected {EXPECTED_VERSES}. "
            f"Aborting — not caching incomplete data."
        )
        sys.exit(1)

    # Atomic write: write .tmp then rename — safe against mid-write interruption
    tmp = QURAN_CACHE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)
    tmp.replace(QURAN_CACHE)

    log.info(f"  Saved {len(verses)} verses -> {QURAN_CACHE}")
    return verses


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def ayah_url(reciter: str, surah: int, verse: int) -> str:
    return f"{EVERYAYAH_BASE}/{reciter}/{surah:03d}{verse:03d}.mp3"


def download_one(reciter: str, surah: int, verse: int, dest: Path) -> bool:
    """Download one MP3. Returns True if file is on disk and valid after the call."""
    if dest.exists() and dest.stat().st_size > 1000:
        return True  # already downloaded in a previous run

    url = ayah_url(reciter, surah, verse)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(DOWNLOAD_DELAY)
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 1000:
                # Atomic write — avoid leaving a partial file on disk
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(resp.content)
                tmp.replace(dest)
                return True
            if resp.status_code == 404:
                log.debug(f"  404 (file not on server): {url}")
                return False
            log.debug(f"  HTTP {resp.status_code} for {url} (attempt {attempt})")
        except requests.RequestException as e:
            log.debug(f"  Network error for {url}: {e} (attempt {attempt})")
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)  # exponential back-off: 2s, 4s, 8s

    log.warning(f"  Download failed after {MAX_RETRIES} attempts: {url}")
    return False


def download_all_parallel(
    reciter: str, verses: list[dict], audio_dir: Path
) -> dict[tuple[int, int], Optional[Path]]:
    """
    Download all MP3s in parallel using a thread pool.
    Returns {(surah, verse): Path} for success, {(surah, verse): None} for failure.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[int, int], Optional[Path]] = {}
    lock = threading.Lock()

    def _dl(v: dict) -> None:
        s, n = v["surah"], v["verse"]
        dest = audio_dir / f"{s:03d}{n:03d}.mp3"
        ok = download_one(reciter, s, n, dest)
        with lock:
            results[(s, n)] = dest if ok else None

    log.info(f"  Downloading {len(verses)} MP3s ({DOWNLOAD_WORKERS} workers) ...")
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(_dl, v) for v in verses]
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="  Downloading",
            unit="ayah",
            ncols=80,
        ):
            pass

    downloaded = sum(1 for v in results.values() if v is not None)
    log.info(f"  Downloaded: {downloaded}  Failed: {len(verses) - downloaded}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO CONVERSION  MP3 -> 16 kHz mono float32 numpy array
# ─────────────────────────────────────────────────────────────────────────────
def mp3_to_float32(path: Path) -> Optional[np.ndarray]:
    """
    Convert MP3 to 16 kHz mono float32 PCM.

    Uses pydub.get_array_of_samples() — this returns the correct array type
    for the actual sample_width of the decoded audio. We then normalise by
    the actual bit depth, not by an assumed 16-bit value.

    Using audio.raw_data + np.frombuffer(..., dtype=np.int16) is WRONG
    if the source is 8-bit or 32-bit — it silently misinterprets bytes.

    Returns None on any failure so the caller can record a clean error.
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(str(path))
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)

        # get_array_of_samples() honours audio.sample_width automatically
        samples = np.array(audio.get_array_of_samples())

        # Normalise to [-1.0, 1.0] using the actual bit depth
        max_val = float(1 << (audio.sample_width * 8 - 1))
        return samples.astype(np.float32) / max_val

    except Exception as e:
        log.warning(f"  Audio conversion failed for {path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WAV WRITER
# ─────────────────────────────────────────────────────────────────────────────
def save_wav(audio: np.ndarray, path: Path) -> None:
    """Write float32 PCM array as a 16-bit mono WAV file."""
    pcm = (audio * 32767.0).clip(-32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """
    Load the NeMo FastConformer hybrid model in float32.
    IMPORTANT: do NOT use half() or autocast — the server's own comments
    document that FP16 produces all-unknown-token output on real speech.
    """
    log.info(f"Loading model: {MODEL_NAME} ...")
    t0 = time.time()
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
        props = torch.cuda.get_device_properties(0)
        log.info(f"  GPU  : {props.name}")
        log.info(f"  VRAM : {props.total_memory / 1e9:.1f} GB")
        log.info("  dtype: float32 (FP16/autocast intentionally disabled)")
    else:
        log.warning("  No GPU — CPU inference is very slow (~10x slower than T4)")

    # ── RNNT decoder: disable CUDA graphs ─────────────────────────────────────
    # cuStreamGetCaptureInfo returns 5 values on this driver/CUDA combination,
    # but NeMo's binding unpacks 6 → "not enough values to unpack (expected 6,
    # got 5)".  Disabling cuda graphs restores the standard Python greedy path
    # with identical numeric results.  Fix copied from asr_server.py v4.9.
    #
    # Clean approach: use RNNTBPEDecodingConfig (official NeMo dataclass) so
    # change_decoding_strategy() merges it cleanly against the full schema —
    # no OmegaConf struct-key errors possible.
    # ──────────────────────────────────────────────────────────────────────────
    log.info("  Configuring RNNT decoder (greedy_batch, cuda-graphs off) ...")
    try:
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecodingConfig
        rnnt_cfg = RNNTBPEDecodingConfig(strategy="greedy_batch")
        rnnt_cfg.greedy.max_symbols            = 10
        rnnt_cfg.greedy.use_cuda_graph_decoder = False   # disables the crashing path
        model.change_decoding_strategy(decoder_type="rnnt", decoding_cfg=rnnt_cfg)
        log.info("  RNNT decoder configured (cuda-graphs disabled) ✓")
    except Exception as exc:
        log.error(f"  Failed to configure RNNT decoder: {exc}")
        raise   # hard fail — RNNT is required for accurate WER

    log.info(f"  Loaded in {time.time() - t0:.1f}s")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# BATCH TRANSCRIPTION
# ─────────────────────────────────────────────────────────────────────────────
def transcribe_batch(model, wav_paths: list[Path]) -> list[str]:
    """
    Transcribe a batch of WAV files with NeMo and return a plain list of strings.

    RNNT / Hybrid RNNT-CTC return behaviour (critical):
    ----------------------------------------------------
    EncDecHybridRNNTCTCBPEModel.transcribe() returns a TUPLE, not a plain list:
        (List[str|Hypothesis], Optional[List[str|Hypothesis]])
         ^--- RNNT greedy results    ^--- beam-search results (None if not used)

    Iterating directly over the tuple gives the inner list as element 0 and
    None as element 1 — completely wrong.  We must unwrap with results[0].

    Reference: NeMo API docs + EncDecHybridRNNTCTCBPEModel GitHub issue #9598:
    "if type(hypotheses) == tuple: hypotheses = hypotheses[0]"

    return_hypotheses=False  -> plain strings preferred; handled defensively
                                 in case a NeMo version still returns Hypothesis.
    verbose=False            -> suppresses NeMo's internal per-batch tqdm bar
                                 (we have our own outer progress bar).
    num_workers=4            -> DataLoader workers for parallel audio loading.

    Raises on failure — caller catches and records transcription_failed.
    """
    paths_str = [str(p) for p in wav_paths]
    with torch.no_grad():
        raw = model.transcribe(
            paths_str,
            batch_size=BATCH_SIZE,
            return_hypotheses=False,
            num_workers=4,
            verbose=False,
        )

    # Unwrap tuple returned by RNNT / Hybrid models:
    # transcribe() -> (greedy_list, beam_list_or_None)
    # We always want the greedy list (index 0).
    if isinstance(raw, tuple):
        results = raw[0]
    else:
        results = raw

    out = []
    for r in results:
        if isinstance(r, str):
            out.append(r)
        elif hasattr(r, "text"):
            # Hypothesis object returned despite return_hypotheses=False
            out.append(r.text)
        elif hasattr(r, "y_sequence"):
            # Older NeMo BeamSearch result — decode token IDs manually
            out.append(model.tokenizer.ids_to_text(r.y_sequence.tolist()))
        else:
            log.warning(f"  Unexpected transcription result type {type(r)} — using str()")
            out.append(str(r))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT  — atomic writes, one JSON file per reciter
# ─────────────────────────────────────────────────────────────────────────────
def checkpoint_path(reciter: str) -> Path:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    return CHECKPOINT_DIR / f"{reciter}.json"


def load_checkpoint(reciter: str) -> dict:
    """Returns {'surah:verse': result_dict}. Empty dict if no checkpoint exists."""
    p = checkpoint_path(reciter)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"  Checkpoint found: {len(data)} verses already processed")
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"  Corrupt checkpoint for {reciter}: {e} — starting fresh")
        return {}


def save_checkpoint(reciter: str, done: dict) -> None:
    """
    Atomic save: write to .tmp then rename.
    If the process is killed mid-write, the previous checkpoint is untouched.
    No indent — faster writes, smaller files (6,236 entries per reciter).
    """
    p   = checkpoint_path(reciter)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False)
    tmp.replace(p)


# ─────────────────────────────────────────────────────────────────────────────
# FAILED RESULT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _make_failed_result(v: dict, status: str) -> dict:
    """
    Placeholder result for a verse that could not be processed.
    word_errors is set to ref_words and char_errors to ref_chars
    (i.e. every token is wrong) so corpus_wer/cer includes this verse
    at 100% error rate and does not under-count the true error.
    """
    norm      = normalize(v["text"])
    ref_words = len(norm.split())
    ref_chars = len(norm)
    return {
        "surah":       v["surah"],
        "verse":       v["verse"],
        "reference":   v["text"],
        "hypothesis":  "",
        "word_errors": ref_words,
        "ref_words":   ref_words,
        "char_errors": ref_chars,
        "ref_chars":   ref_chars,
        "status":      status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ONE RECITER
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_reciter(reciter: str, verses: list[dict], model) -> list[dict]:
    log.info(f"\n{'='*65}")
    log.info(f"  Reciter : {reciter}")
    log.info(f"  Verses  : {len(verses)}")
    log.info(f"{'='*65}")

    done = load_checkpoint(reciter)

    # Only skip verses that were successfully transcribed ('ok').
    # Verses that previously failed (download_failed, conversion_failed,
    # transcription_failed) are retried on every resume — a transient network
    # error or a crashed batch should not permanently exclude a verse from the
    # final corpus WER, since those failures inflate the error count unfairly.
    remaining = [
        v for v in verses
        if done.get(f"{v['surah']}:{v['verse']}", {}).get("status") != "ok"
    ]
    ok_count_from_ckpt = sum(1 for r in done.values() if r.get("status") == "ok")
    log.info(
        f"  Remaining : {len(remaining)}  "
        f"(skipping {ok_count_from_ckpt} already-OK from checkpoint)"
    )

    if not remaining:
        log.info("  All verses already OK in checkpoint — skipping")
        return list(done.values())

    # Step 1: Download all remaining MP3s
    audio_dir = AUDIO_DIR / reciter
    audio_map = download_all_parallel(reciter, remaining, audio_dir)

    # Step 2: Convert MP3s -> WAV, then batch transcribe
    t_start = time.time()
    with tempfile.TemporaryDirectory(prefix="quran_eval_") as tmpdir:
        wav_dir = Path(tmpdir)

        # Convert all MP3s to WAV first
        prepared: list[tuple[dict, Path]] = []
        log.info("  Converting MP3 -> WAV ...")
        for v in tqdm(remaining, desc="  Converting", unit="ayah", ncols=80):
            key      = f"{v['surah']}:{v['verse']}"
            mp3_path = audio_map.get((v["surah"], v["verse"]))

            if mp3_path is None:
                done[key] = _make_failed_result(v, "download_failed")
                continue

            audio = mp3_to_float32(mp3_path)
            if audio is None:
                done[key] = _make_failed_result(v, "conversion_failed")
                continue

            wav_path = wav_dir / f"{v['surah']:03d}{v['verse']:03d}.wav"
            save_wav(audio, wav_path)
            prepared.append((v, wav_path))

        # Save download/conversion failures before transcription begins
        save_checkpoint(reciter, done)
        n_failed_before = len(remaining) - len(prepared)
        log.info(
            f"  Conversion complete: {len(prepared)} OK, "
            f"{n_failed_before} failed (download/conversion)"
        )

        # Batch transcription
        log.info(f"  Transcribing {len(prepared)} verses (batch_size={BATCH_SIZE}) ...")
        with tqdm(total=len(prepared), desc="  Transcribing", unit="ayah", ncols=80) as pbar:
            for i in range(0, len(prepared), BATCH_SIZE):
                batch        = prepared[i : i + BATCH_SIZE]
                batch_verses = [item[0] for item in batch]
                batch_wavs   = [item[1] for item in batch]

                try:
                    hypotheses = transcribe_batch(model, batch_wavs)
                except Exception as e:
                    # IMPORTANT: record as transcription_failed, NOT as empty
                    # string (which would falsely count as WER=0 / perfect).
                    batch_num = i // BATCH_SIZE + 1
                    log.error(f"  Batch {batch_num} failed: {e}")
                    for v in batch_verses:
                        key = f"{v['surah']}:{v['verse']}"
                        done[key] = _make_failed_result(v, "transcription_failed")
                    save_checkpoint(reciter, done)
                    pbar.update(len(batch))
                    continue

                for v, hyp in zip(batch_verses, hypotheses):
                    key    = f"{v['surah']}:{v['verse']}"
                    ref    = v["text"]
                    w_err, w_ref = word_errors(ref, hyp)
                    c_err, c_ref = char_errors(ref, hyp)
                    done[key] = {
                        "surah":       v["surah"],
                        "verse":       v["verse"],
                        "reference":   ref,
                        "hypothesis":  hyp,
                        "word_errors": w_err,
                        "ref_words":   w_ref,
                        "char_errors": c_err,
                        "ref_chars":   c_ref,
                        "status":      "ok",
                    }

                # Atomic checkpoint after every batch
                save_checkpoint(reciter, done)
                pbar.update(len(batch))

    elapsed    = time.time() - t_start
    ok_count   = sum(1 for r in done.values() if r["status"] == "ok")
    fail_count = len(done) - ok_count
    log.info(
        f"  Finished in {elapsed / 60:.1f} min — "
        f"{ok_count} OK, {fail_count} failed (out of {len(verses)} total)"
    )
    return list(done.values())


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def compute_summary(reciter: str, results: list[dict]) -> dict:
    ok     = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    return {
        "reciter":           reciter,
        "total_verses":      len(results),
        "ok_verses":         len(ok),
        "failed_verses":     len(failed),
        "corpus_wer":        round(corpus_wer(results), 4),
        "corpus_cer":        round(corpus_cer(results), 4),
        # Raw counts include failures (failures have word_errors == ref_words)
        # so you can verify: corpus_wer == total_word_errors / total_ref_words
        "total_word_errors": sum(r["word_errors"] for r in results),
        "total_ref_words":   sum(r["ref_words"]   for r in results),
        "total_char_errors": sum(r["char_errors"] for r in results),
        "total_ref_chars":   sum(r["ref_chars"]   for r in results),
    }


def save_reports(all_results: dict[str, list[dict]]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    summaries = []

    full_report = {
        "model":    MODEL_NAME,
        "date":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "note":     (
            "WER and CER are corpus-level (total errors / total reference tokens), "
            "not per-verse averages. Text normalised: diacritics stripped, "
            "Quranic annotation marks stripped, Arabic/ASCII punctuation stripped, "
            "alef variants unified, teh marbuta -> heh, alef maqsura -> yeh. "
            "WARNING: This model was trained on TarteelAI/EveryAyah (390h). "
            "NVIDIA reference WER: 6.55% (RNNT) / 7.61% (CTC) on EveryAyah test split."
        ),
        "reciters": {},
    }

    for reciter, results in all_results.items():
        s = compute_summary(reciter, results)
        summaries.append(s)
        full_report["reciters"][reciter] = {
            "summary": s,
            "verses":  results,
        }

    # Atomic JSON write
    json_path = RESULTS_DIR / "full_report.json"
    tmp       = json_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    tmp.replace(json_path)
    log.info(f"  Full report -> {json_path}")

    # Summary CSV
    csv_path   = RESULTS_DIR / "summary.csv"
    fieldnames = [
        "reciter", "corpus_wer", "corpus_cer",
        "ok_verses", "failed_verses", "total_verses",
        "total_word_errors", "total_ref_words",
        "total_char_errors", "total_ref_chars",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    log.info(f"  Summary CSV  -> {csv_path}")

    # Console table
    print(f"\n{'='*72}")
    print(f"  EVALUATION RESULTS — {MODEL_NAME}")
    print(f"  Date     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Metric   : corpus-level WER / CER (industry standard)")
    print(f"  NVIDIA   : 6.55% WER (RNNT) / 7.61% WER (CTC) on EveryAyah test split")
    print(f"  WARNING  : Model trained on EveryAyah data — WER is optimistically low")
    print(f"{'='*72}")
    print(f"  {'Reciter':<42} {'WER':>7}  {'CER':>7}  {'OK / Total':>12}")
    print(f"  {'-'*42} {'-'*7}  {'-'*7}  {'-'*12}")
    for s in summaries:
        ok_str = f"{s['ok_verses']} / {s['total_verses']}"
        print(
            f"  {s['reciter']:<42} "
            f"{s['corpus_wer']:>7.2%}  "
            f"{s['corpus_cer']:>7.2%}  "
            f"{ok_str:>12}"
        )
    print(f"{'='*72}")
    print(f"  Full JSON : {json_path}")
    print(f"  Summary   : {csv_path}")
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full Quran ASR baseline evaluation (6,236 verses x N reciters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available reciters:\n"
            + "\n".join(f"  {r}" for r in ALL_RECITERS)
        ),
    )
    p.add_argument(
        "--reciters",
        nargs="+",
        default=ALL_RECITERS,
        help="One or more reciter names (default: all). See list below.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Verses per GPU batch (default: {BATCH_SIZE}). Lower to 8 if OOM.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DOWNLOAD_WORKERS,
        help=f"Parallel download threads (default: {DOWNLOAD_WORKERS}).",
    )
    # When running inside Jupyter / Colab / Kaggle notebooks, sys.argv contains
    # the kernel launcher's own arguments (e.g. -f kernel-xxx.json) which
    # argparse doesn't recognise -> SystemExit: 2.
    # parse_known_args() silently ignores unrecognised arguments so the script
    # works identically whether launched from a terminal or a notebook cell.
    args, unknown = p.parse_known_args()
    if unknown:
        log.debug(f"Ignoring unrecognised args (likely Jupyter kernel args): {unknown}")
    return args


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    global BATCH_SIZE, DOWNLOAD_WORKERS
    BATCH_SIZE       = args.batch_size
    DOWNLOAD_WORKERS = args.workers

    # Validate reciter names before spending time on anything else
    unknown = [r for r in args.reciters if r not in ALL_RECITERS]
    if unknown:
        log.error(f"Unknown reciters: {unknown}")
        log.error(f"Valid options: {ALL_RECITERS}")
        sys.exit(1)

    # Verify ffmpeg, libsndfile, and pydub before any downloading or model loading
    check_dependencies()

    log.info("=" * 65)
    log.info("  Quran ASR Full Evaluation")
    log.info(f"  Model    : {MODEL_NAME}")
    log.info(f"  Reciters : {', '.join(args.reciters)}")
    log.info(f"  Batch    : {BATCH_SIZE} verses/GPU batch")
    log.info(f"  Workers  : {DOWNLOAD_WORKERS} download threads")
    log.info("=" * 65)
    log.warning(
        "DATA NOTE: This model was trained on TarteelAI/EveryAyah (390h). "
        "Testing against EveryAyah.com audio evaluates on seen training data. "
        "WER will be optimistically low. "
        "NVIDIA's reference WER: 6.55%% (RNNT) / 7.61%% (CTC) on EveryAyah test split."
    )

    # Step 1: Quran ground-truth text (download once, cached forever)
    verses = load_quran_text()
    log.info(f"Quran text ready: {len(verses)} verses")

    # Step 2: Load model once — reused across all reciters
    model = load_model()

    # Step 3: Evaluate each reciter
    all_results: dict[str, list[dict]] = {}
    run_start = time.time()
    for reciter in args.reciters:
        all_results[reciter] = evaluate_reciter(reciter, verses, model)

    log.info(f"\nTotal wall-clock time: {(time.time() - run_start) / 60:.1f} min")

    # Step 4: Final report
    save_reports(all_results)


if __name__ == "__main__":
    main()