"""
prepare_quran_manifests.py  v2.0  (bug-fixed)
==============================================
Converts HuggingFace Quran audio datasets into NeMo-format manifest files
for fine-tuning nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0.

Model: nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
Type : EncDecHybridRNNTCTCBPEModel  (FastConformer + RNNT/CTC hybrid)
Notes: This model outputs DIACRITIZED Arabic  (pcd = punctuation + diacritics).
       Manifests MUST therefore use diacritized transcriptions — keep harakat.

NeMo manifest format (one JSON object per line):
  {"audio_filepath": "/abs/path/audio.wav", "text": "بِسْمِ اللَّهِ", "duration": 3.5}

Datasets used (all free on HuggingFace):
  1. tarteel-ai/everyayah                  — 390 h, 70+ reciters, diacritized
  2. Salama1429/tarteel-ai-everyayah-Quran — 829 h extended
  3. Buraaq/quran-audio-text-dataset       — 30 reciters x 6236 ayahs
  4. RetaSy/quranic_audio_dataset          — crowdsourced real-world recordings

Install:
  pip install datasets soundfile librosa tqdm

Usage:
  # All datasets
  python prepare_quran_manifests.py --output_dir ./manifests --audio_dir ./quran_audio

  # Single dataset only
  python prepare_quran_manifests.py --datasets everyayah_ext --output_dir ./manifests

  # Dry-run (no download — just validates code paths)
  python prepare_quran_manifests.py --dry_run
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    raise ImportError("soundfile not installed.  Run: pip install soundfile")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arabic text normalisation
# ---------------------------------------------------------------------------
# The pcd model was trained WITH diacritics — DO NOT strip harakat.
# Only remove Quranic annotation marks that are not part of the Arabic text
# and are not in the model's SentencePiece vocabulary.

_STRIP_CHARS = re.compile(
    r"["
    r"\u0600-\u0605"   # Arabic number sign / annotation marks
    r"\u060B"          # Afghani sign
    r"\u0660-\u0669"   # Arabic-Indic digits (not in model vocab)
    r"\u06D6-\u06DC"   # Quranic annotation marks above letters
    r"\u06DD"          # Arabic End of Ayah marker
    r"\u06DE"          # Arabic Start of Rub El Hizb
    r"\u06DF-\u06E4"   # More Quranic marks
    r"\u06E7-\u06ED"   # More Quranic marks
    r"\u06E9"          # Arabic Place of Sajdah
    r"]"
)

# Unify Alef variants to bare alef
_ALEF_VARIANTS = str.maketrans("إأآٱ", "اااا")


def normalise_text(text: str) -> str:
    """
    Normalise Arabic text for pcd model fine-tuning.

    Rules:
    - KEEP harakat/diacritics  (model is the pcd variant — trained with them)
    - KEEP Arabic punctuation  (model outputs ، ؟ .)
    - STRIP Quranic annotation marks  (sajda, rub-el-hizb etc.)
    - UNIFY Alef variants to bare alef
    - COLLAPSE multiple spaces
    """
    text = text.strip()
    text = text.translate(_ALEF_VARIANTS)
    text = _STRIP_CHARS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

SAMPLE_RATE    = 16000
MAX_DURATION_S = 20.0   # model hard limit (must match asr_server.py)
MIN_DURATION_S = 0.3    # skip near-empty files


def save_wav_mono_16k(array: np.ndarray, src_sr: int, out_path: Path) -> float:
    """
    Resample to 16 kHz mono, save as 16-bit PCM WAV.
    Returns actual duration in seconds.
    librosa is imported here (not at module-top) to allow runs that only
    inspect manifests without saving audio.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa not installed.  Run: pip install librosa")

    arr = np.asarray(array, dtype=np.float32)

    # Mono BEFORE resample — cheaper
    if arr.ndim > 1:
        arr = arr.mean(axis=0)

    if src_sr != SAMPLE_RATE:
        # librosa >= 0.9 requires keyword args orig_sr / target_sr
        arr = librosa.resample(arr, orig_sr=src_sr, target_sr=SAMPLE_RATE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), arr, SAMPLE_RATE, subtype="PCM_16")
    return float(len(arr)) / SAMPLE_RATE


def _duration_ok(array, sr: int) -> bool:
    dur = len(array) / sr
    return MIN_DURATION_S <= dur <= MAX_DURATION_S


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def _write_manifest(path: Path, records: list, dry_run: bool) -> None:
    if dry_run:
        total_dur = sum(r["duration"] for r in records)
        log.info(f"  [DRY RUN] {len(records)} records ({total_dur/3600:.1f} h) -> {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    total_dur = sum(r["duration"] for r in records)
    log.info(f"  Wrote {len(records)} records ({total_dur/3600:.1f} h) -> {path}")


def _make_entry(out_path: Path, text: str, duration: float,
                dry_run: bool, idx: int, tag: str) -> dict:
    filepath = f"<dry_run/{tag}_{idx:07d}.wav>" if dry_run else str(out_path.resolve())
    return {"audio_filepath": filepath, "text": text, "duration": round(duration, 3)}


# ---------------------------------------------------------------------------
# Per-dataset processors
# ---------------------------------------------------------------------------

def process_everyayah_tarteel(audio_dir: Path, output_dir: Path, dry_run: bool) -> dict:
    """
    tarteel-ai/everyayah — 390 h, 70+ reciters, already diacritized.
    This is the model's ORIGINAL training source; we use it to expose
    unseen reciters that were not in the original training split.

    Split strategy: hold out ~10% of reciters for validation.
    """
    from datasets import load_dataset

    log.info("Loading tarteel-ai/everyayah ...")
    ds       = load_dataset("tarteel-ai/everyayah", split="train", trust_remote_code=True)
    reciters = sorted(set(ds["reciter"]))
    n_val    = max(1, len(reciters) // 10)
    val_set  = set(reciters[:n_val])
    log.info(f"  {len(reciters)} reciters | {n_val} held out for val: {list(val_set)[:3]} ...")

    records  = {"train": [], "val": []}
    skipped  = 0

    for i, row in enumerate(tqdm(ds, desc="everyayah")):
        audio   = row.get("audio") or {}
        array   = audio.get("array")
        sr      = audio.get("sampling_rate", SAMPLE_RATE)
        text    = (row.get("text") or "").strip()

        if array is None or not text or not _duration_ok(array, sr):
            skipped += 1
            continue

        text = normalise_text(text)
        if not text:
            skipped += 1
            continue

        split    = "val" if row.get("reciter", "") in val_set else "train"
        duration = float(len(array)) / sr

        if dry_run:
            records[split].append(_make_entry(Path(), text, duration, True, i, "everyayah"))
            continue

        out_path = audio_dir / "everyayah" / split / f"{i:07d}.wav"
        try:
            duration = save_wav_mono_16k(array, sr, out_path)
        except Exception as exc:
            log.debug(f"  Save failed [{i}]: {exc}")
            skipped += 1
            continue

        records[split].append(_make_entry(out_path, text, duration, False, i, "everyayah"))

    log.info(f"  everyayah: train={len(records['train'])} val={len(records['val'])} skipped={skipped}")
    manifests = {}
    for split, recs in records.items():
        if recs:
            p = output_dir / f"everyayah_{split}.json"
            _write_manifest(p, recs, dry_run)
            manifests[split] = p
    return manifests


def process_everyayah_extended(audio_dir: Path, output_dir: Path, dry_run: bool) -> dict:
    """
    Salama1429/tarteel-ai-everyayah-Quran — 829 h extended version.
    Uses the dataset's own train/validation splits where present.
    """
    from datasets import load_dataset, DatasetDict

    log.info("Loading Salama1429/tarteel-ai-everyayah-Quran ...")
    ds = load_dataset(
        "Salama1429/tarteel-ai-everyayah-Quran",
        verification_mode="no_checks",
        trust_remote_code=True,
    )

    if isinstance(ds, DatasetDict):
        to_process = {}
        if "train" in ds:
            to_process["train"] = ds["train"]
        if "validation" in ds:
            to_process["val"] = ds["validation"]
        elif "test" in ds:
            to_process["val"] = ds["test"]
        if not to_process:
            first = list(ds.keys())[0]
            to_process["train"] = ds[first]
    else:
        to_process = {"train": ds}

    manifests = {}
    for split_name, split_ds in to_process.items():
        records = []
        skipped = 0
        for i, row in enumerate(tqdm(split_ds, desc=f"everyayah_ext/{split_name}")):
            audio  = row.get("audio") or {}
            array  = audio.get("array")
            sr     = audio.get("sampling_rate", SAMPLE_RATE)
            text   = (row.get("text") or "").strip()

            if array is None or not text or not _duration_ok(array, sr):
                skipped += 1
                continue

            text = normalise_text(text)
            if not text:
                skipped += 1
                continue

            duration = float(len(array)) / sr

            if dry_run:
                records.append(_make_entry(Path(), text, duration, True, i, "everyayah_ext"))
                continue

            out_path = audio_dir / "everyayah_ext" / split_name / f"{i:07d}.wav"
            try:
                duration = save_wav_mono_16k(array, sr, out_path)
            except Exception as exc:
                log.debug(f"  Save failed [{i}]: {exc}")
                skipped += 1
                continue

            records.append(_make_entry(out_path, text, duration, False, i, "everyayah_ext"))

        log.info(f"  everyayah_ext/{split_name}: {len(records)} OK, {skipped} skipped")
        p = output_dir / f"everyayah_ext_{split_name}.json"
        _write_manifest(p, records, dry_run)
        manifests[split_name] = p

    return manifests


def process_buraaq(audio_dir: Path, output_dir: Path, dry_run: bool) -> dict:
    """
    Buraaq/quran-audio-text-dataset — 30 reciters x 6236 ayahs.
    Split: Surahs 110-114 held out as validation.
    """
    from datasets import load_dataset

    log.info("Loading Buraaq/quran-audio-text-dataset ...")
    ds      = load_dataset("Buraaq/quran-audio-text-dataset", split="train", trust_remote_code=True)
    records = {"train": [], "val": []}
    skipped = 0

    for i, row in enumerate(tqdm(ds, desc="buraaq")):
        audio  = row.get("audio") or {}
        array  = audio.get("array")
        sr     = audio.get("sampling_rate", SAMPLE_RATE)
        text   = (row.get("text") or "").strip()

        if array is None or not text or not _duration_ok(array, sr):
            skipped += 1
            continue

        text = normalise_text(text)
        if not text:
            skipped += 1
            continue

        try:
            surah = int(row.get("surah_number", 0))
        except (ValueError, TypeError):
            surah = 0

        split    = "val" if surah >= 110 else "train"
        duration = float(len(array)) / sr

        if dry_run:
            records[split].append(_make_entry(Path(), text, duration, True, i, "buraaq"))
            continue

        out_path = audio_dir / "buraaq" / split / f"{i:07d}.wav"
        try:
            duration = save_wav_mono_16k(array, sr, out_path)
        except Exception as exc:
            log.debug(f"  Save failed [{i}]: {exc}")
            skipped += 1
            continue

        records[split].append(_make_entry(out_path, text, duration, False, i, "buraaq"))

    log.info(f"  buraaq: train={len(records['train'])} val={len(records['val'])} skipped={skipped}")
    manifests = {}
    for split, recs in records.items():
        if recs:
            p = output_dir / f"buraaq_{split}.json"
            _write_manifest(p, recs, dry_run)
            manifests[split] = p
    return manifests


def process_retasy(audio_dir: Path, output_dir: Path, dry_run: bool) -> dict:
    """
    RetaSy/quranic_audio_dataset — crowdsourced recordings.
    Only rows where final_label == "correct" are used.
    Text column is "Aya" (diacritized verse text).
    """
    from datasets import load_dataset

    log.info("Loading RetaSy/quranic_audio_dataset ...")
    ds     = load_dataset("RetaSy/quranic_audio_dataset", split="train", trust_remote_code=True)
    before = len(ds)
    ds     = ds.filter(lambda r: r.get("final_label") == "correct")
    log.info(f"  Correct recitations: {before} -> {len(ds)}")

    records = {"train": [], "val": []}
    skipped = 0

    for i, row in enumerate(tqdm(ds, desc="retasy")):
        audio  = row.get("audio") or {}
        array  = audio.get("array")
        sr     = audio.get("sampling_rate", SAMPLE_RATE)
        text   = (row.get("Aya") or "").strip()        # <-- "Aya" is the correct column

        if array is None or not text or not _duration_ok(array, sr):
            skipped += 1
            continue

        text = normalise_text(text)
        if not text:
            skipped += 1
            continue

        reciter_id = str(row.get("reciter_id", ""))
        split      = "val" if reciter_id and reciter_id[-1] in "0123" else "train"
        duration   = float(len(array)) / sr

        if dry_run:
            records[split].append(_make_entry(Path(), text, duration, True, i, "retasy"))
            continue

        out_path = audio_dir / "retasy" / split / f"{i:07d}.wav"
        try:
            duration = save_wav_mono_16k(array, sr, out_path)
        except Exception as exc:
            log.debug(f"  Save failed [{i}]: {exc}")
            skipped += 1
            continue

        records[split].append(_make_entry(out_path, text, duration, False, i, "retasy"))

    log.info(f"  retasy: train={len(records['train'])} val={len(records['val'])} skipped={skipped}")
    manifests = {}
    for split, recs in records.items():
        if recs:
            p = output_dir / f"retasy_{split}.json"
            _write_manifest(p, recs, dry_run)
            manifests[split] = p
    return manifests


# ---------------------------------------------------------------------------
# Combine all manifests into one train + one val
# ---------------------------------------------------------------------------

def merge_manifests(output_dir: Path, all_manifests: dict, dry_run: bool) -> dict:
    """
    Merge per-dataset manifests into combined_train.json / combined_val.json.

    BUG FIX: In dry_run mode the manifest files do not exist on disk,
    so we must NOT attempt to open them.  We just log what would happen.
    """
    if dry_run:
        log.info("  [DRY RUN] Skipping combined manifest merge (no files written).")
        return {}

    combined = {"train": [], "val": []}

    for dataset_name, splits in all_manifests.items():
        for split_key, path in splits.items():
            target = "val" if "val" in split_key else "train"
            path   = Path(path)
            if not path.exists():
                log.warning(f"  Missing manifest, skipping: {path}")
                continue
            with open(path, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            combined[target].extend(lines)
            log.info(f"  Merged {len(lines):>6} records from {dataset_name}/{split_key} -> {target}")

    merged = {}
    for split, records in combined.items():
        if not records:
            continue
        random.seed(42)
        random.shuffle(records)
        path = output_dir / f"combined_{split}.json"
        _write_manifest(path, records, dry_run=False)
        merged[split] = path

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASET_MAP = {
    "everyayah":     process_everyayah_tarteel,
    "everyayah_ext": process_everyayah_extended,
    "buraaq":        process_buraaq,
    "retasy":        process_retasy,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare NeMo manifests for Quran ASR fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASET_MAP.keys()),
        choices=list(DATASET_MAP.keys()),
        help="Which datasets to process (default: all)",
    )
    p.add_argument("--output_dir",  default="./manifests",   help="Manifest output directory")
    p.add_argument("--audio_dir",   default="./quran_audio", help="WAV output directory")
    p.add_argument("--dry_run",     action="store_true",     help="Skip downloads — test code paths only")
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir)
    audio_dir  = Path(args.audio_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Quran ASR — NeMo Manifest Preparation  v2.0  (bug-fixed)")
    log.info(f"  Model   : nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0")
    log.info(f"  Datasets: {args.datasets}")
    log.info(f"  Output  : {output_dir}")
    log.info(f"  Audio   : {audio_dir}")
    log.info(f"  Dry run : {args.dry_run}")
    log.info("=" * 60)

    all_manifests: dict = {}
    for name in args.datasets:
        fn     = DATASET_MAP[name]
        result = fn(audio_dir, output_dir, args.dry_run)
        if isinstance(result, dict):
            all_manifests[name] = result
        else:
            log.warning(f"  Dataset '{name}' returned unexpected result: {type(result)}")

    if len(args.datasets) > 1:
        log.info("\nMerging individual manifests ...")
        merged = merge_manifests(output_dir, all_manifests, args.dry_run)
        if merged:
            log.info("\n[ Final manifests ]")
            for split, path in merged.items():
                log.info(f"   {split}: {path}")
    else:
        log.info("\n[ Done ]")

    log.info("\nNext step: run  finetune_quran_asr.py  with these manifests.")


if __name__ == "__main__":
    main()
