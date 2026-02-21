# Quran ASR — Full Evaluation Guide
### Model: `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
### Platform: Kaggle (Free T4 GPU)

---

## Table of Contents

1. [What Is This Evaluation and Why Do We Do It](#1-what-is-this-evaluation-and-why-do-we-do-it)
2. [Understanding the Model](#2-understanding-the-model)
3. [Understanding the Data](#3-understanding-the-data)
4. [Understanding the Metrics](#4-understanding-the-metrics)
5. [Understanding the Script](#5-understanding-the-script)
6. [Kaggle Setup — Step by Step](#6-kaggle-setup--step-by-step)
7. [Running the Evaluation](#7-running-the-evaluation)
8. [Reading Your Results](#8-reading-your-results)
9. [What Comes After This](#9-what-comes-after-this)
10. [Troubleshooting](#10-troubleshooting)
11. [Quick Reference Card](#11-quick-reference-card)

---

## 1. What Is This Evaluation and Why Do We Do It

### The Goal

Before you fine-tune any machine learning model, you need to know **where it starts**. This number is called your **baseline**. Without it, you have no way to know whether your fine-tuning improved things, made them worse, or did nothing at all.

This evaluation measures how accurately the model transcribes Quran recitation **right now, before you touch it**. After fine-tuning, you run the same evaluation again. If the number improves, your fine-tuning worked.

### The Analogy

Think of it like a student taking a placement test before a course. The test score tells you where the student is starting from. At the end of the course, they take the same test again. The difference between the two scores is how much they learned.

### What This Evaluation Produces

- A **WER (Word Error Rate)** number per reciter — the percentage of words the model got wrong
- A **CER (Character Error Rate)** number per reciter — the percentage of characters the model got wrong
- A **full JSON report** with every single verse: what the reference text was, what the model said, and how many errors were made
- A **summary CSV** you can open in Excel or Google Sheets

---

## 2. Understanding the Model

### Model Name
```
nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
```

### What Each Part of the Name Means

| Part | Meaning |
|------|---------|
| `nvidia` | Built by NVIDIA's NeMo team |
| `stt` | Speech-To-Text |
| `ar` | Arabic language |
| `fastconformer` | The neural network architecture (fast version of Conformer) |
| `hybrid` | Has two decoders: CTC (fast) and RNNT (accurate) |
| `large` | Large model size — more parameters, more accurate |
| `pcd` | Outputs **P**unctuation + **D**iacritics (tashkeel) in its transcriptions |
| `v1.0` | Version 1.0 |

### The Two Decoders (Critical to Understand)

This model has two ways to decode speech, used for different purposes:

**CTC Decoder — Fast, Non-Autoregressive**
- Processes audio in ~30-50ms
- Used for **interim (live) results** while someone is still speaking
- Less accurate, but nearly instant
- Used in the server for streaming feedback

**RNNT Decoder — Accurate, Autoregressive**
- Processes the complete utterance
- Used for **final results** after someone finishes speaking
- More accurate, takes slightly longer
- Used in the evaluation script and in the server's final output

> **The evaluation script uses RNNT only.** This is correct — you want to measure the accurate decoder, not the live preview decoder.

### Why FP16 / autocast MUST NOT Be Used

This is documented in your server code (`v4.9` changelog) and is critical:

When you run this model in FP16 (half precision) or with PyTorch `autocast`:
- White noise (warm-up) transcribes correctly — this fools you into thinking it works
- Real speech concentrates energy in specific frequency bands (formants)
- These high-energy activations **overflow in FP16 → become NaN**
- Every token decodes as the SentencePiece unknown character `⁇`
- The model outputs nothing but question marks

**Always use float32 for this model. No exceptions.**

### NVIDIA's Reference Numbers

| Decoder | WER on EveryAyah Test Split |
|---------|----------------------------|
| RNNT | **6.55%** |
| CTC | **7.61%** |

These are NVIDIA's own published numbers. Your evaluation will produce comparable results because you are testing on the same data source.

---

## 3. Understanding the Data

### Ground Truth Text — Tanzil Uthmani

- Source: `api.alquran.cloud/v1/quran/quran-uthmani`
- 6,236 verses across 114 surahs
- Uthmani script with full diacritics (tashkeel)
- Downloaded once, cached to `quran_uthmani.json`
- Validated: script exits if count ≠ 6,236

### Audio Files — EveryAyah.com

- 6,236 MP3 files per reciter
- Studio-quality recordings
- Naming format: `{surah:03d}{verse:03d}.mp3` → e.g. `002001.mp3` = Al-Baqarah verse 1
- 5 reciters evaluated:

| Reciter | Bitrate | Style |
|---------|---------|-------|
| Abdul_Basit_Murattal_64kbps | 64kbps | Murattal (recitation pace) |
| Alafasy_128kbps | 128kbps | Clear, widely known |
| Husary_128kbps | 128kbps | Classical, precise |
| Mohammad_al_Tablaway_128kbps | 128kbps | Egyptian style |
| Minshawi_Murattal_128kbps | 128kbps | Murattal style |

### ⚠️ The Data Contamination Problem

**This is the most important thing to understand about your evaluation results.**

The model was trained on **TarteelAI/EveryAyah — 390 hours** of audio downloaded from the same website this script downloads from (EveryAyah.com).

This means:
- The model has **already heard these exact recordings** during training
- It has learned the specific microphone, room acoustics, and vocal characteristics of each reciter
- The WER you measure (~6-8%) reflects performance on **known data**, not unknown data
- On truly unseen recitation (e.g. someone reading on a phone), real WER would be higher (~15-30%)

**Why run this evaluation at all then?**

1. It establishes your **domain-specific baseline** for EveryAyah audio
2. After fine-tuning, running it again tells you if fine-tuning caused **regression** on this data
3. NVIDIA's 6.55% gives you a concrete comparison point
4. The infrastructure (script, checkpoints, reports) is built and ready for your own test data too

**For a truly unbiased evaluation, you also need your own recordings** — see [Section 9](#9-what-comes-after-this).

---

## 4. Understanding the Metrics

### WER — Word Error Rate

WER measures what percentage of words the model got wrong.

**Formula:**
```
WER = (Substitutions + Deletions + Insertions) / Total Reference Words
```

**Example:**
```
Reference : بسم الله الرحمن الرحيم
Hypothesis: بسم الله الرحيم
                        ↑ deletion

WER = 1 deletion / 4 reference words = 25%
```

### CER — Character Error Rate

Same idea as WER but at the character level instead of word level.

```
CER = Character edit distance / Total Reference Characters
```

CER is always lower than WER because getting one word wrong usually affects fewer characters than words.

### Corpus-Level vs Per-Verse Average (Critical)

The script calculates **corpus-level WER**, not per-verse average. This is the industry standard. Here is why it matters:

**Per-verse average (wrong approach):**
```
Verse 1 (3 words):  1 error → WER = 33%
Verse 2 (20 words): 1 error → WER = 5%
Average WER = (33% + 5%) / 2 = 19%   ← misleading
```

**Corpus-level (correct approach):**
```
Verse 1 (3 words):  1 error
Verse 2 (20 words): 1 error
Total: 2 errors / 23 words = 8.7%    ← honest
```

A 3-word verse and a 20-word verse should not count equally.

### How Failed Verses Are Handled

If a verse fails (download error, conversion error, or transcription crash), the script counts it as **100% wrong** — all its words are counted as errors. It does NOT silently exclude it.

This is the honest approach: if 100 verses couldn't be transcribed, your WER should reflect that, not pretend those verses don't exist.

### Text Normalization Before Comparison

Both the reference text and the model's hypothesis are normalized identically before comparison:

| Step | What It Does | Why |
|------|-------------|-----|
| Strip diacritics | Removes tashkeel (فَتْحَة، كَسْرَة، etc.) | Test word recognition, not vowel marks |
| Strip punctuation | Removes ،؛؟ and ASCII . , ; : | Model outputs punctuation, reference doesn't |
| Alef variants → ا | إ أ آ ٱ all become ا | Model and reference may disagree on which alef form to use |
| ة → ه | Teh marbuta becomes heh | Some models output ه where text has ة |
| ى → ي | Alef maqsura becomes yeh | Words like على، إلى — model outputs ي, reference has ى |
| Collapse whitespace | Multiple spaces → single space | Clean token boundaries |

---

## 5. Understanding the Script

### File Structure After Running

```
/kaggle/working/
├── evaluate_full.py          ← the evaluation script
├── quran_uthmani.json        ← cached Quran text (6,236 verses)
├── evaluation.log            ← full log of everything that happened
│
├── audio/
│   └── Alafasy_128kbps/
│       ├── 001001.mp3        ← Al-Fatiha verse 1
│       ├── 001002.mp3        ← Al-Fatiha verse 2
│       └── ...               ← 6,236 files total
│
├── checkpoints/
│   └── Alafasy_128kbps.json  ← progress saved after every 16-verse batch
│
└── results/
    ├── full_report.json      ← every verse, every reciter, full detail
    └── summary.csv           ← one row per reciter, WER and CER
```

### How Checkpointing Works

The script saves progress after every batch of 16 verses. If the Kaggle session is killed (9-hour timeout, network drop, OOM crash), you can simply re-run the exact same command. It will:

1. Load the checkpoint
2. Skip all verses already marked `"ok"`
3. Retry all verses that previously failed (download/conversion/transcription errors)
4. Continue from where it stopped

Nothing is lost. You never re-download or re-transcribe a verse that already succeeded.

### The Complete Internal Flow

```
main()
  │
  ├── parse_args()              Read --reciters and --batch-size from command line
  ├── check_dependencies()      Verify ffmpeg, soundfile, pydub — exit if missing
  ├── load_quran_text()         Download 6,236 verses from alquran.cloud (or load cache)
  ├── load_model()              Download and load NVIDIA model onto GPU in float32
  │
  └── for each reciter:
        evaluate_reciter()
          │
          ├── load_checkpoint()           Load previously completed verses
          ├── build remaining list        Only verses not yet "ok"
          ├── download_all_parallel()     Download MP3s (8 threads, rate-limited)
          │
          └── for each verse:
                mp3_to_float32()          Decode MP3 → 16kHz mono float32 PCM
                save_wav()                Write to /tmp as 16-bit WAV
          
          └── for each batch of 16:
                transcribe_batch()        GPU inference → Arabic text hypothesis
                word_errors()             Levenshtein distance at word level
                char_errors()             Levenshtein distance at character level
                save_checkpoint()         Atomic write (safe to kill mid-run)
  │
  └── save_reports()            Write full_report.json and summary.csv, print table
```

### Key Configuration Values

```python
BATCH_SIZE       = 16      # verses per GPU batch — safe for T4 (15GB VRAM)
DOWNLOAD_WORKERS = 8       # parallel MP3 download threads
DOWNLOAD_DELAY   = 0.15    # seconds between requests per worker (polite)
MAX_RETRIES      = 4       # retry attempts per failed download
SAMPLE_RATE      = 16000   # Hz — model's required input sample rate
EXPECTED_VERSES  = 6236    # hard validation on Quran text download
```

---

## 6. Kaggle Setup — Step by Step

### Prerequisites

- Kaggle account with **phone number verified**
  - Go to: `kaggle.com/settings` → Phone Verification
  - Without this, GPU T4 x2 is grayed out and unavailable

### Step 1: Create a New Notebook

1. Go to `kaggle.com/code`
2. Click **New Notebook**
3. Click **Settings** (top menu)
4. Set **Accelerator** → `GPU T4 x2`
5. Set **Internet** → `On`

### Step 2: Cell 1 — Install Dependencies

Delete the default content in the first cell. Paste:

```python
!pip install nemo_toolkit[asr] pydub --quiet
!apt-get install -y ffmpeg libsndfile1 -q
```

Run the cell. Takes ~5 minutes.

> **ffmpeg and libsndfile1 are usually pre-installed on Kaggle.** You will see "already the newest version" — that is fine.

After the cell finishes: **Run → Restart & clear cell outputs**

This restart is mandatory. NeMo modifies the Python environment during install and the changes only take effect after a fresh kernel.

### Step 3: Cell 2 — Verify the Environment

```python
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_properties(0).name)
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

import nemo.collections.asr as nemo_asr
print("NeMo: OK")

import pydub
print("pydub: OK")

import soundfile
print("soundfile: OK")
```

**Expected output:**
```
GPU available: True
GPU name: Tesla T4
VRAM: 15.0 GB
NeMo: OK
pydub: OK
soundfile: OK
```

Do not proceed if GPU available shows `False`.

### Step 4: Cell 3 — Write the Script to Disk

```python
%%writefile evaluate_full.py
# paste your entire evaluate_full.py content here
```

Run the cell. You should see:
```
Writing evaluate_full.py
```

### Step 5: Cell 4 — Run Single Reciter Test

```python
!python evaluate_full.py --reciters Alafasy_128kbps --batch-size 16
```

**Why single reciter first?**
- Takes ~35 minutes instead of ~3 hours
- Confirms the script works correctly on this environment
- If there is a bug or crash, you find out quickly
- Once this succeeds, you run all 5 reciters

### Step 6: Cell 5 — Run All Reciters (after Step 5 succeeds)

```python
!python evaluate_full.py --batch-size 16
```

Alafasy is already checkpointed from Step 5, so it is skipped automatically. The remaining 4 reciters take ~25-30 minutes each.

### Step 7: Cell 6 — Save Results

Run this **before the session ends** (Kaggle sessions expire after 9 hours):

```python
import shutil

shutil.copytree("results", "/kaggle/working/results", dirs_exist_ok=True)
shutil.copytree("checkpoints", "/kaggle/working/checkpoints", dirs_exist_ok=True)

print("Results saved to /kaggle/working/")
```

Then click **Save Version** (top right) → **Save & Run All** to commit and preserve output files.

---

## 7. Running the Evaluation

### Command Reference

```bash
# Run all 5 reciters (default)
!python evaluate_full.py --batch-size 16

# Run a single reciter
!python evaluate_full.py --reciters Alafasy_128kbps --batch-size 16

# Run multiple specific reciters
!python evaluate_full.py --reciters Alafasy_128kbps Husary_128kbps --batch-size 16

# If GPU out-of-memory errors occur, lower batch size
!python evaluate_full.py --batch-size 8

# Resume after interruption (checkpoints are automatic — just re-run)
!python evaluate_full.py --batch-size 16
```

### What --batch-size Means

The model processes audio in batches. Instead of transcribing one verse at a time, it transcribes multiple verses simultaneously on the GPU.

```
batch-size 16 = send 16 verses to GPU at once → all processed in parallel
```

| Batch Size | VRAM Usage | Speed | Use When |
|-----------|-----------|-------|----------|
| 8 | ~5 GB | Slower | Getting OOM errors |
| 16 | ~8-10 GB | Good | **Default — use this** |
| 32 | ~14 GB | Fast | If you have extra VRAM |

### Time Estimates on T4

| Task | Time |
|------|------|
| Install + restart | ~5 min |
| Environment verification | ~1 min |
| Model download + load | ~3 min |
| Per reciter (first time) | ~25-35 min |
| Per reciter (resumed) | Proportional to remaining verses |
| 5 reciters total | ~2-3 hours |

### Progress Output During Run

You will see output like this:

```
2026-02-21 00:30:00  INFO  Loading Quran text from cache: quran_uthmani.json
2026-02-21 00:30:00  INFO    6236 verses loaded from cache
2026-02-21 00:30:00  INFO  Loading model: nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
2026-02-21 00:33:00  INFO    GPU  : Tesla T4
2026-02-21 00:33:00  INFO    VRAM : 15.0 GB
2026-02-21 00:33:00  INFO    Loaded in 180.3s

=================================================================
  Reciter : Alafasy_128kbps
  Verses  : 6236
=================================================================
  Downloading 6236 MP3s (8 workers) ...
  Downloading: 100%|████████████| 6236/6236 [08:45<00:00]
  Downloaded: 6236  Failed: 0

  Converting MP3 -> WAV ...
  Converting: 100%|████████████| 6236/6236 [02:10<00:00]

  Transcribing 6236 verses (batch_size=16) ...
  Transcribing: 100%|████████████| 6236/6236 [18:30<00:00]

  Finished in 29.2 min — 6236 OK, 0 failed
```

---

## 8. Reading Your Results

### Console Output

After all reciters complete, the script prints a results table:

```
════════════════════════════════════════════════════════════════════════
  EVALUATION RESULTS — nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
  Date     : 2026-02-21 03:15:00
  Metric   : corpus-level WER / CER (industry standard)
  NVIDIA   : 6.55% WER (RNNT) / 7.61% WER (CTC) on EveryAyah test split
  WARNING  : Model trained on EveryAyah data — WER is optimistically low
════════════════════════════════════════════════════════════════════════
  Reciter                                      WER       CER    OK / Total
  ────────────────────────────────────────── ─────── ─────── ────────────
  Abdul_Basit_Murattal_64kbps                 7.82%    2.31%    6236/6236
  Alafasy_128kbps                             6.41%    1.94%    6236/6236
  Husary_128kbps                              6.89%    2.10%    6236/6236
  Mohammad_al_Tablaway_128kbps                8.15%    2.45%    6236/6236
  Minshawi_Murattal_128kbps                   7.54%    2.28%    6236/6236
════════════════════════════════════════════════════════════════════════
```

*(Numbers above are illustrative — your actual results may differ slightly)*

### What to Look For

**WER around 6-10%** → Normal. Expected given the model was trained on this data.

**WER significantly above 10%** → Investigate. Possible causes:
- Network errors caused many download failures (check `failed_verses` in CSV)
- Wrong model loaded
- Text normalization issue

**Failed verses > 0** → Check `evaluation.log` for the specific errors. A handful of 404s from EveryAyah is normal. More than ~50 failures is worth investigating.

**WER = 100%** → Something went badly wrong. Model probably didn't load correctly.

### The full_report.json File

Every single verse is in this file. You can inspect specific errors:

```json
{
  "surah": 2,
  "verse": 255,
  "reference": "اللَّهُ لَا إِلَهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
  "hypothesis": "الله لا إله إلا هو الحي القيوم",
  "word_errors": 0,
  "ref_words": 7,
  "char_errors": 0,
  "ref_chars": 34,
  "status": "ok"
}
```

This shows Ayat al-Kursi was transcribed perfectly (0 word errors, 0 char errors).

### The summary.csv File

Open in Excel or Google Sheets. Columns:

| Column | Meaning |
|--------|---------|
| reciter | Reciter name |
| corpus_wer | Word Error Rate (0.0 = perfect, 1.0 = everything wrong) |
| corpus_cer | Character Error Rate |
| ok_verses | Verses successfully transcribed |
| failed_verses | Verses that failed (download/conversion/transcription) |
| total_verses | Should always be 6,236 |
| total_word_errors | Raw count — divide by total_ref_words to verify WER |
| total_ref_words | Total reference word count across all verses |
| total_char_errors | Raw count |
| total_ref_chars | Total reference character count |

---

## 9. What Comes After This

### Immediate Next Steps

**Step 1: Record your own test set**

This is the most important thing for honest evaluation. Even 100 verses from 3-5 different people on phone microphones gives you a real uncontaminated baseline.

Why: The EveryAyah WER (~6-8%) is optimistic. On real users with real microphones in real environments, the model's actual WER before fine-tuning will be higher. You need to know this number.

**Step 2: Evaluate on your own recordings**

Run the same evaluation pipeline but with your own audio instead of EveryAyah. This is your true baseline — the number your fine-tuning needs to beat.

**Step 3: Prepare your fine-tuning data**

Collect and label training data for fine-tuning. The more it represents your actual deployment conditions (microphone type, background noise, speaker demographics), the better.

**Step 4: Fine-tune**

Fine-tune using NeMo's training pipeline on your collected data.

**Step 5: Re-evaluate**

Run this exact script again on the fine-tuned model. Compare:
- EveryAyah WER: should stay similar (not regress)
- Your own recordings WER: should improve significantly

### How to Know If Fine-Tuning Worked

| Scenario | EveryAyah WER | Your Data WER | Interpretation |
|----------|--------------|----------------|---------------|
| ✅ Good | Similar (~6-10%) | Lower than before | Fine-tuning worked |
| ⚠️ Overfit | Higher than before | Lower than before | Overfitting to your data |
| ❌ Broken | Much higher | Much higher | Something went wrong in training |
| ➡️ No change | Same | Same | Fine-tuning had no effect |

---

## 10. Troubleshooting

### GPU not available (False)

```
GPU available: False
```

→ Go to Settings → Accelerator → select GPU T4 x2
→ If grayed out, verify phone at kaggle.com/settings

### NeMo import error after kernel restart

```
ModuleNotFoundError: No module named 'nemo'
```

→ The kernel restart cleared the install. Re-run Cell 1 (the pip install cell), then restart again.

### Out of memory (CUDA OOM)

```
RuntimeError: CUDA out of memory
```

→ Lower the batch size:
```python
!python evaluate_full.py --batch-size 8
```

### Download failures / name resolution errors

```
Failed to establish a new connection: [Errno -3] Temporary failure in name resolution
```

→ Internet is OFF. Go to Settings → Internet → On.

### Session killed before completion (9-hour timeout)

→ Just re-run the same command. Checkpoints are saved after every batch. The script resumes automatically from where it stopped.

### Many 404 errors in logs

```
404 (file not on server): https://everyayah.com/data/...
```

→ A handful of 404s is normal — a few files may not exist on the server. More than ~50 is unusual. Check if EveryAyah.com is accessible from Kaggle.

### WER seems too high (>20%)

→ Check `failed_verses` count in summary.csv. If many verses failed, WER is inflated by failures counted at 100% error. Check `evaluation.log` for the failure reason.

### pydub SyntaxWarning

```
SyntaxWarning: invalid escape sequence '\('
```

→ Harmless warning from an older version of pydub. Does not affect functionality. Ignore it.

### Megatron / OneLogger warnings at startup

```
[NeMo W] Megatron num_microbatches_calculator not found
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY
```

→ Harmless. These are NeMo's internal warnings about optional components. Ignore them.

---

## 11. Quick Reference Card

### Commands

```bash
# Single reciter (first test)
!python evaluate_full.py --reciters Alafasy_128kbps --batch-size 16

# All reciters (full evaluation)
!python evaluate_full.py --batch-size 16

# Resume after interruption
!python evaluate_full.py --batch-size 16   # same command — checkpoints are automatic

# Out of memory fix
!python evaluate_full.py --batch-size 8
```

### Expected Numbers

| What | Expected Value |
|------|---------------|
| WER on EveryAyah (RNNT) | ~6-10% |
| NVIDIA reference WER | 6.55% |
| Verses per reciter | 6,236 |
| Time per reciter (T4) | ~25-35 min |
| Total time (5 reciters) | ~2-3 hours |
| GPU VRAM needed (batch 16) | ~8-10 GB |

### Key Files

| File | Description |
|------|-------------|
| `evaluate_full.py` | The evaluation script |
| `quran_uthmani.json` | Cached Quran text (auto-downloaded) |
| `evaluation.log` | Full run log |
| `audio/{reciter}/` | Downloaded MP3 files |
| `checkpoints/{reciter}.json` | Progress checkpoint (auto-saved) |
| `results/full_report.json` | Complete results, every verse |
| `results/summary.csv` | Summary table, one row per reciter |

### Metric Formulas

```
WER = total word edit distance / total reference words
CER = total char edit distance / total reference chars

Both are corpus-level (not per-verse averages).
Both include failed verses at 100% error rate.
Lower is better. 0.0 = perfect. 1.0 = everything wrong.
```

---

*This guide covers the full evaluation pipeline for `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` on Kaggle. After completing this evaluation, you have your pre-fine-tuning baseline and are ready to proceed to data collection and fine-tuning.*
