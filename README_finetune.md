# Quran ASR Fine-Tuning Pipeline  v4.0
### Model: `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
### Status: Every API verified against actual NeMo stable source code

---

## Complete Bug History — v1 → v4  (23 bugs total)

### `finetune_quran_asr.py`

| # | Found in | Severity | Bug | Fix |
|---|----------|----------|-----|-----|
| 1 | v1 | 💥 CRASH | `from nemo.utils import exp_manager as em_utils` — nemo.utils.exp_manager is a MODULE not callable | `from nemo.utils.exp_manager import exp_manager` |
| 2 | v1 | 💥 CRASH | `em_utils.exp_manager()` called inside `run_phase()` — `em_utils` undefined in scope | Removed, correct calls inlined |
| 3 | v1 | 💥 CRASH | `model.update_config_after_loading()` — method does not exist in NeMo | Use `setup_training_data()` + `setup_optimization()` |
| 4 | v1 | 💥 CRASH | `cfg2.trainer["accumulate_grad_batches"] = …` — OmegaConf DictConfig raises `ReadonlyConfigError` | Pass `accumulate_grad_batches` directly to `pl.Trainer(...)` |
| 5 | v1 | 💥 CRASH | `json.loads()` called before `import json` in `evaluate_model()` | `import json` moved to module top |
| 6 | v1 | 💥 CRASH | `ASRLinearAdapterConfig` import path does not exist in NeMo | `model.add_adapter()` with `_target_`-based OmegaConf config |
| 7 | v1 | 💥 CRASH | `model.cfg.model.encoder.d_model` — double `.model` in path | `model.cfg.encoder.d_model` |
| 8 | v1 | ⚠️ WRONG | `spec_augment` placed inside `train_ds.augmentor` — wrong section | Removed; model already has SpecAugment baked in |
| 9 | v1 | 💥 CRASH | `"trim_silence": False` in dataset config — not a valid NeMo `AudioToText` field | Removed |
| 10 | v1 | 💥 CRASH | `"use_start_end_token"` in dataset config — belongs in decoding config | Removed |
| 11 | v1 | 💥 CRASH | `"last_epoch": -1` in scheduler — NeMo `CosineAnnealing` has no such param | Removed |
| 12 | v1 | 💥 CRASH | Adapter mode called broken `run_phase()` | Rewritten with clean direct calls |
| 13 | v1 | 💥 CRASH | `import lightning.pytorch as pl` — NeMo stable uses `pytorch_lightning` | `try pytorch_lightning / except lightning.pytorch` |
| 14 | v2 | ⚠️ SILENT | `model.set_trainer(trainer)` never called — NeMo requires it before `setup_optimization()` | Added before all `setup_*` calls |
| 15 | v2 | ⚠️ WRONG | `change_decoding_strategy()` passed fresh OmegaConf config — missing required sub-fields | `copy.deepcopy(model.cfg.decoding)` then modify existing fields |
| 16 | v2 | ⚠️ WRONG | Setup calls before trainer creation — scheduler max_steps computed without dataloader | Correct order: trainer → exp_manager → set_trainer → setup_* → fit |
| 17 | v2 | ⚠️ DEAD CODE | `CTC_LOSS_WEIGHT` defined but never applied (model already has correct value baked in) | Removed |
| 18 | v2 | ⚠️ DEAD CODE | `is_train` parameter in `make_dataset_cfg()` declared but never used inside function | Removed |
| 19 | v3 | 💥 CRASH | `change_decoding_strategy(decoding_cfg)` without `decoder_type` for hybrid model | `_change_decoding()` helper: `if hasattr(model, 'cur_decoder'): change_decoding_strategy(cfg, decoder_type='rnnt')` |
| 20 | v3 | 💥 CRASH | `model.transcribe(batch, batch_size=N)` — old API, crashes on NeMo >= 1.22 | `transcribe_files()` helper: try new `get_transcribe_config()` API, fall back to old |

### `prepare_quran_manifests.py`

| # | Found in | Severity | Bug | Fix |
|---|----------|----------|-----|-----|
| 21 | v1 | 💥 CRASH | `merge_manifests()` called `open()` on files never written in `dry_run` mode | Early return when `dry_run=True` |
| 22 | v1 | ⚠️ DEAD CODE | Unused `from datasets import Dataset` import inside `_process_hf_dataset()` | Removed |
| 23 | v1 | ⚠️ WRONG | Lambda over `val_reciters` — silent fallthrough to `"train"` if row key was missing | Replaced with direct `if row.get("reciter", "") in val_set` check |

---

## Source Verification

Every API in `finetune_quran_asr.py` was verified against real NeMo source code:

| API | Verified in |
|-----|-------------|
| `from nemo.utils.exp_manager import exp_manager` | NeMo/stable/examples/asr/speech_to_text_aed.py |
| `import pytorch_lightning as pl` | NeMo/stable/examples/asr/transcribe_speech.py |
| `model.set_trainer(trainer)` | NeMo/stable/examples/asr/transcribe_speech.py (called before inference) |
| `model.setup_multiple_validation_data(cfg)` | NeMo/stable/nemo/core/classes/modelPT.py |
| `model.setup_training_data(cfg)` | NeMo/stable/nemo/core/classes/modelPT.py |
| `model.setup_optimization(cfg)` | NeMo/stable/nemo/core/classes/modelPT.py |
| `change_decoding_strategy(cfg, decoder_type='rnnt')` | NeMo/stable/examples/asr/transcribe_speech.py line ~195 |
| `if hasattr(asr_model, 'cur_decoder')` | NeMo/stable/examples/asr/transcribe_speech.py line ~192 |
| `model.get_transcribe_config()` + `override_config=` | NeMo/stable/examples/asr/transcribe_speech.py lines ~270-280 |
| `from nemo.collections.asr.metrics.wer import word_error_rate` | NeMo/stable/examples/asr/speech_to_text_eval.py |
| `model.add_adapter(name, cfg)` | NeMo adapter framework |
| `model.save_to(path)` | NeMo ModelPT base class |

---

## Setup

```bash
apt-get install -y libsndfile1 ffmpeg
pip install nemo_toolkit[asr] pytorch-lightning omegaconf
pip install datasets soundfile librosa tqdm
```

---

## Step 1 — Prepare Manifests

```bash
python prepare_quran_manifests.py \
  --output_dir ./manifests \
  --audio_dir  ./quran_audio

# Test without downloading anything
python prepare_quran_manifests.py --dry_run
```

---

## Step 2 — Fine-Tune

```bash
# Standard (any GPU >= 12 GB)
python finetune_quran_asr.py \
  --train_manifest ./manifests/combined_train.json \
  --val_manifest   ./manifests/combined_val.json

# Low VRAM (< 12 GB GPU)
python finetune_quran_asr.py \
  --train_manifest ./manifests/combined_train.json \
  --val_manifest   ./manifests/combined_val.json \
  --adapter_mode --batch_size 4
```

---

## Step 3 — Evaluate

```bash
python finetune_quran_asr.py \
  --eval \
  --eval_model    ./outputs/QuranASR_final/quran_fastconformer_hybrid_pcd.nemo \
  --eval_manifest ./manifests/combined_val.json \
  --beam_size 4
```

---

## Step 4 — Use in Your Server

```python
MODEL_NAME = "./outputs/QuranASR_final/quran_fastconformer_hybrid_pcd.nemo"
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_NAME)
```

---

## Correct NeMo Training Order (verified from source)

```python
# 1. Create trainer first
trainer = pl.Trainer(precision=32, ...)

# 2. Register exp_manager callbacks on the trainer
exp_manager(trainer, cfg)

# 3. Bind trainer to model (BEFORE any setup_* calls)
model.set_trainer(trainer)

# 4. Configure data and optimizer
model.setup_training_data(train_cfg)
model.setup_multiple_validation_data(val_cfg)
model.setup_optimization(optim_cfg)

# 5. Train
trainer.fit(model)
```

---

## GPU Memory Guide

| GPU | VRAM | Mode | `--batch_size` |
|-----|------|------|----------------|
| A100 80GB / H100 | 80 GB | Full fine-tune | 16 (auto) |
| A100 40GB | 40 GB | Full fine-tune | 16 (auto) |
| RTX 4090 / 3090 | 24 GB | Full fine-tune | 12 (auto) |
| RTX 4080 / T4 | 16 GB | Full fine-tune | 8 (auto) |
| RTX 3080 12GB | 12 GB | Full fine-tune | 4 (auto) |
| RTX 3070 / 8 GB | 8 GB | `--adapter_mode` | 4 |
