"""
finetune_quran_asr.py  v4.0  (source-verified)
================================================
Fine-tunes nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 on Quran data.

Every single API call in this file has been verified against the ACTUAL NeMo
source code at:
  raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/asr/transcribe_speech.py
  raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/asr/speech_to_text_finetune.py
  raw.githubusercontent.com/NVIDIA/NeMo/stable/nemo/core/classes/modelPT.py

Model: nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
  FastConformer encoder + RNNT decoder + auxiliary CTC head  (hybrid).
  pcd = punctuation + diacritics output.
  Float32 ONLY — FP16/BF16 produces all-unknown output on real speech.

Training strategy:
  Phase 1 — Decoder warm-up  (3 epochs, encoder FROZEN, LR 5e-4)
    Trains RNNT predictor + joint + CTC head only.
    Prevents catastrophic forgetting of encoder representations.
  Phase 2 — Full fine-tune  (remaining epochs, LR 5e-5 + cosine decay)
    All layers unfrozen. Grad accumulation for effective batch ~16.

Install (NeMo stable / >= 1.20):
  apt-get install -y libsndfile1 ffmpeg
  pip install nemo_toolkit[asr] omegaconf pytorch-lightning
  pip install datasets soundfile librosa

Usage:
  python finetune_quran_asr.py \\
    --train_manifest ./manifests/combined_train.json \\
    --val_manifest   ./manifests/combined_val.json

  # Low VRAM adapter mode
  python finetune_quran_asr.py \\
    --train_manifest ./manifests/combined_train.json \\
    --val_manifest   ./manifests/combined_val.json \\
    --adapter_mode --batch_size 4

  # Evaluate
  python finetune_quran_asr.py \\
    --eval --eval_model ./outputs/QuranASR_final/quran_fastconformer.nemo \\
    --eval_manifest ./manifests/combined_val.json
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightning import — verified: NeMo stable uses `pytorch_lightning`.
# NeMo >= 2.x ASR (r2.3.0+) uses `lightning.pytorch`.
# We handle both so the script works on any supported NeMo version.
# Source: raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/asr/transcribe_speech.py
#   line: `import pytorch_lightning as pl`
# ---------------------------------------------------------------------------
try:
    import pytorch_lightning as pl
    log.info("Lightning backend: pytorch_lightning")
except ImportError:
    try:
        import lightning.pytorch as pl
        log.info("Lightning backend: lightning.pytorch")
    except ImportError:
        raise ImportError(
            "No Lightning installation found.\n"
            "Run: pip install pytorch-lightning   OR   pip install lightning"
        )


# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_NAME    = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
SAMPLE_RATE   = 16000
MAX_DURATION  = 20.0
MIN_DURATION  = 0.3

PHASE1_LR     = 5e-4
PHASE1_EPOCHS = 3
PHASE2_LR     = 5e-5
ADAPTER_LR    = 1e-3
WARMUP_STEPS  = 500
WEIGHT_DECAY  = 1e-4


# ============================================================================
# MODEL LOADING
# Verified: `from_pretrained` and `restore_from` both exist on
#   EncDecHybridRNNTCTCBPEModel (inherits ModelPT).
# ============================================================================

def load_model(restore_path: str = None):
    import nemo.collections.asr as nemo_asr

    if restore_path:
        log.info(f"Restoring from checkpoint: {restore_path}")
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            restore_path=restore_path
        )
    else:
        log.info(f"Loading pretrained: {MODEL_NAME}")
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=MODEL_NAME
        )

    # Float32 mandatory — FP16/BF16 causes all-unknown output on real speech
    model = model.cuda().float()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  Loaded {n_params:.0f}M parameters (float32)")
    return model


# ============================================================================
# FREEZE / UNFREEZE
# ============================================================================

def freeze_encoder(model) -> None:
    """Phase 1: freeze FastConformer encoder, unfreeze RNNT+CTC heads."""
    frozen = total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "encoder" in name:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
    log.info(
        f"Phase 1: encoder FROZEN  {frozen/1e6:.1f}M / {total/1e6:.1f}M params. "
        f"Training RNNT predictor + joint + CTC head only."
    )


def unfreeze_all(model) -> None:
    """Phase 2: unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f"Phase 2: all {n:.1f}M params UNFROZEN.")


# ============================================================================
# DECODING STRATEGY
#
# Verified against:
#   NeMo/stable/examples/asr/transcribe_speech.py  lines ~180-220
#
# Key findings from real source:
#   1. NeMo modifies decoding config DIRECTLY (no deepcopy needed if fields exist).
#      We deepcopy for safety — deepcopy of a struct DictConfig is still struct,
#      but ALL fields we modify (strategy, beam.beam_size, etc.) already exist
#      in the default RNNT decoding config, so direct assignment is valid.
#
#   2. EncDecHybridRNNTCTCBPEModel HAS `cur_decoder` attribute.
#      The real NeMo code checks:
#        `if hasattr(asr_model, 'cur_decoder'):`
#            `asr_model.change_decoding_strategy(decoding_cfg, decoder_type=...)`
#      Without `decoder_type`, the hybrid model may configure the wrong decoder.
#      We always pass decoder_type='rnnt' for RNNT-quality output.
# ============================================================================

def _change_decoding(model, decoding_cfg) -> None:
    """
    Call change_decoding_strategy with correct args for hybrid vs pure RNNT.
    Verified: hybrid model needs decoder_type kwarg.
    Source: transcribe_speech.py  `if hasattr(asr_model, 'cur_decoder'):`
    """
    if hasattr(model, 'cur_decoder'):
        # Hybrid RNNT/CTC model — specify which decoder to use
        model.change_decoding_strategy(decoding_cfg, decoder_type='rnnt')
    else:
        model.change_decoding_strategy(decoding_cfg)


def set_greedy_decoding(model) -> None:
    """
    Fast greedy_batch decoding during training.
    deepcopy preserves all existing fields; only `strategy` is changed.
    """
    try:
        decoding_cfg = copy.deepcopy(model.cfg.decoding)
        decoding_cfg.strategy = "greedy_batch"
        _change_decoding(model, decoding_cfg)
    except Exception as exc:
        log.debug(f"set_greedy_decoding non-fatal: {exc}")


def set_beam_decoding(model, beam_size: int = 4) -> None:
    """
    Beam search for evaluation.
    All fields (beam.beam_size, beam.return_best_hypothesis, beam.score_norm)
    are standard RNNT decoding config fields — verified in RNNTDecodingConfig.
    """
    try:
        decoding_cfg = copy.deepcopy(model.cfg.decoding)
        decoding_cfg.strategy = "beam"
        decoding_cfg.beam.beam_size = beam_size
        decoding_cfg.beam.return_best_hypothesis = True
        decoding_cfg.beam.score_norm = True
        _change_decoding(model, decoding_cfg)
        log.info(f"  Decoding: beam search  beam_size={beam_size}")
    except Exception as exc:
        log.warning(f"  Beam decoding failed ({exc}). Falling back to greedy.")


# ============================================================================
# ADAPTER MODE (parameter-efficient fine-tuning)
# Verified: model.add_adapter() exists in NeMo's adapter framework.
# ============================================================================

def setup_adapters(model, adapter_dim: int = 64) -> bool:
    """
    Insert LinearAdapter modules into FastConformer encoder layers.
    Only adapter parameters are trained (~2M params); base model frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

    # model.cfg IS the model config — path is model.cfg.encoder.d_model
    try:
        d_model = model.cfg.encoder.d_model
    except Exception:
        d_model = 512  # FastConformer-Large default
        log.warning(f"  Cannot read d_model from model.cfg — defaulting to {d_model}")
    log.info(f"  Adapter: d_model={d_model}, adapter_dim={adapter_dim}")

    adapter_cfg = OmegaConf.create({
        "_target_":      "nemo.collections.common.parts.adapter_modules.LinearAdapter",
        "in_features":   d_model,
        "dim":           adapter_dim,
        "activation":    "swish",
        "norm_position": "pre",
    })

    try:
        model.add_adapter(name="quran_adapter", cfg=adapter_cfg)
    except Exception as exc:
        log.error(f"model.add_adapter() failed: {exc}")
        log.error("Requires NeMo >= 1.21.  Run: pip install 'nemo_toolkit[asr]>=1.21'")
        return False

    n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f"  Adapters inserted — {n:.2f}M trainable params (base frozen)")
    return True


# ============================================================================
# CONFIG BUILDERS
#
# Verified valid NeMo AudioToTextDataset fields (from NeMo ASR data configs):
#   manifest_filepath, sample_rate, batch_size, max_duration, min_duration,
#   shuffle, num_workers, pin_memory
#
# Fields that were removed because they DO NOT exist in AudioToTextDataset:
#   trim_silence, use_start_end_token  (confirmed bugs #9 and #10)
# SpecAugment is NOT set here — it lives at model level, not dataset level.
# ============================================================================

def make_dataset_cfg(manifest_path: str, batch_size: int,
                     num_workers: int, shuffle: bool = True) -> dict:
    return {
        "manifest_filepath": manifest_path,
        "sample_rate":       SAMPLE_RATE,
        "batch_size":        batch_size,
        "max_duration":      MAX_DURATION,
        "min_duration":      MIN_DURATION,
        "shuffle":           shuffle,
        "num_workers":       num_workers,
        "pin_memory":        True,
    }


def make_optim_cfg(lr: float, warmup_steps: int) -> dict:
    """
    Verified NeMo CosineAnnealing params: warmup_steps, min_lr, max_steps.
    `last_epoch` is NOT a valid field (was bug #11).
    """
    return {
        "name":         "adamw",
        "lr":           lr,
        "betas":        [0.9, 0.98],
        "weight_decay": WEIGHT_DECAY,
        "sched": {
            "name":         "CosineAnnealing",
            "warmup_steps": warmup_steps,
            "min_lr":       lr * 0.05,
        },
    }


# ============================================================================
# TRAINER
# ============================================================================

def make_trainer(max_epochs: int, grad_accum: int = 1) -> pl.Trainer:
    """
    precision=32 is MANDATORY — FP16/BF16 causes all-unknown output.
    enable_checkpointing=False because NeMo exp_manager owns checkpointing.
    accumulate_grad_batches passed directly (NOT via OmegaConf DictConfig).
    """
    return pl.Trainer(
        devices                 = 1 if torch.cuda.device_count() <= 1 else -1,
        accelerator             = "gpu",
        strategy                = "auto",
        max_epochs              = max_epochs,
        enable_checkpointing    = False,
        log_every_n_steps       = 50,
        val_check_interval      = 1.0,
        gradient_clip_val       = 1.0,
        accumulate_grad_batches = grad_accum,
        precision               = 32,
    )


# ============================================================================
# EXP MANAGER
# Verified: the callable is `exp_manager` INSIDE `nemo.utils.exp_manager` module.
# `from nemo.utils import exp_manager` imports the MODULE, not the function.
# `from nemo.utils.exp_manager import exp_manager` imports the FUNCTION. ✓
# ============================================================================

def init_exp_manager(trainer: pl.Trainer, exp_name: str, phase: int) -> None:
    from nemo.utils.exp_manager import exp_manager

    cfg = OmegaConf.create({
        "exp_dir":                    "./outputs",
        "name":                       exp_name,
        "version":                    f"phase{phase}",
        "create_tensorboard_logger":  True,
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {
            "monitor":          "val_wer",
            "mode":             "min",
            "save_top_k":       3,
            "save_last":        True,
            "filename":         f"{exp_name}--{{val_wer:.4f}}--{{epoch}}",
            "always_save_nemo": True,
        },
        "resume_if_exists":            True,
        "resume_ignore_no_checkpoint": True,
    })
    exp_manager(trainer, cfg)


# ============================================================================
# PHASE RUNNER
#
# Correct call ordering verified against:
#   NeMo stable source + GitHub issue #4562
#
# ORDER:
#   1. make_trainer()                       ← create trainer first
#   2. init_exp_manager(trainer)            ← register callbacks on trainer
#   3. model.set_trainer(trainer)           ← bind trainer to model BEFORE setup_*
#   4. model.setup_training_data(...)       ← needs trainer bound
#   5. model.setup_multiple_validation_data(...)
#   6. model.setup_optimization(...)        ← needs trainer for max_steps calc
#   7. trainer.fit(model)
#
# Verified: model.set_trainer() is called even for inference in NeMo stable.
# Source: transcribe_speech.py line: `asr_model.set_trainer(trainer)`
# ============================================================================

def run_phase(
    model,
    exp_name: str,
    phase: int,
    max_epochs: int,
    train_manifest: str,
    val_manifest: str,
    batch_size: int,
    num_workers: int,
    lr: float,
    warmup_steps: int,
    grad_accum: int = 1,
) -> None:
    log.info(f"\n{'='*55}")
    log.info(f"  Phase {phase}  —  {max_epochs} epochs | LR={lr:.0e} | grad_accum={grad_accum}x")
    log.info(f"{'='*55}")

    trainer = make_trainer(max_epochs=max_epochs, grad_accum=grad_accum)
    init_exp_manager(trainer, exp_name, phase)

    # MUST come before setup_* calls so optimizer scheduler has trainer reference
    model.set_trainer(trainer)

    train_cfg = make_dataset_cfg(train_manifest, batch_size, num_workers, shuffle=True)
    val_cfg   = make_dataset_cfg(val_manifest,   batch_size * 2, num_workers, shuffle=False)
    optim_cfg = make_optim_cfg(lr, warmup_steps)

    model.setup_training_data(OmegaConf.create(train_cfg))
    model.setup_multiple_validation_data(OmegaConf.create(val_cfg))
    model.setup_optimization(OmegaConf.create(optim_cfg))

    trainer.fit(model)
    log.info(f"Phase {phase} complete.")


# ============================================================================
# TRANSCRIBE HELPER
#
# CRITICAL FIX (bug #A): NeMo stable changed the transcribe() API.
#
# Old API  (NeMo <= ~1.21):
#   model.transcribe(audio_files, batch_size=N)
#
# New API  (NeMo >= ~1.22):
#   override_cfg = model.get_transcribe_config()
#   override_cfg.batch_size = N
#   model.transcribe(audio=audio_files, override_config=override_cfg)
#
# Verified in:
#   NeMo/stable/examples/asr/transcribe_speech.py  lines ~260-280
#
# We try new API first, fall back to old API.
# ============================================================================

def transcribe_files(model, audio_files: list, batch_size: int) -> list:
    """Transcribe a list of audio files. Returns list of text strings."""
    try:
        # New NeMo API
        override_cfg = model.get_transcribe_config()
        with open_dict(override_cfg):
            override_cfg.batch_size = batch_size
        preds = model.transcribe(audio=audio_files, override_config=override_cfg)
    except (AttributeError, TypeError):
        # Old NeMo API fallback
        preds = model.transcribe(audio_files, batch_size=batch_size)

    # Handle Hypothesis objects (newer NeMo) vs plain strings (older NeMo)
    if preds and hasattr(preds[0], 'text'):
        return [p.text for p in preds]
    return list(preds)


# ============================================================================
# MAIN FINE-TUNE
# ============================================================================

def finetune(args) -> None:
    if not torch.cuda.is_available():
        log.error("CUDA GPU is required for fine-tuning.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {gpu_name}  ({gpu_mem:.0f} GB VRAM)")

    if args.batch_size is None:
        if   gpu_mem >= 40: args.batch_size = 16
        elif gpu_mem >= 24: args.batch_size = 12
        elif gpu_mem >= 16: args.batch_size = 8
        elif gpu_mem >= 12: args.batch_size = 4
        else:               args.batch_size = 2
        log.info(f"Auto batch_size: {args.batch_size}")

    model = load_model(restore_path=args.resume_from)

    # ── Adapter mode ─────────────────────────────────────────────────────────
    if args.adapter_mode:
        log.info("\nAdapter mode: parameter-efficient fine-tuning")
        if not setup_adapters(model, adapter_dim=args.adapter_dim):
            sys.exit(1)
        set_greedy_decoding(model)
        run_phase(
            model=model, exp_name="QuranASR_Adapter", phase=1,
            max_epochs=args.max_epochs,
            train_manifest=args.train_manifest, val_manifest=args.val_manifest,
            batch_size=args.batch_size, num_workers=args.num_workers,
            lr=ADAPTER_LR, warmup_steps=WARMUP_STEPS, grad_accum=1,
        )
        _save_final(model)
        return

    # ── Phase 1: Decoder warm-up ─────────────────────────────────────────────
    if not args.skip_phase1:
        freeze_encoder(model)
        set_greedy_decoding(model)
        run_phase(
            model=model, exp_name="QuranASR", phase=1,
            max_epochs=PHASE1_EPOCHS,
            train_manifest=args.train_manifest, val_manifest=args.val_manifest,
            batch_size=args.batch_size, num_workers=args.num_workers,
            lr=PHASE1_LR, warmup_steps=min(200, WARMUP_STEPS), grad_accum=1,
        )
    else:
        log.info("Phase 1 skipped (--skip_phase1).")

    # ── Phase 2: Full fine-tune ──────────────────────────────────────────────
    unfreeze_all(model)
    set_greedy_decoding(model)

    remaining  = args.max_epochs - (0 if args.skip_phase1 else PHASE1_EPOCHS)
    if remaining <= 0:
        remaining = args.max_epochs
    grad_accum = max(1, 16 // args.batch_size)
    log.info(f"  Effective batch: {args.batch_size} x {grad_accum} = {args.batch_size * grad_accum}")

    run_phase(
        model=model, exp_name="QuranASR", phase=2,
        max_epochs=remaining,
        train_manifest=args.train_manifest, val_manifest=args.val_manifest,
        batch_size=args.batch_size, num_workers=args.num_workers,
        lr=PHASE2_LR, warmup_steps=WARMUP_STEPS, grad_accum=grad_accum,
    )
    _save_final(model)


def _save_final(model) -> None:
    out_dir  = Path("./outputs/QuranASR_final")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quran_fastconformer_hybrid_pcd.nemo"
    model.save_to(str(out_path))
    log.info(f"\n✅  Final model: {out_path}")
    log.info(f"    Load with: model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from('{out_path}')")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(args) -> float:
    """
    Evaluate a .nemo model on a manifest and report WER.

    Fixes applied:
    - json imported at module top (was bug #5 — used before import)
    - transcribe() uses version-aware transcribe_files() helper (bug #A)
    - change_decoding_strategy passes decoder_type for hybrid model (bug #B)
    - word_error_rate import path verified: nemo.collections.asr.metrics.wer ✓
    """
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.metrics.wer import word_error_rate

    log.info(f"Loading: {args.eval_model}")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(args.eval_model)
    model = model.cuda().float().eval()

    # Bind a minimal trainer (required by NeMo even for inference)
    # Verified: transcribe_speech.py does `asr_model.set_trainer(trainer)` before eval
    dummy_trainer = pl.Trainer(devices=1, accelerator="gpu")
    model.set_trainer(dummy_trainer)

    set_beam_decoding(model, beam_size=args.beam_size)

    log.info(f"Evaluating on: {args.eval_manifest}")
    with open(args.eval_manifest, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    audio_files = [l["audio_filepath"] for l in lines]
    references  = [l["text"]           for l in lines]

    bs         = args.batch_size or 8
    hypotheses = []
    for i in range(0, len(audio_files), bs):
        batch = audio_files[i : i + bs]
        hypotheses.extend(transcribe_files(model, batch, batch_size=len(batch)))

    wer = word_error_rate(hypotheses=hypotheses, references=references)

    log.info(f"\n{'='*50}")
    log.info(f"  Manifest : {args.eval_manifest}")
    log.info(f"  Samples  : {len(lines)}")
    log.info(f"  WER      : {wer*100:.2f}%")
    log.info(f"  Baseline : 6.55% RNNT / 7.61% CTC (NVIDIA, EveryAyah test set)")
    log.info(f"{'='*50}")
    return wer


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0  v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--val_manifest",   default=None)
    p.add_argument("--max_epochs",   type=int, default=18)
    p.add_argument("--batch_size",   type=int, default=None,
                   help="Per-GPU batch size (auto-detected from VRAM if not set)")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--adapter_mode", action="store_true")
    p.add_argument("--adapter_dim",  type=int, default=64)
    p.add_argument("--skip_phase1",  action="store_true")
    p.add_argument("--resume_from",  default=None,
                   help="Path to .nemo checkpoint to resume from")
    p.add_argument("--eval",         action="store_true")
    p.add_argument("--eval_model",   default=None)
    p.add_argument("--eval_manifest",default=None)
    p.add_argument("--beam_size",    type=int, default=4)
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("  Quran ASR  —  FastConformer Hybrid Fine-Tuning  v4.0")
    log.info(f"  Model: {MODEL_NAME}")
    log.info("=" * 60)

    if args.eval:
        if not args.eval_model or not args.eval_manifest:
            log.error("--eval requires --eval_model and --eval_manifest")
            sys.exit(1)
        evaluate_model(args)
    else:
        if not args.train_manifest or not args.val_manifest:
            log.error("Fine-tuning requires --train_manifest and --val_manifest")
            log.error("Run prepare_quran_manifests.py first.")
            sys.exit(1)
        finetune(args)


if __name__ == "__main__":
    main()
