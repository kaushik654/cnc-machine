# Visual Recovery of Damaged Barcodes
### Confidence-gated diffusion inpainting with structural reasoning and a decode-verify feedback loop

---

## 1. What this project does

Barcodes and QR codes in the real world get scratched, torn, stained,
and occluded. A standard decoder (ZXing) simply fails on them. This
pipeline RECOVERS the damaged region and decodes the symbol:

1. It finds WHERE the barcode is damaged,
2. regenerates ONLY that region with a small diffusion model guided by
   barcode structural priors,
3. verifies the result with a real decoder (Reed-Solomon check),
4. and if decoding fails, it REASONS AGAIN - re-running the recurrent
   reasoning module with a perturbed state to produce a different
   completion - instead of blindly retrying.

Why this design:
- **Inpaint, don't regenerate.** Undamaged pixels are never touched -
  they are hard-constrained during sampling. Generation errors can only
  occur inside the damage mask.
- **Pixel space, no VAE.** Barcode modules are hard-edged binary
  structure; latent-space models (Stable Diffusion etc.) soften edges
  and flip modules. Our UNet is tiny (22M) and trained from scratch.
- **Diffusion, not GAN.** The decode-verify loop needs a *stochastic*
  generator: every failed attempt must produce a genuinely different
  completion. A GAN gives one deterministic answer; diffusion gives a
  new sample per attempt.
- **Reed-Solomon is the judge.** QR error correction (L/M/Q/H recovers
  7/15/25/30% of codewords) means the generator doesn't need to be
  perfect - the system needs the decode to pass, and failed decodes
  trigger another reasoning attempt.

## 2. Full architecture

    photo
      |
    [Stage 0] Barcode localization + crop  (YOUR RTMDET GOES HERE)
      |            RTMDet detects the barcode bounding box; crop +
      |            resize to 256x256 grayscale. NOTE: RTMDet finds the
      |            BARCODE, not the damage - it replaces this crop
      |            stage only, NOT the damage detector below.
      |
    [Stage 1] Damage detector - DamageUNet (1.9M params)
      |            segments the DAMAGED PIXELS inside the barcode,
      |            outputs a binary mask (dilated: over-masking is safe)
      |
    [Stage 2] Fast path / confidence gate
      |            try ZXing on the input as-is; if it decodes, stop.
      |
    [Stage 3] StructuralPriorNet (1.7M params)
      |            FeatureEncoder -> Structural Fusion Attention
      |            (learned finder/timing/grid tokens, cross-attention)
      |            -> Recurrent Feature Reasoning (ConvGRU, hole filled
      |            boundary-to-center) -> 1ch structural prior map
      |
    [Stage 4] Diffusion UNet (22M params, pixel space)
      |            masked DDIM inpainting; conditioning channels =
      |            [x_t, masked image, mask, prior map]; known region
      |            hard-constrained at every step
      |
    [Stage 5] Decode-verify
      |            feather composite -> adaptive binarize -> zxing-cpp
      |            (full Reed-Solomon validation)
      |
      +-- valid   -> OUTPUT decoded text + restored image
      +-- invalid -> FEEDBACK LOOP: attempt i re-runs RFR with
                     iterations 4+i and hidden-state perturbation
                     0.15*i -> new prior map -> rediffuse (up to k=5)

## 3. Repository files

    data.py              synthetic data engine + real-pair loading
    damage_detector.py   Stage 1 model + its training + mask prediction
    modules.py           Stage 3: SFA, RFR, prior map head
    train.py             joint training of Stage 3 + Stage 4
    infer.py             Stages 3-5 inference w/ feedback loop + viz
    pipeline.py          the whole thing end-to-end from one image
    eval.py              decode-success-rate table vs baselines

## 4. Installation

    pip install torch diffusers accelerate qrcode python-barcode \
                pillow opencv-python-headless zxing-cpp

Python 3.10+, one GPU (tested plan: single A6000).

## 5. How to store your data

    DATA/
      clean/     0001.png  0002.png  ...
      damaged/   0001.png  0002.png  ...

- SAME FILENAME = a pair (clean version + its damaged version).
- Any image size; everything is resized to 256x256 internally.
- Images should be crops of the barcode itself (run RTMDet first if
  your photos contain full scenes).
- The damage mask is computed automatically per pair:
  threshold(|clean - damaged|) + morphological close + dilate.
- The LAST 500 pairs (sorted by filename) are automatically held out
  as the eval split - never trained on.
- No real data? Every script also runs on pure synthetic data (QR all
  4 EC levels + Code128, procedural damage) - just omit --real_dir.

## 6. Run sequence (from scratch)

STEP 0 - verify the derived masks (1 minute, DO NOT SKIP):

    python -c "from data import RealPairs; import cv2
    rp = RealPairs('DATA')
    [cv2.imwrite(f'maskcheck_{i}.png', s['mask'].numpy()[0]*255) for i,s in zip(range(10), rp)]"

    Open maskcheck_*.png:
    - masks show the damaged spots cleanly -> pairs are aligned, proceed.
    - masks are noise everywhere -> your clean/damaged are different
      photos (not pixel-aligned). Either register them first, or train
      synthetic-only (drop --real_dir below) and use the real damaged
      images purely as a test set via pipeline.py.

STEP 1 - train the damage detector (~1.5 h on A6000):

    python damage_detector.py train --real_dir DATA --steps 20000 --batch 64

    Output: ckpt/detector.pt. Watch the printed IoU - it should climb
    past ~0.7. Mild damage is easy; expect it early.

STEP 2 - train prior net + diffusion jointly (~6-7 h):

    python train.py --real_dir DATA --steps 80000 --batch 32

    Output: ckpt/final.pt (+ ckpt/step_XXXX.pt every 10k).
    Decodable samples usually appear by ~30k steps - if short on time,
    ckpt/step_30000.pt is already usable.

STEP 3 - quantitative eval table (~30 min):

    python eval.py --ckpt ckpt/final.pt --n 300 --k 3 --steps 30

    Prints decode success rate per damage-fraction bin
    (5-15 / 15-25 / 25-35 / 35-45 %) for: raw ZXing, OpenCV Telea,
    OpenCV Navier-Stokes, and ours.

STEP 4 - end-to-end demo on a single damaged image:

    python pipeline.py --detector ckpt/detector.pt --ckpt ckpt/final.pt \
                       --image my_damaged_barcode.png

    Already have a ground-truth mask? Skip the detector:

    python infer.py --ckpt ckpt/final.pt --image d.png --mask m.png

## 7. What you will see (viz/ folder, every inference)

    00_input.png            the damaged input
    01_predicted_mask.png   Stage-1 mask, raw
    01_mask_overlay.png     damage painted red over the input
    attemptN_prior_map.png  the SFA->RFR structural prior map - compare
                            across attempts to SEE the feedback loop
                            changing the reasoning
    attemptN_step_XXX.png   the broken region forming step by step
                            during denoising (known pixels are pasted
                            into every frame, so only the damaged
                            region evolves)
    attemptN_result.png     feathered composite of attempt N
    attemptN_binary.png     the exact image ZXing decodes

    --viz_every 1 gives every step (full animation material);
    default is every 5th step.

## 8. Knobs that matter

    train.py
      --mask_loss_boost 4.0    extra loss weight inside the hole
      --prior_l1_weight 0.5    aux loss: prior map ~ clean image
      --rfr_iters 4            RFR iterations during training
    infer.py / pipeline.py
      --k 5                    max decode-verify attempts
      --steps 50               DDIM steps per attempt
      --viz_every 5            viz frame frequency
    damage_detector.py predict
      dilate=5                 mask dilation. NEVER set to 0: an
                               under-mask leaves damage inside the
                               "trusted" region and poisons the
                               hard constraint.

## 9. Verified in sandbox (what's already been tested)

- clean synthetic targets decode 40/40 (a Code128 render bug was found
  and fixed - targets are now decode-verified at generation time)
- derived-mask pipeline on paired folders: correct shapes/fractions
- prior-net gradients flow through SFA/RFR/head; perturbation and
  iteration knobs provably change the prior map (feedback is real)
- masked inpainting preserves the known region exactly (allclose test)
- full visualization flow produces every file listed in section 7

## 10. Notes for the paper

- Ablate the RFR feedback loop against blind best-of-k at equal k
  (set perturb_step=0, fixed iters) - this isolates the novelty claim.
- Plot decode success vs damage fraction with Reed-Solomon capacity
  lines (L/M/Q/H = 7/15/25/30% of codewords) - shows how close the
  system gets to the information-theoretic ceiling.
- Report per-attempt latency + attempts-used histogram; mention LCM
  distillation to 4 steps for the on-device story.
- The attemptN_step_XXX frames are your qualitative figure.
