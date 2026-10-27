# Pipeline walkthrough

This document follows one study from raw files to a prediction, and names the notebook section that does each step.

## 0. Run modes

`main()` has two paths:

| Path | When | What happens |
|---|---|---|
| **Inference only** | A weights package (`manifest.json` + checkpoints) is attached | Decode test images, run every ensemble member, combine by rank |
| **Train from scratch** | No package attached | Header pass, labels, cache, train, predict |

On top of that, a **smoke mode** switches on automatically when `test.csv` has ≤ 10 rows (which is what Kaggle runs when a version is saved). It limits the run to 60 studies and 1 epoch.

## 1. Labels (notebook section 1, `src/knee_mri/report_labels.py`)

```
report ──► normalise ──► unwrap ──► split into clauses ──► per target:
                                                           find anatomy  ─┐
                                                           find finding  ─┼─► polarity × weight ──► soft score + confidence
                                                           negation/unc. ─┘
```

* **Soft score.** No mention gives 0.28 with almost zero confidence; an explicit negative gives ≈0.16; a clear positive goes up to 0.97. "Not mentioned" and "explicitly normal" are kept apart on purpose.
* **Confidence** grows with the number of supporting clauses and becomes the per-sample loss weight.
* **Merging with a label table.** If a pre-computed table (`report_labels*.csv`) is attached, it overrides the in-notebook extractor for the studies it covers. If a label dataset is attached but broken, the run stops (`LabelSourceError`) instead of silently using weaker labels.

Final training targets:

| Source | Target | Loss weight |
|---|---|---|
| Expert label | 0 / 1 | 3.0 |
| Report label | soft score | 0.25 + 0.75 × confidence |

## 2. DICOM (notebook section 4)

1. **Header pass.** One header per series, read in 16 threads without pixel data.
2. **Classification.** Fat suppression from description keywords and `ScanOptions`; weighting from the description, else from TR/TE (TR < 800 ms → T1, TE > 60 ms → T2, otherwise PD).
3. **Slot picking.** For each of the 6 slots, the matching series with the most slices.
4. **Ordering.** Slice position along the normal of the image plane; `InstanceNumber` as fallback. Has its own time budget and an optional JSON cache.
5. **Decoding.** Slices from the central 20–80 % of the stack; rescale slope/intercept; 130 mm centre crop via `PixelSpacing`; 1–99 percentile normalisation; bilinear resize; stored as uint8.
6. **Laterality.** Tag first, geometry second. Right knees mirrored (horizontal flip for coronal/axial, slice order reversed for sagittal).
7. **Cache.** `(studies, 6 slots, slices, H, W)` uint8 plus a `(studies, 6)` mask. Size planned from `MemAvailable` before decoding.

At 224 px and 130 mm, one pixel is 0.58 mm and one ViT patch (14 px) covers about 8 mm.

## 3. Model (notebook section 5)

```
per slot: 3 slices as RGB ──► DINOv2-small ──► [CLS ‖ mean(patches)]  (768-d)
6 slot vectors + slot embeddings ──► SlotHead
    for each target: query · slots ──(+ anatomical prior, mask)──► softmax ──► weighted sum ──► logit
```

Only the last 6 of 12 transformer blocks and the final layer norm are trained.

## 4. Training (notebook sections 7–8)

* Batches of 4 studies, augmentation on the GPU.
* AdamW, two learning rates (backbone 8e-6, head 1e-3), weight decay 0.02, OneCycle with 15 % warm-up, mixed precision.
* Holdout: `md5(report text) mod 5 == 0`. Identical reports always fall in the same split.
* Each epoch logs macro AUC on the holdout and on the expert-labelled studies inside the holdout. The best epoch (on the holdout) is kept.

## 5. Inference (notebook section 6)

* One worker thread per GPU pulls ensemble members from a queue.
* After loading, each member's **fingerprint** (output on a fixed random input) must match the stored one within 0.002.
* Sliding windows over the cached slices; per-target pooling across windows (`max`, top-2 mean, or mean).
* The scheduler measures load time and time per window, and shrinks the number of windows or stops adding members when the 8-hour budget runs short.
* A failed member is retried on the other GPU, then dropped ("costs one vote, not the run").
* Members are combined by **rank averaging**. `submission.csv` is rewritten after every member, so a timeout still leaves a valid partial ensemble.

## 6. Safety net

1. An all-0.5 submission is written before anything else.
2. Any unexpected exception writes the all-0.5 fallback (except `LabelSourceError`, which is meant to stop the run).
3. Every submission file is validated: column order, row count, unique IDs, same ID set as `test.csv`, finite values.
