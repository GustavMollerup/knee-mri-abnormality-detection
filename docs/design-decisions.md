# Design decisions

Short records of the main choices: what was chosen, what else was possible, and why.

---

### 1. Weak labels from reports instead of training only on expert labels
**Choice:** derive a soft label + confidence for every training study from its report, and train on both sources.
**Alternatives:** train on the expert-labelled subset only; use reports as model input.
**Why:** the expert subset is too small for 12 targets. Reports exist for every training study but not at test time, so they can only be a teacher, never an input.

### 2. Rule-based extractor instead of a language model
**Choice:** multilingual regex lexicons with negation, uncertainty, severity and grade rules.
**Alternatives:** an LLM, or a fine-tuned multilingual text classifier.
**Why:** runs offline inside Kaggle's time limit (≈30 s for 4,400 reports), every decision can be traced to one rule, and each rule can be switched off (`FEATURES`) and measured against the expert labels. An LLM is the natural next step, with the rules kept as a baseline.

### 3. Separate weights for expert and report labels
**Choice:** expert labels get weight 3.0; report labels `0.25 + 0.75 × confidence`.
**Why:** a report that only mentions a structure in passing should push the model less than one that describes a finding in three places. The floor of 0.25 keeps silent reports in the data instead of throwing them away.

### 4. Six named slots instead of "all series"
**Choice:** pick one series per slot (plane × fat suppression × fluid sensitivity) and mask missing slots.
**Alternatives:** feed every series; pick one series per plane.
**Why:** studies have very different numbers of series. Fixed slots give the model a consistent meaning per input position, and the mask handles missing protocols without inventing data.

### 5. Physical cropping (130 mm) instead of pixel cropping
**Why:** scanners use different fields of view and pixel spacings. Cropping in millimetres means a meniscus has roughly the same size in pixels across scanners.

### 6. Mirroring right knees
**Why:** medial and lateral are separate targets. Mirroring puts the medial side in the same place for every knee, so the model does not need to learn both orientations. For sagittal series the medial–lateral axis is the slice direction, so the slice order is reversed instead of flipping the image.

### 7. DINOv2-small with partial fine-tuning
**Choice:** self-supervised ViT, last 6 of 12 blocks trainable.
**Alternatives:** ImageNet CNN (e.g. ResNet/EfficientNet), a 3D CNN, full fine-tuning.
**Why:** DINOv2 features transfer well to grey-scale medical images; the small variant fits two T4 GPUs and the time budget; freezing half the network limits overfitting on noisy labels.

### 8. 2.5D input: three slices as RGB channels
**Why:** reuses the pretrained 3-channel input layer while giving the model a little through-plane context. A full 3D model would need far more memory and data.

### 9. Attention over slots with an anatomical prior
**Choice:** one learned query per target attending over the slot features, plus an optional fixed bias towards the planes a radiologist would use.
**Alternatives:** concatenate all slot features; average them.
**Why:** different findings are best seen on different series (ACL on sagittal, MCL on coronal). Per-target attention lets each target choose; the prior is a sensible starting point when data is limited.

### 10. Leakage-safe holdout by report hash
**Why:** some reports are duplicated across studies. Splitting by a hash of the report text keeps identical reports on the same side of the split, so the holdout score is not inflated.

### 11. Rank averaging in the ensemble
**Why:** the metric is ROC-AUC, which only depends on ordering. Ranks put members with different calibration on the same scale before averaging.

### 12. Per-target pooling over slice windows
**Why:** a fracture visible on one slice is a fracture (`max`); effusion is spread out (`mean`). Pooling the same way for all targets would either miss focal findings or over-react to noise.

### 13. Checkpoint fingerprints
**Why:** a checkpoint can load without errors and still compute something different if preprocessing or architecture changed. Comparing outputs on a fixed random input catches that before a wrong model reaches the submission.

### 14. Fail safe, but fail loud where it matters
**Choice:** write a valid 0.5 submission first and fall back to it on unexpected errors, but stop the run on `LabelSourceError`.
**Why:** a crash late in an 8-hour run should still give a scoreable file. A broken label source is different: continuing would train a worse model and hide it in a log line.

### 15. Time-budget scheduler and memory planning
**Why:** Kaggle kills the notebook at the time limit, and RAM differs between sessions. Sizing the cache from `MemAvailable` and scheduling members from measured timings keeps the run inside both limits and degrades gracefully instead of failing.
