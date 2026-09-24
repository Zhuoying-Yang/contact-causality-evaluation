# Contact Causality Evaluation

Evaluation of plausible robot-object contact in generated manipulation videos.

## Final main method

**FuturePair36**

Main implementation:

`vlm_contact_oracle9/whole_video_contact_event_v2/run_contact_event_v2.py`

Pipeline:

1. Geometry proposes likely contact moments.
2. Qwen3-VL-8B evaluates plausible contact support.
3. Candidate must have object-response evidence within the following 36 frames.
4. The highest eligible contact-event score becomes the video score.

FuturePair36 is used because it is more robust to noisy automatic response-onset localization than hard backward gating.

## Final corrected evaluation

Script:

`final_results/evaluate_final_corrected.py`

Metrics:

`final_results/CONTACT_FINAL_METRICS.csv`

Ground-truth corrections:

`final_results/FINAL_GT_CORRECTIONS.csv`

Main results:

- FuturePair36, full expanded evaluation: AUROC 0.8100, coverage 30/32
- FuturePair36, common 25-case comparison: AUROC 0.8750
- Original Backward: AUROC 0.7193
- Multi-Episode Backward: AUROC 0.7427
- Strong 1.5x Backward: AUROC 0.7778
- Strong 2.0x Backward: AUROC 0.7647
- Local pre/post causal ablation: AUROC 0.7295

Corrected labels:

- Cosmos3_seed101_0035 -> NO_CONTACT
- Cosmos3_seed103_0037 -> CONTACT

## Important limitation

FuturePair36 does not strictly enforce that object motion begins after contact.

Its main failure mode is:

object moves first -> visually valid contact occurs shortly afterward -> motion continues

`Cosmos2.5_0006` is the main example.

Therefore FuturePair36 should be interpreted as a robust video-level contact plausibility score, not a precise causal-onset detector.

## Ablations

- `backward36_all38_v1`
- `backward36_multi_episode_v1`
- `backward36_strong15_v1`
- `backward36_strong20_v1`
- `local_causal_pair_v1`
