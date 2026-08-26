# CNN-LSTM Hybrid Model for Swimming Technique Fatigue Analysis

A hybrid deep learning pipeline that estimates frame-by-frame technique-induced fatigue in the 100m freestyle from race footage. A CNN extracts swimmer pose (keypoints) from video frames, and those pose sequences are used to derive a custom fatigue score which an LSTM learns to refine over time — producing a fatigue curve for a swim instead of a single end-of-race number.

This project was written up as an AP Research paper (*CNN-LSTM Hybrid Deep Learning Model for Technique Analysis of Elite Swimmers in the 100m Freestyle*), which covers the full literature review, methodology justification, and findings/limitations in more depth than this README does. If paper is not linked it is in progress for revision 

## Motivation

The standard way swimmers manage race fatigue is **tapering** — reducing training load before a competition so the body recovers. Tapering is tuned to a swimmer's physical condition, but it doesn't say anything about *where in a race their technique breaks down*. This project's goal was to see whether pose data extracted from video, combined with a hand-derived fatigue metric and a sequence model, could produce a fatigue-over-time curve that a swimmer or coach could actually act on — as a complement to (not a replacement for) tapering.

## Pipeline Overview

```
Race/training footage (video)
        │
        ▼
CNN pose estimator (ResNet-18)  →  pose_cnn_model.pth      [CNN.py]
        │  (17 keypoints per frame)
        ▼
Pseudo fatigue labels (hand-derived composite equation)
        │
        ▼
LSTM trained on CNN features → pseudo labels  →  fatigue_lstm_model.pth   [LSTM.py]
        │
        ▼
Inference: CNN + LSTM → fatigue-over-time curve, plotted   [Model.py]
```

## Files

| File | Role |
|---|---|
| `CNN.py` | Trains the pose-estimation CNN (`PoseResNet`, a ResNet-18 backbone) on the SwimXYZ synthetic swimmer dataset. Pairs each video with its COCO-format pelvis keypoint annotations, letterboxes frames to 256×256, caches everything in memory, and trains with MSE loss / Adam. Saves `pose_cnn_model.pth`. |
| `LSTM.py` | Loads the frozen CNN, runs it over a race clip to get a pose sequence, derives **pseudo fatigue labels** from that sequence using the composite fatigue equation below, then trains a 2-layer LSTM (`FatigueLSTM`) to predict fatigue from the CNN's pose features. Saves `fatigue_lstm_model.pth`. |
| `Model.py` | Inference/evaluation script. Loads both trained models, runs a race clip through the full pipeline, and plots predicted fatigue against the pseudo-label curve for that clip. |

## The Fatigue Score

Because no public dataset labels swimmer fatigue directly, this project defines its own composite fatigue score from four technique indicators, each derived from the CNN's pose output and weighted by how much of a swimmer's potential performance gain the literature attributes to that factor:

- **E** — stroke efficiency: `(velocity × stroke length) / stroke rate`
- **K** — kick strength (approximated via stroke rate in this implementation)
- **A** — body asymmetry, measured as the horizontal distance between two hip/shoulder keypoints
- **S** — split time, as a velocity proxy

```
ΔE = 1 - (E / E0)
ΔK = 1 - (K / K0)
ΔA = (A - A0) / A0
ΔS = (S - S0) / S0

P = 0.60(1-ΔE) + 0.15(1-ΔK) + 0.15 · (1/(1+ΔA)) + 0.10(1-ΔS)

Fatigue Score = 100 × (1 - P)
```

Weights (0.60 / 0.15 / 0.15 / 0.10) come from Sanders et al. (2021) and Papic et al. (2024) on how much each factor contributes to elite swimmer performance gains — stroke efficiency dominates, kick strength and asymmetry are secondary, and split time is treated as a global (least independent) signal. All deltas are computed relative to the swimmer's own first 10 frames as a personal baseline, not against a fixed reference value.

**Implementation note:** in the current code (`compute_pseudo_fatigue_labels`), stroke rate and split time are modeled as simple linear ramps over the clip rather than measured directly from the video — only asymmetry (`A`) comes from the CNN's actual keypoint predictions. This makes the "pseudo label" partly synthetic, which is why the LSTM's job is to learn a correction on top of it rather than simply reproduce it.

## Model Details

- **CNN (`PoseResNet`)**: ResNet-18 (ImageNet-pretrained) backbone → two linear layers → 17 keypoints × (x, y), trained with MSE loss, Adam (lr `1e-3`), batch size 256, 15 epochs.
- **LSTM (`FatigueLSTM`)**: 2-layer LSTM, hidden size 64, input size 34 (17 keypoints × 2 coordinates). `LSTM.py` trains it to directly predict the pseudo fatigue label from pose features, for 50 epochs on a single race clip's pose sequence.
- **Inference-time model (`Model.py`)**: redefines `FatigueLSTM` with an `alpha` blending term. At test time, the final fatigue prediction is `alpha × pseudo_label + (1 - alpha) × learned_correction`, with `alpha = 0.9` — meaning the hand-derived pseudo-label formula dominates the final output, and the LSTM contributes a smaller learned correction on top of it.

  This heavy weighting toward the pseudo-label exists because the CNN's pose estimates on real race footage are unreliable: the CNN was trained on **SwimXYZ**, a synthetic (3D-rendered) swimmer dataset that only provides one camera angle per model and has no water turbulence or splash rendered. Real race footage has both — turbulent water, splashing, and camera angles the CNN never saw in training — so its keypoint predictions on real video are noisy. Leaning 90% on the hand-derived pseudo-label and only 10% on the CNN-derived LSTM correction was a way to keep the output usable despite that domain gap, rather than trusting the CNN's real-world pose tracking as much as its pseudo-label input.

  This also means the plain direct-prediction architecture used during training in `LSTM.py` differs from the blended architecture used at inference in `Model.py`; the two share compatible LSTM/linear layer shapes (so saved weights still load), but the forward pass — and therefore what the model is actually outputting — is different between training and inference. Worth being aware of if you're trying to reproduce or extend the fatigue predictions.

## Dataset

- **Pose training data**: [SwimXYZ](https://github.com/) — a synthetic (3D-rendered) swimmer dataset with COCO-format keypoint annotations for aerial and front camera angles. Not included in this repo; download separately and point `CNN.py`'s `video_base` / `label_base` at it.
- **Evaluation footage**: short (~17–21s) spliced clips of elite swimmers' 100m freestyle races from the [2024 Paris Olympics](https://www.youtube.com/watch?v=q14W1uCJag4), used for research/educational analysis. Not included in this repo — source your own clips and update the hardcoded video paths in `LSTM.py` / `Model.py`.

## Usage

These scripts currently use hardcoded, machine-specific file paths (Windows `D:\` drive paths) rather than CLI arguments or a config file — update the constants at the top of `main()` / `test_lstm()` in each script before running:

```bash
# 1. Train the pose CNN on SwimXYZ
python CNN.py
# → produces pose_cnn_model.pth

# 2. Generate pseudo fatigue labels and train the LSTM on a race clip
python LSTM.py
# → produces fatigue_lstm_model.pth

# 3. Run inference on a clip and plot the fatigue curve
python Model.py
```

## Hardware Used

Same machine as used for the [Mistral-7B LLM Custom Training](../Mistral-7B-LLM-custom-trained) project:

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12GB GDDR6 VRAM) |
| CPU | Intel i5-11400/F (2.60 GHz base, 4.40 GHz boost) |
| RAM | 16GB (32GB recommended) |
| Storage | 500GB SSD + 1TB HDD (full 1TB not required) |
| CUDA | 11.7 |
| Python | 3.9 |
| PyTorch | 1.13 |
| OpenCV | 4.5 |

Caching all training frames in memory (rather than reading from disk per batch) maximized GPU utilization but pushed RAM and VRAM usage close to their 12GB limits.

## Results

Evaluated on spliced clips of five elite male 100m freestyle swimmers (2024 Paris Olympics):

| Swimmer | Clip Length | Race Time (s) | Max Fatigue | Avg Fatigue | SD |
|---|---|---|---|---|---|
| Pan Zhanle | 21s | 46.40 | 11.26 | 3.80 | 3.92 |
| Nandor Nemeth | 17s | 47.50 | 12.55 | 5.64 | 4.93 |
| Kyle Chalmers | 17s | 47.48 | 13.27 | 8.19 | 4.38 |
| Maxime Grousset | 17s | 47.71 | 13.10 | 9.47 | 4.16 |
| David Popovici | 17s | 47.49 | 10.88 | 5.02 | 4.28 |

The fastest swimmer in the set (Pan Zhanle) showed the lowest average fatigue and fewest spikes, consistent with the idea that faster swimmers conserve technique more effectively through the race.

## Known Limitations

- **Small, spliced sample**: only 5 swimmers, each represented by a single short clip rather than a full continuous race, so fatigue values reflect only part of each swim.
- **Single camera angle, no water turbulence in training data**: the CNN was trained on SwimXYZ synthetic 3D swimmer renders, which cover only one camera angle per model and don't simulate splashing or turbulent water. Real race footage has both, so the CNN's pose predictions on real video are noisy — this is the main reason the final fatigue prediction leans 90% on the hand-derived pseudo-label rather than trusting the learned correction more heavily.
- **Partly synthetic labels**: as noted above, two of the four fatigue components (stroke rate, split time) are modeled rather than measured in the current pseudo-labeling code.
- **Hardcoded paths**: all three scripts assume a specific local folder layout (`D:\SwimmingDataset\...`) and file names for saved models — update these before running on a different machine.

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list (PyTorch, torchvision, OpenCV, NumPy, Matplotlib).

## Citation

If referencing this work, see the accompanying paper: *CNN-LSTM Hybrid Deep Learning Model for Technique Analysis of Elite Swimmers in the 100m Freestyle* (Villalba, AP Research). If paper is not avaliable, it is in progress for edits.


