# AI-VS3: AI-Assisted Vertical Stem Stratified Scouting

Code for the article **"AI-VS3 (AI-Assisted Vertical Stem Stratified Scouting): A Temporal Multimodal
Framework for Greenhouse Tomato Plant Health Assessment"** (*Computers and Electronics in Agriculture*).

AI-VS3 turns structured greenhouse scouting images into an interpretable, plant-level **Health Index (HI)**
that can be tracked over time:

1. **VS3 scouting**: nine images per tomato stem per session (Low, Mid and High canopy, three views each).
2. **Leaf detection**: YOLOv11 locates the primary leaf, which is cropped.
3. **Zero-shot VLM assessment**: a structured JSON prompt returns health status, disease category,
   disease, severity, confidence, visible symptoms and reasoning.
4. **Hierarchical Health Index**: image → canopy level → plant aggregation with disease-specific
   threat, spread potential, multi-image agreement and vertical canopy weighting.
5. **Temporal trajectories**: Health Index per plant across weekly sessions.

<!-- FIGURE: Fig. 2 - VS3 scouting protocol
![VS3 scouting protocol](assets/fig2_vs3_protocol.png) -->

---

## Repository structure

```
AIVS3/
├── aivs3/                          Python package
│   ├── tomato_threat_scoring.py    Health Index (Section 4.3, Algorithm 1)
│   ├── evaluation.py               VLM benchmark metrics (Table 4)
│   ├── temporal.py                 trajectories, trend classes, Figures 7-8 (Section 5.7)
│   ├── vlm.py                      prompt, API clients and parallel runner (Section 4.2)
│   ├── detection.py                primary-leaf selection and cropping (Section 4.1)
│   └── workbook.py                 results workbook (annotations + predictions)
├── scripts/                        pipeline, run in order
│   ├── 01_prepare_images.py
│   ├── 02a_train_yolo.py
│   ├── 02b_detect_and_crop.py
│   ├── 03_run_vlm_benchmark.py
│   ├── 04_build_results_workbook.py
│   ├── 05_evaluate_vlms.py
│   ├── 06_score_health_index.py
│   └── 07_temporal_analysis.py
├── configs/
│   ├── models.yaml                 the ten VLMs and their settings
│   ├── yolo.yaml                   detector training settings
│   └── prompts/                    system and user prompt
├── notebooks/
│   ├── 01_health_index_quickstart.ipynb
│   └── 02_visualize_bboxes.ipynb
├── tests/                          unit tests (pytest)
├── data/README.md                  dataset layout and column schema
├── environment.yml / requirements.txt / pyproject.toml
└── CITATION.cff / LICENSE
```

---

## Installation

```bash
git clone https://github.com/<user>/AIVS3.git
cd AIVS3
conda env create -f environment.yml
conda activate aivs3
```

With pip only: `pip install -r requirements.txt && pip install -e .`

For GPU training of the detector, install the PyTorch build that matches your CUDA version
([pytorch.org](https://pytorch.org/get-started/locally/)) before `ultralytics`.

**API keys** (needed only for step 3): copy `.env.example` to `.env` and add your keys.
`.env` is git-ignored.

---

## Data

Download the Tomato-VS3 dataset and place it in `data/` as described in [data/README.md](data/README.md).

<!-- DATASET: add the dataset link / DOI here -->

| Item | Value |
|---|---|
| Plants (stems) | 50 (C1-001 to C1-050) |
| Sessions | 4 (03, 13, 17 and 26 February 2026) |
| Images per plant per session | 9 (Low / Mid / High × A, B, C) |
| Labels | health status, disease category, disease, severity, confidence, visible symptoms, reasoning (agronomist annotations) |

---

## Quick start: Health Index and trajectories from the annotations

No API keys or GPU are needed.

```bash
python scripts/06_score_health_index.py --input data/Autonomous_Scouting_Dataset.xlsx \
       --sheets Week_01 Week_02 Week_03 Week_04 --source annotation --name Agronomist
python scripts/07_temporal_analysis.py --input outputs/health_index/Agronomist_Health_Index.csv
```

This reproduces the Health Index of Section 5.6, the trend distribution of Table 5, the trajectories
of Table 6 and Figures 7 and 8. The same steps are shown in
[notebooks/01_health_index_quickstart.ipynb](notebooks/01_health_index_quickstart.ipynb).

---

## Full pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `01_prepare_images.py rename / collect / resize` | VS3-named images in `data/images` |
| 2a | `02a_train_yolo.py --config configs/yolo.yaml` | detector weights, validation metrics |
| 2b | `02b_detect_and_crop.py --weights <best.pt>` | primary-leaf crops in `data/crops` |
| 3 | `03_run_vlm_benchmark.py --input crop` | `outputs/vlm_raw/crop/<model>.jsonl` |
| 4 | `04_build_results_workbook.py --jsonl outputs/vlm_raw/crop` | `outputs/VLM_Results.xlsx` |
| 5 | `05_evaluate_vlms.py` | `outputs/evaluation/` (Table 4) |
| 6 | `06_score_health_index.py` | `outputs/health_index/` |
| 7 | `07_temporal_analysis.py` | `outputs/temporal/` (Table 5, Table 6, Figures 7-8) |

Every script documents its options with `--help`.

### 1. Image preparation

```bash
python scripts/01_prepare_images.py rename  --root raw/Session_03_Feb --session Date_03_Feb_2026 --dry-run
python scripts/01_prepare_images.py rename  --root raw/Session_03_Feb --session Date_03_Feb_2026
python scripts/01_prepare_images.py collect --root raw/Session_03_Feb --out data/images_full_res --id-dir data/id_images
python scripts/01_prepare_images.py resize  --src data/images_full_res --out data/images --scale 0.5
```

### 2. Leaf detection

```bash
python scripts/02a_train_yolo.py --config configs/yolo.yaml                        # YOLOv11-L
python scripts/02a_train_yolo.py --config configs/yolo.yaml --model yolo11m.pt --name yolo11m_leaf
python scripts/02b_detect_and_crop.py --images data/images --out data/crops \
       --weights outputs/detection/yolo11l_leaf/weights/best.pt
```

Crops keep the original file name, so steps 3-7 work identically on crops and full images.

<!-- TABLE: Table 3 - leaf detection benchmark -->

### 3. VLM inference

```bash
python scripts/03_run_vlm_benchmark.py --input crop                         # all ten models
python scripts/03_run_vlm_benchmark.py --input full --models GPT_5.2 Sonnet_4.6
python scripts/03_run_vlm_benchmark.py --input crop --limit 5               # smoke test
```

All models receive the same prompt ([configs/prompts](configs/prompts)) and image. Temperature is 0
for models that accept it; reasoning models use a reasoning-effort setting instead
([configs/models.yaml](configs/models.yaml)). Runs resume automatically.

| Provider | Models |
|---|---|
| OpenAI | GPT-4o, GPT-4.1, GPT-5.1, GPT-5.2 |
| Anthropic | Claude Sonnet 4.6, Claude Sonnet 4.0, Claude Haiku 4.5 |
| Google | Gemini Flash 2.5, Gemini Flash Lite 2.5, Gemini Flash Lite 3.1 |

<!-- FIGURE: Fig. 4 - multimodal VLM reasoning framework
![VLM framework](assets/fig4_vlm_framework.png) -->

### 4-5. Benchmark evaluation

```bash
python scripts/04_build_results_workbook.py --jsonl outputs/vlm_raw/crop
python scripts/05_evaluate_vlms.py
```

Categorical fields are scored by accuracy after label normalisation (a healthy leaf counts as
"no disease" for category, disease and severity); free text is scored with BLEU-4, ROUGE-L and
TF-IDF cosine similarity.

<!-- TABLE: Table 4 - VLM benchmark results -->
<!-- FIGURE: Fig. 6 - qualitative comparison of agronomist and VLM outputs
![Qualitative comparison](assets/fig6_qualitative.png) -->

### 6. Health Index

```bash
# agronomist annotations
python scripts/06_score_health_index.py --input data/Autonomous_Scouting_Dataset.xlsx \
       --sheets Week_01 Week_02 Week_03 Week_04 --source annotation --name Agronomist
# predictions of every model
python scripts/06_score_health_index.py --input outputs/VLM_Results.xlsx --source predicted --all-sheets
```

<!-- FIGURE: Fig. 5 - hierarchical Health Index framework
![Health Index](assets/fig5_health_index.png) -->

| Level | Computation |
|---|---|
| Image | R = E · S (disease threat × severity) |
| Canopy level | T = min{1, V · M · G · [α · max R + (1 − α) · mean R]} |
| Agreement | G = 0.8 + 0.3 A, A = 0.7 D + 0.3 K |
| Plant | T<sub>P</sub> = min{1, B · (0.75 · max T + 0.25 · Σ W T)}, HI = 1 − T<sub>P</sub> |
| Indicators | Q = mean stated confidence, U = max (1 − A) |

All coefficients are defined at the top of
[aivs3/tomato_threat_scoring.py](aivs3/tomato_threat_scoring.py) and written to the `Scoring_Config`
sheet of every output workbook. Disease-specific threat coefficients follow Supplementary Tables A8-A12.

### 7. Temporal analysis

```bash
python scripts/07_temporal_analysis.py --input outputs/health_index/Agronomist_Health_Index.csv
```

Each plant is classified by Δ = HI(Week 4) − HI(Week 1) into strong decline, decline, stable,
improvement or strong improvement, and flagged as volatile if its trajectory contains both a rise and a
fall of at least 0.20 between consecutive sessions.

<!-- FIGURE: Fig. 7 - population-level Health Index
![Population trend](assets/fig7_population_trend.png) -->
<!-- FIGURE: Fig. 8 - plant-level trajectories
![Trajectory heatmap](assets/fig8_trajectories.png) -->
<!-- TABLE: Table 5 - trajectory classes -->

---

## Tests

```bash
pytest
```

The tests check the Health Index rules (agreement bands, overrides, aggregation), the trajectory
classes, the label normalisation and the text metrics on small synthetic examples.

---

## Citation

```bibtex
@article{nasir2026aivs3,
  title   = {{AI-VS3} (AI-Assisted Vertical Stem Stratified Scouting): A Temporal Multimodal
             Framework for Greenhouse Tomato Plant Health Assessment},
  author  = {Nasir, Muhammad Fahad and Rehman, Mobeen Ur and Shafay, Muhammad and
             Hussain, Irfan and Abdulrahman, Yusra},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  doi     = {}
}
```

## License

Code: [MIT](LICENSE). The dataset is distributed under its own license.

## Acknowledgment

This research was funded by Khalifa University of Science and Technology through the
Silal: Mobile application strawberry disease inspection project (KU-EXT-8475000026-2025).
