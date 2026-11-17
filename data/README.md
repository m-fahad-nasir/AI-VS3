# Data

The Tomato-VS3 dataset is distributed separately (see the main README). Place it here:

```
data/
├── Autonomous_Scouting_Dataset.xlsx   agronomist annotations, one sheet per session
├── images/                            scouting images (full frame)
├── crops/                             primary-leaf crops, written by step 2b
└── yolo/                              leaf bounding boxes in Ultralytics format
    ├── dataset.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## Image naming

Every image name encodes session, plant, canopy level and view:

```
Date_03_Feb_2026_C1-001-High-A.jpg
└─ session ────┘ └plant┘ └lvl┘ └view (A, B, C)
```

Levels are `Low`, `Mid` and `High`; each level has three views (A, B, C), giving nine images per
plant per session. Step 1 (`scripts/01_prepare_images.py rename`) produces these names from the
camera's capture order.

## Annotation workbook

`Autonomous_Scouting_Dataset.xlsx` has one sheet per session (`Week_01` to `Week_04`).
Row 1 is a title, row 2 the header, then one row per image:

| Column | Values |
|---|---|
| `Plant_Stem_Number` | `Plant C1-001` ... `Plant C1-050` (written on the first row of each plant block) |
| `Plant Images Level` | `High`, `Middle`, `Low` |
| `File Name` | image file name (see above) |
| `Health Status` | `Healthy`, `Not Healthy` |
| `Disease Category` | `Fungal Leaf Disease`, `Bacterial Leaf Disease`, `Viral Leaf Disease`, `Nutrient Disease`, `Abiotic`, `Nematode`, `Uncertain` |
| `Disease` | specific disease or disorder |
| `Severity` | `Low`, `Medium`, `High`, `Uncertain` |
| `Confidence` | `Low`, `Medium`, `High` |
| `Visible Symptoms` | free text |
| `Reasoning` | free text |

Only rows whose `File Name` is an image file are read; any other row in a sheet is ignored.

## Results workbook

`scripts/04_build_results_workbook.py` writes `outputs/VLM_Results.xlsx` with one sheet per model.
Each annotation column appears twice, as `<column> (GT)` and `<column> (Predicted)`.
