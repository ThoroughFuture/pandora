# Digital Pathology Virtual Staining Pipeline

## 1. Data Preprocessing
**Script:** `preprocess/a_create_patches_fp.py`

**Function:** Extract image patches (tiling) from Whole Slide Images (WSI) for downstream processing.

## 2. Feature Extraction & Classification
**Script:** `preprocess/run_allmarker.py`

**Functions:**
- Extract biomarker features from tissue patches
- Determine optimal thresholds (cutoff values) for binary classification
- Classify samples as positive or negative based on biomarker expression levels

## 3. Model Training
**Script:** `staining/train_ddp.py`

**Function:** Train the non-linear prediction head for virtual staining using Distributed Data Parallel (DDP).

**Note:** DDP enables multi-GPU training for accelerated convergence.

## 4. Whole-Slide Inference & Visualization
**Script:** `staining/inference_whole_wsi768_croods.py`

**Function:** Perform inference on entire WSIs (768×768 coordinate patches) and generate visualization overlays of virtual staining results.