# MoViT-SAM: patient-level, leakage-free evaluation of Mpox differential diagnosis

Code for the manuscript *"Data leakage, clinical utility and deployment of a hybrid MobileNetV2–Vision Transformer model (MoViT-SAM) for Mpox differential diagnosis: a patient-level retrospective evaluation in Ogun State, Nigeria"* (submitted to BMC Medical Informatics and Decision Making).

## What the notebook does

`MoViT_SAM_full_new.ipynb` reproduces every result in the manuscript in one run:

| Manuscript item | Notebook section |
|---|---|
| Data integrity audit (perceptual hashing, patient IDs, skin tone by ITA) | 4 |
| Patient-level split (Table 1) | 5 |
| Focal loss with label smoothing, Mixup, SAM and H-SAM training | 6–10 |
| Five-fold stratified-group cross-validation, pooled out-of-fold results (Tables 3–5, Figs 3 and 5) | 11–13 |
| Cross-fitted temperature scaling (Fig 4) | 14 |
| Binary screening, predictive values, decision curves, error taxonomy (Table 7, Fig 6) | 15 |
| Leakage quantification (Table 2) | 16 |
| Single-run ablation (Table 6) | 17 |
| TensorFlow Lite export, FP32 and INT8 (Table 8) | 18 |
| Cross-dataset check on MSLD v1.0 with overlap audit (Table 9, Fig 7) | 21 |

Leakage safeguards enforced in code: patients are split before any augmentation, augmentation is applied only to training batches, early stopping uses an inner split of training patients only, and temperature scaling is cross-fitted.

## How to run

1. Open the notebook on Kaggle.
2. Settings: Accelerator **GPU T4 ×2**, Internet **On**.
3. **Save Version → Save & Run All**. The run takes about 3 hours.
4. Results are written to `results/`, models to `saved_models/`, and every file used in the manuscript is collected in `for_paper.zip`.

Both datasets (MSLD v2.0 and MSLD v1.0) are downloaded automatically through the Kaggle API. The notebook stops early if no GPU is present or if it does not find exactly 755 MSLD v2.0 original images.

## Data

- MSLD v2.0: https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20
- MSLD v1.0: https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset

## Environment of the reported run

Python 3.13, TensorFlow 2.20.0 with tf-keras 2.20.1, transformers 4.46.3, scikit-learn 1.6.1, on Kaggle with two GPUs. Two complete runs reproduced every reported metric to four decimal places, apart from CPU latency.

## Deployed model

The exported model takes raw RGB input in the range 0–255, resized to 224 × 224, and output index 0 is Monkeypox. `saved_models/model_card.json` records the label order, the input format and the calibration temperature (T ≈ 0.47). Calibrated probabilities are obtained by dividing the log-probabilities by T and re-applying softmax.

## Funding

This work was funded by the Canadian International Development Research Centre (IDRC) under Grant Agreement 109981, with funding from IDRC and the UK government’s Foreign, Commonwealth and Development Office, as part of the Artificial Intelligence for One Health (AIA4OneHealth) initiative, also called AI4PEP Nigeria.
