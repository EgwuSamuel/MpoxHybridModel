# A comparative analysis of deep learning models for Mpox skin lesion classification

Code for the paper

> **A comparative analysis of convolutional, transformer and hybrid deep learning models for Mpox skin lesion diagnosis under leakage-free patient-level evaluation**
> Sodiya Adesina Simon, Egwu Samuel Onuche-Ojo, Aborisade Dada O, Abdullahi Ahmed, Olasoju Mary Idowu

Nine pretrained models from three architecture families are trained with one protocol and evaluated on
identical patient-level five-fold cross-validation folds of the Mpox Skin Lesion Dataset v2.0
(MSLD v2.0: 755 original images, 524 patients, six classes). The code also measures how much two
common evaluation errors, splitting by image and augmenting before splitting, inflate accuracy.

## Repository layout

```
notebooks/
  01_cross_validation_comparison.ipynb        nine models, identical folds, statistics, leakage
  02_final_model_export_cross_dataset.ipynb   reference model, TensorFlow Lite, efficiency, MSLD v1.0
analysis/
  make_figures.py          figures, patient-level confidence intervals and the late fusion control
requirements.txt
```

The repository holds code only. Datasets, trained models and result files are not included.

## Models

All backbones are frozen and used as feature extractors. Only the classification head is trained.

| Name in the code | Name in the paper | Backbone pretraining |
|---|---|---|
| `MobileNetV2` | MobileNetV2 | ImageNet-1k, supervised |
| `EfficientNetV2` | EfficientNetV2-B0 | ImageNet-1k, supervised |
| `ResNet50` | ResNet50 | ImageNet-1k, supervised |
| `DenseNet121` | DenseNet121 | ImageNet-1k, supervised |
| `ViT-B/16` | ViT-B/16 | ImageNet-21k, supervised (`google/vit-base-patch16-224-in21k`) |
| `DINO ViT-B/16` | DINO ViT-B/16 | ImageNet-1k, self-supervised (`facebook/dino-vitb16`) |
| `Hybrid+Adam` | Hybrid + Adam | MobileNetV2 + ViT-B/16, features concatenated |
| `MoViT-SAM` | Hybrid + SAM (reference model) | MobileNetV2 + ViT-B/16, trained with SAM |
| `Hybrid+H-SAM` | Hybrid + H-SAM | MobileNetV2 + ViT-B/16, trained with a norm-adaptive SAM |

## Evaluation protocol

- Patient identity is taken from the image filenames and every split is made by patient.
- The same five stratified-group folds are used for every model.
- Augmentation and Mixup are applied to training batches only.
- Early stopping and learning-rate scheduling use an inner validation split of the training
  patients. The evaluation fold is never used for model selection.
- Temperature scaling is cross-fitted: the temperature applied to a fold is fitted on the other folds.
- Models are compared with a Friedman test on fold accuracies and exact McNemar tests on paired
  out-of-fold predictions, Holm-adjusted, at image level and at patient level.
- Leakage is measured by repeating the training of one model per family under an image-level split
  that ignores patients and under augmentation before splitting.

## Data

The datasets are public and are not redistributed here.

- MSLD v2.0: <https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20>
- MSLD v1.0, used only for the cross-dataset check: <https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset>

Only the original images of MSLD v2.0 are used. The augmented images shipped with the dataset are discarded.

## Running the code

### Training (Kaggle, GPU)

The notebooks were developed on Kaggle with two NVIDIA T4 GPUs, Python 3.13, TensorFlow 2.20.0,
tf-keras 2.20.1 and transformers 4.46.3. Section 0 of each notebook installs the pinned packages.

1. Create a Kaggle notebook from `notebooks/01_cross_validation_comparison.ipynb`, attach MSLD v2.0
   as an input or turn Internet on, and select a GPU.
2. Optional check: set `SMOKE_TEST = True` in Section 1. Every model is trained for one epoch on two folds.
3. Full run: leave `SMOKE_TEST = False` and use *Save Version > Save & Run All*. It takes roughly
   nine hours. Each fold is checkpointed, so an interrupted session resumes where it stopped.
   The notebook writes `comparative_results.zip`.
4. Run `notebooks/02_final_model_export_cross_dataset.ipynb` the same way in a new session. It trains
   one model, is much shorter, and writes `part2_results.zip`.

The notebooks also run outside Kaggle. Set `MOVIT_DATA_ROOT` to the dataset folder and install
`requirements.txt`.

GPU training is not bit-for-bit deterministic, so two runs can differ slightly.

### Figures and statistics (no GPU, no training)

`analysis/make_figures.py` works from the files written by the two notebooks.

1. Unpack `comparative_results.zip` and copy every file from its `results/` and `comparative/`
   folders into `results/cross_validation/` at the root of this repository.
2. Unpack `part2_results.zip` and copy every file from its folders into `results/final_model/`.
3. Run the script.

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python analysis/make_figures.py
```

The script checks that the stored predictions reproduce the stored accuracies and calibration
values, draws the figures into `figures/`, and writes `analysis/summary_numbers.json`. It also
computes the confidence intervals by resampling patients instead of images, and the late fusion
control, which averages the out-of-fold probabilities of two models trained separately. It takes a
few minutes.

## Citation

The paper is under review. Until it is published, please cite this repository:

```
Sodiya AS, Egwu SO, Aborisade DO, Abdullahi A, Olasoju MI (2026) A comparative analysis of convolutional,
transformer and hybrid deep learning models for Mpox skin lesion diagnosis under leakage-free
patient-level evaluation. Code: https://github.com/EgwuSamuel/MpoxHybridModel
```

## Acknowledgements

This work is part of Artificial Intelligence for One Health (AI4OneHealth, AI4PEP Nigeria) and was
funded by the Canadian International Development Research Centre (IDRC) under Grant Agreement 109981,
with funding from IDRC and the UK government's Foreign, Commonwealth and Development Office. The views
expressed are those of the authors.
