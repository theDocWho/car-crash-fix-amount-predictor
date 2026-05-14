# Dataset Citations

Datasets used (or planned for use) in the Car Crash Fix Amount Predictor capstone project. Each entry lists the source, license, suggested citation, and how the dataset is used in this project. Every result we publish should reproduce the relevant attribution.

---

## 1. CarDD — Car Damage Detection

- **Kaggle mirror**: [nasimetemadi/car-damage-detection](https://www.kaggle.com/datasets/nasimetemadi/car-damage-detection)
- **Original release**: CarDD_release (used directly under `data/raw/car-damage-detection/CarDD_release/`)
- **Size**: 5.7 GB, ≈4,000 images with COCO and SOD annotations
- **Categories** (6): `dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`
- **License**: "other" on Kaggle; the underlying CarDD release is intended for academic research — verify on the source page before any commercial redistribution.
- **Suggested citation**:

  > Wang, X., Li, W., & Pan, Z. (2023). *CarDD: A New Dataset for Vision-Based Car Damage Detection.* IEEE Transactions on Intelligent Transportation Systems, 24(7), 7202–7214. https://doi.org/10.1109/TITS.2023.3258480

- **Used in this project for**:
  - YOLOv8 detection training (Variant B).
  - ResNet50 multi-label classification training (Variant A) — labels are damage *types*, not parts.
  - Provides COCO-format segmentation that we convert to YOLO bboxes.

---

## 2. Comprehensive Car Damage Detection

- **Kaggle**: [samwash94/comprehensive-car-damage-detection](https://www.kaggle.com/datasets/samwash94/comprehensive-car-damage-detection)
- **Size**: 642 MB, ≈2,300 images
- **Categories** (6): `F_Crushed`, `F_Normal`, `F_Breakage`, `R_Crushed`, `R_Normal`, `R_Breakage` (front/rear × condition; folder-based labels)
- **License**: see the dataset page on Kaggle; the uploader has not specified a formal license.
- **Suggested citation**:

  > Wash, S. (2024). *Comprehensive Car Damage Detection* [Data set]. Kaggle. https://www.kaggle.com/datasets/samwash94/comprehensive-car-damage-detection

- **Used in this project for**:
  - Auxiliary head: damage *location* (front vs rear) and *condition* (normal / crushed / breakage).
  - Baseline whole-image classification reference.

---

## 3. IAAI Insurance Auto Auction Dataset (Rebrowser sample)

- **Kaggle**: [rebrowser/iaai-dataset](https://www.kaggle.com/datasets/rebrowser/iaai-dataset)
- **Size in our copy**: 11 MB (30 daily CSV + 30 daily Parquet files), 12,353 rows
- **Coverage**: 2025-11-16 to 2026-04-19, sample of ~5.3% of the full corpus (567,559 records)
- **Premium-masked fields** in this free sample (every row literally `"[PREMIUM]"`): `vin`, `imageUrl`, `image360Url`, `estimatedRepairCost`, `buyNowPrice`, `minimumBidAmount`, `sellerName`, `whoCanBuy`, `aisle`, `stall`, `listingUrl`, etc.
- **Free, usable fields**: `year`, `make`, `model`, `bodyStyle`, `vehicleClass`, `exteriorColor`, `mileage`, `primaryDamage` (location), `secondaryDamage`, `lossType`, `engine`, `transmission`, `drivetrain`, `fuelType`, `titleType`, `currencyCode`.
- **License / terms**: see Rebrowser's distribution page. Larger non-premium slices may be available for academic researchers — apply at https://rebrowser.net/free-datasets-for-research. Full commercial dataset at https://rebrowser.net/products/datasets/iaai.
- **Suggested citation**:

  > Rebrowser (2026). *IAAI Insurance Auto Auction Dataset (preview sample)* [Data set]. Kaggle. https://www.kaggle.com/datasets/rebrowser/iaai-dataset
  >
  > Insurance Auto Auctions (IAA), Inc. is the original source of the underlying auction listings. This dataset is a scrape/preview redistributed by Rebrowser.

- **Used in this project for**:
  - Car-metadata distribution source (year / make / model / bodyStyle distributions used when synthesizing tabular features for un-labeled images).
  - Damage-location vocabulary reference.
  - Reference table for the three-tier cost-estimation fallback chain.
  - **Not** used for real cost regression supervision (no usable cost rows in the free sample).

---

## 4. Stanford Cars (Phase 1.5 make/model/year identifier)

- **Kaggle mirror used in this project**: [eduardo4jesus/stanford-cars-dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset) — directly downloadable (other mirrors such as `jessicali9530/stanford-cars-dataset` require rules-acceptance and may return 403).
- **Original release**: 16,185 images across 196 classes (Make Model Year), ~2GB
- **Note**: original Stanford host (ai.stanford.edu) is intermittent; most users now grab from Kaggle / HuggingFace mirrors.
- **License**: research / academic use; verify per-mirror terms.
- **Suggested citation**:

  > Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). *3D Object Representations for Fine-Grained Categorization.* 4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13). Sydney, Australia. https://ai.stanford.edu/~jkrause/cars/car_dataset.html

- **Used in this project for**:
  - Fine-tuning the make/model/year identification head (Phase 1.5).
  - Recovering car identity from damage-only images where filename/EXIF heuristics fail.

---

## Software & framework citations (for completeness in the final report)

These are not datasets but should be acknowledged in the report:

- **PyTorch**: Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
- **torchvision** for ResNet50 weights and transforms.
- **Ultralytics YOLOv8**: Jocher, G., Chaurasia, A., & Qiu, J. (2023). *YOLO by Ultralytics* (v8). https://github.com/ultralytics/ultralytics — AGPL-3.0.
- **XGBoost**: Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.
- **ImageNet** (ResNet50 pretraining): Deng, J., et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR '09.
- **EasyOCR**: JaidedAI. https://github.com/JaidedAI/EasyOCR.
- **FX rates**: `exchangerate.host` (primary), `open.er-api.com` (fallback), `frankfurter.app` (fallback). All free public APIs; rate snapshots recorded with every prediction.

---

## How to cite this project

If you use or build on this work, please cite:

> Roy, A. (2026). *Car Crash Fix Amount Predictor: A calibrated damage-recognition and repair-cost estimation system with a versioned parts-cost catalog.* Capstone project. https://github.com/<your-repo-path-here>
