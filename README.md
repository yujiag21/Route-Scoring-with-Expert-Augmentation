# A Deep Learning with Expert Augmentation Approach for Route Scoring in Organic Synthesis
This project provides a pipeline for scoring synthetic routes based on their features and structure. The key steps involve setting up the environment, processing route features, and generating predictions for route scores.
## Prerequisites
- Anaconda or Miniconda installed on your system.
## Setup Instructions
1. **Create and activate Conda environment**
   
   Use the provided `environment.yml` to create a new Conda environment. This will install all the necessary dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate aizynth-dev
   ```

## Usage
1. **Prepare Route Data**
   
   Place your route dictionary in the `data/` directory. The file should be a JSON file containing the synthetic routes you want to score.
2. **Generate Route Features**
   
   Use `route_feature_processing.py` to process the route dictionary and extract relevant features for scoring.
   ```bash
   python route_feature_processing.py --input_file <path_to_route_json> --output_file <path_to_features_json>
   ```
   - `<path_to_route_json>`: Path to the input JSON file containing route dictionaries.
   - `<path_to_features_json>`: Path where the processed features should be saved as a JSON file.
3. **Predict Route Scores**
   
   Once you have the route features, use the `main.py` script to predict the scores for each route.
   ```bash
   python main.py --input_file <path_to_features_json> --output_path <path_to_score_output>
   ```
   - `<path_to_features_json>`: Path to the JSON file generated in the previous step.
   - `<path_to_score_output>`: Path where the score results will be saved.
4. **Predict with Fine-tuned Model (Expert-Augmented)**

   Use `main_fine_tuned.py` to run the LoRA fine-tuned classification model alongside the original regression network. It produces three outputs in one CSV:
   - **Per-reaction 5-class score** (1–5 scale) — feasibility/quality of each reaction step
   - **Route-level 3-class rating** (`Bad` / `Plausible` / `Good`) — driven by the lowest step score
   - **Continuous route score** — original `NeuralNetwork` regression output

   ```bash
   python main_fine_tuned.py --input_file <path_to_features_json> --output_path <path_to_score_output>
   ```

   Optional flags:
   - `--save_picture`: also render route images into the output directory
   - `--save_excel`: also save an `.xlsx` alongside the CSV
   - `--model_dir <dir>`: directory holding `fine_tune_encoder_cls.pt`, `encoder_sdf_prediction.pt`, `main_network_sdf_prediction.pt`, and (optionally) `hyperparams.json`. Defaults to `model/`.
   - `--lora_alpha <int>`: override the LoRA scaling factor (otherwise read from `hyperparams.json`).

   All other architecture hyperparameters (`encoding_size`, `lora_r`, `num_classes`, `input_size`) are auto-inferred from the checkpoint's `state_dict`, so you don't need to pass them.

   Example:
   ```bash
   python main_fine_tuned.py --input_file route_10.json --output_path route_score_fine_tuned --save_picture
   ```

   Output columns in `prediction_fine_tuned.csv`:
   | Column | Description |
   |--------|-------------|
   | `route_score` | Continuous regression score from `NeuralNetwork` |
   | `route_rating_3bin` / `route_rating_3bin_name` | Route-level 3-class rating (0=Bad, 1=Plausible, 2=Good) |
   | `step_scores_5cls` | List of per-reaction 5-class scores (1–5 scale) |
   | `step_scores_3bin` | List of per-reaction 3-class scores |
   | `step_labels_5cls` | List of per-reaction text labels |

## File Structure
- `route_feature_processing.py`: Script to process route data and generate feature sets.
- `main.py`: Script to predict the regression score for routes based on processed features.
- `main_fine_tuned.py`: Script that runs the fine-tuned LoRA classifier and the regression network jointly, producing per-step scores, a route-level rating, and a continuous score.
- `fine_tuning/`: Fine-tuning scripts (`fine_tune_encoder_5_classes.py`, `fine_tune_encoder.py`) for adapting the encoder with LoRA on expert-labeled data.
- `finder.yml`: Conda environment configuration file.
- `data/`: Directory to store route data and output files.
- `model/`: Directory containing pre-trained and fine-tuned model checkpoints used for scoring.
- `reaction_class_summ_20.csv`: Summary of reaction classes for reference.
## Additional Files for AiZynthFinder 
- `emols-stock-2023-01-01.csv`: Stock file used in AiZynthFinder
- `usp_filter_model.hdf5`, `usp_keras_model.hdf5`,`uspto_unique_templates.csv.gz`: Pre-trained models used in AiZynthFinder.
These files and example dataset can be downloaded from https://drive.google.com/drive/folders/15inTFu800g69YNlBnxUvHDgfuvD7Lcos?usp=sharing
## Notes
- Ensure your route dictionary follows the correct format (containing smiles and in_stock attributes for molecules, mapped_reaction_smiles and classification attributes for reactions) before processing with `route_feature_processing.py`.

## Fine tune

The encoder can be further fine-tuned on a small expert-labeled dataset (per-reaction 5-class feasibility scores). LoRA is applied to every linear layer of the `DeepSetEncoder`; the pretrained base weights stay frozen, and only the LoRA adapters (`A`, `B`) and a new classification head are trained.

**K-fold cross-validation (default):**
```bash
python fine_tuning/fine_tune_encoder_5_classes.py --mode kfold
```

**Full-data training (saves checkpoint):**
```bash
python fine_tuning/fine_tune_encoder_5_classes.py --mode full --save_model
```

**Custom hyperparameters:**
```bash
python fine_tuning/fine_tune_encoder_5_classes.py \
    --mode full --seed_data 42 --num_epochs 300 --lr 0.003 --save_model
```

The trained checkpoint is saved as `best_model_lora_encoder_cls.pt` in the configured `save_dir`, together with a `hyperparams.json`. To deploy it, copy (or symlink) the checkpoint into `model/` as `fine_tune_encoder_cls.pt` and run `main_fine_tuned.py` as described above.

