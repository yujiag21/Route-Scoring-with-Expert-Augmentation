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
3. **Prepare Route Data**
   
   Place your route dictionary in the `data/` directory. The file should be a JSON file containing the synthetic routes you want to score.
4. **Generate Route Features**
   
   Use `route_feature_processing.py` to process the route dictionary and extract relevant features for scoring.
   ```bash
   python route_feature_processing.py --input_file <path_to_route_json> --output_file <path_to_features_json>
   ```
   - `<path_to_route_json>`: Path to the input JSON file containing route dictionaries.
   - `<path_to_features_json>`: Path where the processed features should be saved as a JSON file.
6. **Predict Route Scores**
   
   Once you have the route features, use the `main.py` script to predict the scores for each route.
   ```bash
   python main.py --input_file <path_to_features_json> --output_path <path_to_score_output>
   ```
   - `<path_to_features_json>`: Path to the JSON file generated in the previous step.
   - `<path_to_score_output>`: Path where the score results will be saved.
## File Structure
- `route_feature_processing.py`: Script to process route data and generate feature sets.
- `main.py`: Script to predict the score for routes based on processed features.
- `finder.yml`: Conda environment configuration file.
- `data/`: Directory to store route data and output files.
- `model/`: Directory containing pre-trained models for scoring.
- `reaction_class_summ_20.csv`: Summary of reaction classes for reference.
## Additional Files for AiZynthFinder
- `emols-stock-2023-01-01.csv`: Stock file used in AiZynthFinder
- `usp_filter_model.hdf5`, `usp_keras_model.hdf5`,`uspto_unique_templates.csv.gz`: Pre-trained models used in AiZynthFinder.
## Notes
- Ensure your route dictionary follows the correct format (containing smiles and in_stock attributes for molecules, mapped_reaction_smiles and classification attributes for reactions) before processing with `route_feature_processing.py`.
## Contact
For further questions or support, please reach out to yujia.guo@aalto.fi.
