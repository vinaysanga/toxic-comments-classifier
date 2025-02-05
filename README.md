# Toxic Comments Classifier

This project implements a comprehensive machine learning system for detecting and classifying toxic comments in text data. It features a dual-approach methodology, combining both traditional machine learning techniques and state-of-the-art Large Language Models (LLMs) for robust toxic comment detection.

## Project Structure

- [`classic_ml_approaches/`](classic_ml_approaches/): Implementation of traditional machine learning models for toxicity detection.
- [`data/`](data/): Dataset storage and preprocessing scripts for handling toxic comment data.
- [`llm/`](llm/): Large Language Model implementations and experiments
  - [`checkpoints/`](llm/checkpoints/): Checkpoints for trained models. Use this for inference.ipynb without training your own model. See `inference.ipynb` for usage.
  - [`experiment.ipynb`](llm/experiment.ipynb): Notebook for data preprocessing, tokenization, and model training. Alternatively use `trainer.py` for training.
  - [`inference.ipynb`](llm/inference.ipynb): Notebook for model inference. Use this for testing your own sentence(s).
  - [`trainer.py`](llm/trainer.py): Script for training models. Also produces validation reports.
  - [`test.py`](llm/test.py): Model testing and evaluation scripts.
- [`Toxic Comment Classifier Report.pdf`](Toxic%20Comment%20Classifier%20Report.pdf): Detailed technical report on the project methodology and results.

## Getting Started

1. Clone the repository.
2. Select your approach (classical ML or LLM).
3. Install the requirements (from the specific directory):
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook or scripts:
   - For classical ML: Use the notebook in [`classic_ml_approaches/`](classic_ml_approaches/)
   - For LLM-based approach: Navigate to [`llm/`](llm/) and run the notebooks or scripts if you have the data already tokenized.

## Features

- Multiple model approaches (classical ML and LLMs)
- GPU support for model training
- Comprehensive evaluation metrics
- Modular codebase structure

## Requirements

The project dependencies are as follows:
- Classical machine learning approaches: [`classic_ml_approaches/requirements.txt`](classic_ml_approaches/requirements.txt)
- LLM-based approaches: [`llm/requirements.txt`](llm/requirements.txt)

For best compatibility, use Python 3.12+.

## Usage

1. Prepare your data in the [`data/`](data/) directory
2. Choose your approach:
   - For classical ML: Use the scripts in [`classic_ml_approaches/`](classic_ml_approaches/)
   - For LLM-based solutions: Navigate to the [`llm/`](llm/) directory
3. Training:
   - Use [`llm/trainer.py`](llm/trainer.py) for training LLM models
   - Follow the notebooks for step-by-step training process
4. Inference:
   - Use [`llm/inference.ipynb`](llm/inference.ipynb) for making predictions
   - Run [`llm/test.py`](llm/test.py) for model evaluation

For detailed methodology and results, refer to the [`Toxic Comment Classifier Report.pdf`](Toxic%20Comment%20Classifier%20Report.pdf)

## License

This project is open source and available under the [MIT License](LICENSE).