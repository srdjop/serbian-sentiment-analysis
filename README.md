# Serbian Sentiment Analysis

This project implements sentiment analysis on the Serbian language by fine-tuning a pre-trained BERT model. The main model used, **`classla/bcms-bertic`**, is adapted to the `SerbMR-2C.csv` dataset for classifying text into two categories: **positive** and **negative**.

The project also includes a feature to compare the performance of this specialized model against a more general multilingual model, **`google-bert/bert-base-multilingual-cased`**.

---

## Project Structure

-   **`src/`**: Contains the Python scripts for the project.
    -   `train_model.py`: Script for training and fine-tuning the model.
    -   `evaluate_models.py`: Script for evaluating and comparing the performance of all trained models.
    -   `inference.py`: Script for running sentiment analysis on new text.
-   **`data/`**: Contains the `SerbMR-2C.csv` dataset used for training.
-   **`models/`**: The folder where the trained models and tokenizers are saved.

---

## Installation and Setup

### 1. Clone the repository

Open your terminal and clone the project from GitHub.
```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
```

2.  **Create and activate a virtual environment:**
    -   On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install required libraries:**
    **Note**: *torch* and *torchvision* are not included in requirements.txt due to their specific installation requirements. You must install them manual after the other dependencies.
    ```bash
    pip install -r requirements.txt
    ```
    - **For user with an NVIDIA GPU:**
    Install PyTorch with CUDA support. Visit the official PyTorch website https://pytorch.org/get-started/locally/ for the most precise command. Example: 
    ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ```
    - **For user without a GPU or with an AMD/Apple GPU**
    Install the PyTorch CPU-only version. Example:
    ```bash
    pip3 install torch torchvision
    ```

4.  **Training a model (optional):**
    This step is necessary only if you don't have a trained model in your models/ folder.

    To train a model, run train_model.py. You can specify a model by its name from the Hugging Face Hub.

    - Using the default model (classla/bcms-bertic):
    ```bash
    python src/train_model.py
    ```
    - Using a different model (e.g., google-bert/bert-base-multilingual-cased):
    ```bash
    python src/train_model.py --model_name google-bert/bert-base-multilingual-cased
    ```

    Trained models will be saved in the models/ folder.


5.  **Model Evaluation:**
    After training, you can evaluate the performance of your models.
    The evaluation script can be used for models loaded directly from the Hugging Face Hub (default) or for those you have saved locally in the models/ folder (need to change the path in the script).
    
    ```bash
    python src/evaluate_models.py
    ```
7. **Sentiment Analysis (Inference)**
    Use a trained model to analyze new text. You can add your own sentences to the *texts_to_analyze* list in the script and then run the script.
    The inference script can be used for models loaded directly from the Hugging Face Hub (default) or for those you have saved locally in the models/ folder (need to change the path in the script).

    ```bash
    python src/inference.py
    ```
