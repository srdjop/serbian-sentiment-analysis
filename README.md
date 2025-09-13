# Serbian Sentiment Analysis

This project explores sentiment analysis in the Serbian language by fine-tuning pre-trained `BERT` models. The initial hypothesis was that by training a BERT model on Serbian text using a tailored dataset, it would be possible to achieve a level of accuracy comparable to sentiment models trained on English datasets such as IMDB. However, this hypothesis was not confirmed, as the performance gap turned out to be substantial.

For this purpose, two models were evaluated:
- `Multilingual BERT` (google-bert/bert-base-multilingual-cased)
- `BERTić` (classla/bcms-bertic), a monolingual model specifically designed for South Slavic languages

The evaluation was performed on the `SerbMR-2C dataset`, using `F1-score` as the main metric. The multilingual model reached an F1-score of **0.71**, which is significantly lower than the English benchmark on IMDB **0.93**. However, the monolingual `BERTić` achieved an F1-score of **0.91**, demonstrating that language-specific models can substantially improve sentiment analysis performance in low-resource languages like Serbian.

---

## Project Structure

-   **`src/`**: Contains the Python scripts for the project.
    -   `train_model.py`: Script for training and fine-tuning the model.
    -   `evaluate_models.py`: Script for evaluating and comparing the performance of all trained models.
    -   `inference.py`: Script for running sentiment analysis on new text.
-   **`data/`**: Contains the `SerbMR-2C.csv` dataset used for training.
-   **`models/`**: The folder where the trained models and tokenizers are saved.
-   **`eda_viz/`**: Contains visualizations for exploratory data analysis (EDA), including plots for word and character distributions in reviews.
-   **`error_analysis/`**: Contains analysis of the top 10 most confident prediction errors for each model, helping to understand model weaknesses.
-   **`confusion_matrices/`**: Contains confusion matrices for all trained models to visualize classification performance.

---

## Installation and Setup

### 1. Clone the repository

Open your terminal and clone the project from GitHub.
```bash
    git clone [https://github.com/srdjop/serbian-sentiment-analysis.git]
    cd serbian-sentiment-analysis
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
    **Note**: *torch* and *torchvision* are not included in requirements.txt due to their specific installation requirements. You must install the manual after the other dependencies.
    ```bash
    pip install -r requirements.txt
    ```
    - **For users with an NVIDIA GPU:**
    Install PyTorch with CUDA support. Visit the official PyTorch website https://pytorch.org/get-started/locally/ for the most precise command. Example: 
    ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ```
    - **For users without a GPU or with an AMD/Apple GPU**
    Install the PyTorch CPU-only version. Example:
    ```bash
    pip3 install torch torchvision
    ```

4.  **Training a model (optional):**
    This step is necessary only if you want to train the model again; it will use the pretrained models directly from my Hugging Face Hub profile.

    To train a model, run *train_model.py*. You can specify a model by its name from the Hugging Face Hub.

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
6. **Sentiment Analysis (Inference)**
    Use a trained model to analyze new text. You can add your own sentences to the *texts_to_analyze* list in the script and then run the script.
    The inference script can be used for models loaded directly from the Hugging Face Hub (default) or for those you have saved locally in the models/ folder (need to change the path in the script).

    ```bash
    python src/inference.py
    ```
7. **Viewing EDA and Analysis Results**
- EDA Visualizations: Open plots in `eda_viz/` to explore word and character distributions.
- Error Analysis: Check `error_analysis/` to see the top 10 most confident errors for each model.
- Confusion Matrices: Open images in `confusion_matrices/` to understand model misclassifications.

