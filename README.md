# 🩺 Physician Notetaker

## Overview
**Physician Notetaker** is an AI-powered system designed to assist healthcare professionals by automating the extraction of medical information from patient-physician conversations. The system performs **Medical Named Entity Recognition (NER)** to identify symptoms and treatments, and **Sentiment & Intent Analysis** to understand the patient's emotional state and purpose.

This project was built to address the following key tasks:
1.  **Medical NLP Summarization**: Extracting structured medical details (Symptoms, Treatment, Diagnosis, Prognosis).
2.  **Sentiment & Intent Analysis**: Classifying patient sentiment (e.g., Anxious, Neutral) and intent (e.g., Symptom Reporting, Seeking Reassurance).

## 📂 Project Structure

```
Physician Notetaker/
├── Data/
│   ├── medical_sentiment_intent_dataset.csv  # Dataset for Task 2
│   ├── i2b2/                                 # Original NER dataset
│   └── i2b2_Transformed/                     # Preprocessed NER dataset (BIO format)
├── Models/
│   ├── biobert_ner_model/                    # Fine-tuned BioBERT for NER
│   └── multitask_distilbert/                 # Fine-tuned Multi-task DistilBERT
├── Notebooks/
│   ├── processing.ipynb                      # Data preprocessing (Label mapping)
│   ├── ner_training.ipynb                    # NER training and inference pipeline
│   ├── sent_Int_train.ipynb                  # Sentiment & Intent training pipeline
│   └── sent_int_pred.ipynb                   # Sentiment & Intent prediction demo
├── README.md
└── requirements.txt
```

## 🚀 Setup & Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.
2.  **Install dependencies**:
    Ensure you have Python installed. Install the required libraries using:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Key dependencies include `transformers`, `torch`, `datasets`, `seqeval`, `pandas`, `numpy`, and `scikit-learn`.*

## 🧠 Approach & Methodology

### 1. Medical NLP Summarization (NER)
**Goal**: Extract specific medical entities (Symptoms, Treatments) and generate a structured JSON report.

*   **Model**: We fine-tuned **`dmis-lab/biobert-base-cased-v1.1`**, a BERT model pre-trained on biomedical text (PubMed, PMC), which is highly effective for medical domain tasks.
*   **Data**: The model was trained on a transformed version of the **i2b2 dataset**.
    *   *Preprocessing*: The `processing.ipynb` notebook maps original i2b2 labels (e.g., `problem`, `treatment`, `test`) to our target labels (`SYMPTOM`, `TREATMENT`).
*   **Pipeline**:
    1.  **Token Classification**: The BioBERT model predicts BIO tags (`B-SYMPTOM`, `I-TREATMENT`, etc.) for each token.
    2.  **Entity Extraction**: A custom post-processing function aggregates tokens into complete entity strings.
    3.  **Rule-Based Extraction**: Diagnosis and Prognosis are extracted using keyword matching heuristics (e.g., looking for "diagnosed with", "expected recovery") to complement the NER model.
*   **Evaluation**: The model is evaluated using the **F1-score** (via `seqeval`) rather than standard loss. F1 is a more robust metric for token classification tasks, especially given the class imbalance (prevalence of "O" tags) inherent in NER.
*   **Notebook**: `Notebooks/ner_training.ipynb`

### 2. Sentiment & Intent Analysis
**Goal**: Simultaneously classify the sentiment and intent of a patient's statement.

*   **Model**: A **Multi-Task Learning (MTL)** architecture based on **`distilbert-base-uncased`**.
    *   *Why Multi-Task?* Sentiment and intent are often correlated. Sharing the underlying language representation (BERT encoder) allows the model to learn robust features that benefit both tasks while being more efficient than two separate models.
*   **Architecture**:
    *   **Shared Encoder**: DistilBERT base.
    *   **Heads**: Two separate linear layers (classification heads) on top of the pooled output:
        *   `sentiment_head`: Classifies into Negative, Neutral, Positive.
        *   `intent_head`: Classifies into Symptom Reporting, Appointment Booking, General Query, etc.
*   **Loss Function**: The total loss is the sum of the CrossEntropyLoss for both heads: $L_{total} = L_{sentiment} + L_{intent}$.
*   **Notebooks**:
    *   Training: `Notebooks/sent_Int_train.ipynb`
    *   Prediction: `Notebooks/sent_int_pred.ipynb`

## 💻 Usage

### Data Preparation
If you have raw i2b2 data, run `Notebooks/processing.ipynb` first to generate the `i2b2_Transformed` dataset required for the NER model.

### Training the Models
To retrain the models, execute the cells in the respective notebooks:
1.  **NER Model**: Run `Notebooks/ner_training.ipynb`. This will save the model to `Models/biobert_ner_model`.
2.  **Sentiment/Intent Model**: Run `Notebooks/sent_Int_train.ipynb`. This will save the model to `Models/multitask_distilbert`.

### Inference / Testing
You can test the trained models directly in the notebooks:

**For Medical Extraction:**
Open `ner_training.ipynb` and go to the **Prediction** section.
```python
text = "Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain."
result = build_final_json(text, patient_name="Nitin")
print(result)
```

**For Sentiment & Intent:**
Open `sent_int_pred.ipynb`.
```python
text = "I have severe headache and feel very uncomfortable"
print(predict(text))
# Output: {'text': ..., 'sentiment': 'negative', 'intent': 'symptom_reporting'}
```

## 📊 Results & Limitations
*   **NER**: Successfully extracts symptoms and treatments from clinical text.
    *   **Note**: The current NER model was trained on a **small subset of the i2b2 dataset** for demonstration purposes. Consequently, the accuracy may not be optimal for complex or unseen clinical narratives.
    *   **Improvements**: Performance can be significantly enhanced by:
        *   Training on the full i2b2 or larger biomedical datasets (e.g., BC5CDR, NCBI Disease).
        *   Implementing data augmentation techniques to increase training variety.
        *   Hyperparameter tuning (learning rate, batch size, epochs).
        *   Using larger models like `BioBERT-Large` or `ClinicalBERT`.
*   **Sentiment/Intent**: The multi-task model achieves high F1 scores on the test set, effectively distinguishing between urgent symptom reporting and general queries.
