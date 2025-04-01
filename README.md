# Spam SMS Detection

## Overview
This project aims to develop a machine learning model to classify SMS messages as spam or non-spam. The goal is to build a robust system that accurately identifies spam messages while minimizing false positives.

## Dataset
- The dataset contains SMS messages labeled as:
  - **Spam (1)**: Unwanted promotional or fraudulent messages
  - **Ham (0)**: Legitimate messages
- Ensure the dataset (`sms_data.csv`) is placed in the `dataset/` folder.

## Project Structure
```
spam-sms-detection/
│── dataset/              # Dataset files
│── models/               # Trained models
│── notebooks/            # Jupyter notebooks (if any)
│── src/                  # Python scripts
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── .gitignore            # Ignore unnecessary files
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/spam-sms-detection.git
   cd spam-sms-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Preprocess and Train Models**
   Run the `spam_detection.py` script to train models:
   ```bash
   python src/spam_detection.py
   ```
   This will save trained models in the `models/` directory.

2. **Make Predictions**
   Load a trained model and predict whether an SMS message is spam:
   ```python
   import joblib
   
   model = joblib.load('models/spam_model.pkl')
   vectorizer = joblib.load('models/vectorizer.pkl')
   new_sms = ["Congratulations! You've won a free iPhone. Click here to claim."]
   X_new = vectorizer.transform(new_sms)
   prediction = model.predict(X_new)
   print("Spam" if prediction[0] == 1 else "Not Spam")
   ```

## Models Used
- **Naïve Bayes Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

## Evaluation Metrics
- **Accuracy**
- **Precision, Recall & F1 Score**
- **Confusion Matrix Analysis**

## Contributions
Feel free to contribute by submitting pull requests or opening issues.

## License
This project is licensed under the MIT License.
