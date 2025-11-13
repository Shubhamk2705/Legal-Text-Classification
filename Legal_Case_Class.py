import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CONFIGURATION
# ==============================
DATASET_PATH = '/kaggle/input/dataset/legal_text_classification_numbered_final.csv'

# Label mapping
LABEL_MAP = {
    0: 'cited',
    1: 'applied',
    2: 'referred',
    -1: 'other'
}

# ==============================
# DATA LOADING
# ==============================
def load_data(file_path):
    """Load CSV dataset"""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {len(df)} rows")
    return df

# ==============================
# TEXT PREPROCESSING
# ==============================
def clean_text(text):
    """Clean and normalize text"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def prepare_data(df):
    """Prepare training data"""
    # Map numeric labels to text labels
    df['label'] = df['numeric_label'].map(LABEL_MAP)
    
    # Combine case_title and case_text
    df['text'] = df['case_title'].fillna('') + ' ' + df['case_text'].fillna('')
    
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 10]
    
    # Remove rows with NaN labels (unmapped numeric_label values)
    df = df.dropna(subset=['label'])
    
    # Optional: Remove 'other' class if you want
    # df = df[df['numeric_label'] != -1]
    
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nTotal valid samples: {len(df)}")
    
    return df[['text', 'label', 'numeric_label']]

# ==============================
# MODEL TRAINING
# ==============================
def train_classifier(X_train, y_train, X_test, y_test):
    """Train the classification model with multiple epochs"""
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    
    # Step 1: Vectorization (Convert text to numbers)
    print("\nStep 1: Converting text to numerical features...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"✓ Vectorization complete. Feature size: {X_train_vec.shape[1]}")
    
    # Step 2: Model Training with iterations
    print("\nStep 2: Training classifier...")
    print("-" * 60)
    
    # Using Logistic Regression with iterations (similar to epochs)
    model = LogisticRegression(
        max_iter=1000,      # Number of iterations/epochs
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
        verbose=1,          # Show training progress
        class_weight='balanced'
    )
    
    # Train the model
    model.fit(X_train_vec, y_train)
    
    print("\n✓ Training completed!")
    
    # Step 3: Evaluation
    print("\nStep 3: Evaluating model...")
    print("-" * 60)
    
    # Training accuracy
    train_pred = model.predict(X_train_vec)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {train_acc*100:.2f}%")
    
    # Test accuracy
    test_pred = model.predict(X_test_vec)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    return vectorizer, model, test_pred

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_case(text, vectorizer, model):
    """Predict label for new case text"""
    # Clean the text
    cleaned = clean_text(text)
    
    # Vectorize
    text_vec = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    # Get confidence scores for all classes
    results = {}
    for idx, label in enumerate(model.classes_):
        results[label] = probabilities[idx]
    
    return prediction, results

# ==============================
# MAIN EXECUTION
# ==============================

print("="*60)
print("LEGAL CASE CLASSIFICATION SYSTEM")
print("="*60)

# 1. Load dataset
print("\n[1/6] Loading dataset...")
df = load_data(DATASET_PATH)

# 2. Prepare data
print("\n[2/6] Preparing data...")
df_clean = prepare_data(df)

# 3. Split data
print("\n[3/6] Splitting data into train and test sets...")
X = df_clean['text'].values
y = df_clean['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 4. Train model
print("\n[4/6] Training model...")
vectorizer, model, y_pred = train_classifier(X_train, y_train, X_test, y_test)

# 5. Save model
print("\n[5/6] Saving model...")
joblib.dump(model, '/kaggle/working/legal_classifier.joblib')
joblib.dump(vectorizer, '/kaggle/working/vectorizer.joblib')
print("✓ Model saved successfully!")

# 6. Detailed evaluation
print("\n[6/6] Detailed Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# TEST PREDICTIONS
# ==============================
print("\n" + "="*60)
print("TESTING PREDICTIONS")
print("="*60)

test_cases = [
    "The court cited the precedent from Smith v. Jones case in its decision.",
    "The legal principles were applied to determine liability in this matter.",
    "The judgment referred to multiple constitutional provisions and amendments.",
    "This case involved breach of contract and damages were awarded."
]

for i, test_text in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Text: {test_text[:80]}...")
    
    prediction, probabilities = predict_case(test_text, vectorizer, model)
    
    print(f"\nPrediction: {prediction.upper()}")
    print(f"Confidence: {probabilities[prediction]*100:.2f}%")
    print("\nAll probabilities:")
    for label in sorted(probabilities.keys()):
        print(f"  {label:12s}: {probabilities[label]*100:.2f}%")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
