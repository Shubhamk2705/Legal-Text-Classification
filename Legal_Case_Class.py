# ==============================
# COMPLETE LEGAL CASE CLASSIFICATION PIPELINE
# Training + Prediction with Summarization
# ==============================

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==============================
# STEP 1: DATA LOADING & PREPROCESSING
# ==============================

def load_email_dataset(file_path):
    """Load the email dataset from CSV or Excel"""
    print("Loading dataset...")
    
    # Check file extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv, .xlsx, or .xls")
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst row preview:")
    print(df.head(1))
    return df

def detect_dataset_structure(df):
    """
    Detect if dataset has word frequencies or direct text columns
    """
    # Check for common text column names
    text_columns = ['text', 'full_text', 'case_text', 'legal_text', 'content', 'description', 'summary']
    label_columns = ['label', 'outcome', 'classification', 'category', 'class']
    
    found_text_col = None
    found_label_col = None
    
    # Case-insensitive search for text column
    for col in df.columns:
        col_lower = col.lower()
        if any(text_name in col_lower for text_name in text_columns):
            found_text_col = col
            break
    
    # Case-insensitive search for label column
    for col in df.columns:
        col_lower = col.lower()
        if any(label_name in col_lower for label_name in label_columns):
            found_label_col = col
            break
    
    # If no text column found, check if it's word frequency format
    if found_text_col is None:
        # Check if most columns are numeric (word frequencies)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > len(df.columns) * 0.5:
            return 'word_frequency', None, None
    
    return 'direct_text', found_text_col, found_label_col

def create_full_text_from_word_counts(row):
    """
    Reconstruct email text from word frequency columns
    Each column represents a word, value represents frequency
    """
    # Skip the first column (Email No.)
    word_columns = row.index[1:]
    
    email_words = []
    for word in word_columns:
        count = int(row[word]) if pd.notna(row[word]) and row[word] > 0 else 0
        # Repeat word based on its frequency
        email_words.extend([word] * count)
    
    return ' '.join(email_words)

def extract_label_from_text(text):
    """
    Extract classification label from email content
    Based on legal case outcomes
    """
    text_lower = text.lower()
    
    # Priority-based keyword matching
    if 'affirmed' in text_lower:
        return 'affirmed'
    elif 'dismissed' in text_lower:
        return 'dismissed'
    elif 'reversed' in text_lower:
        return 'reversed'
    elif 'remanded' in text_lower:
        return 'remanded'
    elif 'granted' in text_lower:
        return 'granted'
    elif 'denied' in text_lower:
        return 'denied'
    elif 'cited' in text_lower or 'precedent' in text_lower:
        return 'cited'
    elif 'applied' in text_lower or 'liable' in text_lower:
        return 'applied'
    else:
        return 'other'

def prepare_training_data(df):
    """Prepare dataset for training - handles both word frequency and direct text formats"""
    print("\nPreparing training data...")
    
    # Detect dataset structure
    dataset_type, text_col, label_col = detect_dataset_structure(df)
    
    print(f"Dataset type detected: {dataset_type}")
    
    if dataset_type == 'direct_text':
        # Dataset already has text and possibly labels
        print(f"Text column: {text_col}")
        print(f"Label column: {label_col}")
        
        if text_col is None:
            # If no obvious text column, use the first string column
            string_cols = df.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                text_col = string_cols[0]
                print(f"Using first string column as text: {text_col}")
            else:
                raise ValueError("No text column found in dataset!")
        
        df['full_text'] = df[text_col].astype(str)
        
        # Handle labels
        if label_col is not None:
            df['label'] = df[label_col].astype(str)
        else:
            # Extract labels from text if not provided
            print("No label column found. Extracting labels from text...")
            df['label'] = df['full_text'].apply(extract_label_from_text)
    
    else:  # word_frequency format
        # Create full text from word frequencies
        df['full_text'] = df.apply(create_full_text_from_word_counts, axis=1)
        
        # Extract labels
        df['label'] = df['full_text'].apply(extract_label_from_text)
    
    # Remove empty texts
    df = df[df['full_text'].str.strip().str.len() > 0]
    
    print(f"\nLabel Distribution:")
    print(df['label'].value_counts())
    print(f"\nTotal samples: {len(df)}")
    
    # Show sample
    print(f"\nSample text:")
    print(df['full_text'].iloc[0][:200] + "...")
    
    return df[['full_text', 'label']]

# ==============================
# STEP 2: TEXT CLEANING
# ==============================

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================
# STEP 3: TEXT SUMMARIZATION
# ==============================

def summarize_legal_text(text):
    """
    Create concise summary focusing on key legal outcomes and decisions
    """
    text_lower = text.lower()
    summary_parts = []
    
    # Check for appeal outcomes
    if 'appeal' in text_lower:
        if 'dismissed' in text_lower:
            summary_parts.append("Appeal dismissed")
        elif 'granted' in text_lower:
            summary_parts.append("Appeal granted")
        elif 'affirmed' in text_lower:
            summary_parts.append("Appeal affirmed")
        elif 'reversed' in text_lower:
            summary_parts.append("Appeal reversed")
    
    # Check for court decisions
    if 'trial court' in text_lower:
        if 'affirmed' in text_lower:
            summary_parts.append("trial court decision affirmed")
        elif 'reversed' in text_lower:
            summary_parts.append("trial court decision reversed")
        else:
            summary_parts.append("trial court decision")
    
    # Check for other outcomes
    if 'damages' in text_lower and 'awarded' in text_lower:
        summary_parts.append("damages awarded")
    
    if 'remanded' in text_lower:
        summary_parts.append("case remanded")
    
    if 'defendant' in text_lower and 'liable' in text_lower:
        summary_parts.append("defendant liable")
    
    # If we found key points, join them
    if summary_parts:
        return '; '.join(summary_parts) + '.'
    
    # Fallback: extract first meaningful sentence
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        if len(sentence.strip()) > 20:
            return sentence.strip()[:100] + '.'
    
    return text[:100] + '...'

# ==============================
# STEP 4: MODEL TRAINING
# ==============================

def train_model(X_train, y_train, X_test, y_test):
    """Train the classification model"""
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*60)
    
    # TF-IDF Vectorization
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return vectorizer, classifier

def save_model(vectorizer, classifier):
    """Save trained model and vectorizer"""
    print("\nSaving model and vectorizer...")
    joblib.dump(classifier, '/kaggle/working/legal_case_classifier.joblib')
    joblib.dump(vectorizer, '/kaggle/working/tfidf_vectorizer.joblib')
    print("✓ Model saved: /kaggle/working/legal_case_classifier.joblib")
    print("✓ Vectorizer saved: /kaggle/working/tfidf_vectorizer.joblib")

# ==============================
# STEP 5: PREDICTION PIPELINE
# ==============================

def predict_with_summarization(text, vectorizer, classifier):
    """
    Complete pipeline: Summarize text and classify
    Outputs in the exact format requested
    """
    print("\n" + "="*60)
    print("PROCESSING LEGAL CASE")
    print("="*60)
    
    # Display full text
    print(f'\nFull Text:\n"{text}"')
    
    # Step 1: Summarization
    summary = summarize_legal_text(text)
    print(f'\nStep 1 – Summarization:\n"{summary}"')
    
    # Step 2: Classification
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    prediction = classifier.predict(text_tfidf)[0]
    probabilities = classifier.predict_proba(text_tfidf)[0]
    confidence = max(probabilities)
    
    print(f'\nStep 2 – Classification:')
    print(f'Predicted Label: `{prediction}`')
    print(f'Confidence: {confidence:.4f} ({confidence*100:.2f}%)')
    
    return summary, prediction, confidence

def predict_with_top3(text, vectorizer, classifier):
    """Prediction with top 3 outcomes"""
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    probs = classifier.predict_proba(text_tfidf)[0]
    outcomes = classifier.classes_
    top3 = sorted(zip(outcomes, probs), key=lambda x: x[1], reverse=True)[:3]
    return top3

# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    
    print("="*60)
    print("LEGAL CASE CLASSIFICATION & SUMMARIZATION PIPELINE")
    print("="*60)
    
    # CONFIGURATION - Your actual dataset path
    DATASET_PATH = '/kaggle/input/legaltextsumm/legal_text_classification.csv'
    
    # ============================================
    # PART A: TRAINING
    # ============================================
    
    print("\n[PHASE 1: MODEL TRAINING]")
    print("-"*60)
    
    # 1. Load dataset
    df = load_email_dataset(DATASET_PATH)
    
    # 2. Prepare data
    processed_df = prepare_training_data(df)
    
    # 3. Clean text
    processed_df['cleaned_text'] = processed_df['full_text'].apply(clean_text)
    
    # 4. Split data
    X = processed_df['cleaned_text'].values
    y = processed_df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 5. Train model
    vectorizer, classifier = train_model(X_train, y_train, X_test, y_test)
    
    # 6. Save model
    save_model(vectorizer, classifier)
    
    # ============================================
    # PART B: DEMONSTRATION
    # ============================================
    
    print("\n" + "="*60)
    print("[PHASE 2: PREDICTION DEMONSTRATION]")
    print("="*60)
    
    # Example texts for testing
    example_cases = [
        {
            "text": "Ordinarily that discretion will be exercised in the manner indicated by the court in previous judgments. In this case, the appeal was dismissed as the trial court's decision was affirmed.",
            "expected": "affirmed"
        },
        {
            "text": "The defendant was found liable for breach of contract and the court applied the relevant precedents.",
            "expected": "applied"
        },
        {
            "text": "This case has been cited in multiple subsequent judgments as a key precedent for intellectual property disputes.",
            "expected": "cited"
        }
    ]
    
    for i, case in enumerate(example_cases, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}")
        print('='*60)
        summary, prediction, confidence = predict_with_summarization(
            case['text'], 
            vectorizer, 
            classifier
        )
        
        # Show top 3 predictions
        top3 = predict_with_top3(case['text'], vectorizer, classifier)
        print(f'\nTop 3 Predictions:')
        for rank, (outcome, prob) in enumerate(top3, 1):
            print(f"  {rank}. {outcome}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nModel is ready for use!")
    print("\nTo use with new text:")
    print(">>> summary, label, conf = predict_with_summarization(your_text, vectorizer, classifier)")
    print("="*60)
    
    # ============================================
    # PART C: FUNCTION FOR NEW PREDICTIONS
    # ============================================
    
    def analyze_new_case(text):
        """
        Use this function to analyze new legal cases
        """
        # Load model if not in memory
        try:
            vec = vectorizer
            clf = classifier
        except:
            vec = joblib.load('/kaggle/working/tfidf_vectorizer.joblib')
            clf = joblib.load('/kaggle/working/legal_case_classifier.joblib')
        
        summary, prediction, confidence = predict_with_summarization(text, vec, clf)
        top3 = predict_with_top3(text, vec, clf)
        
        return {
            'summary': summary,
            'prediction': prediction,
            'confidence': confidence,
            'top3': top3
        }
    
    print("\n✓ Use analyze_new_case(text) function to analyze new legal cases")