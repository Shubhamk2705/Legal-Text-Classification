# ==============================
# INTERACTIVE LEGAL CASE ANALYZER FOR KAGGLE
# Use this in a Kaggle notebook cell
# ==============================

import pandas as pd
import numpy as np
import re
import joblib
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import Layout

# ==============================
# LOAD TRAINED MODEL
# ==============================

print("Loading trained model...")
try:
    vectorizer = joblib.load('/kaggle/working/tfidf_vectorizer.joblib')
    classifier = joblib.load('/kaggle/working/legal_case_classifier.joblib')
    print("✓ Model loaded successfully!")
except:
    print("❌ Model not found. Please train the model first using the training script.")
    vectorizer = None
    classifier = None

# ==============================
# HELPER FUNCTIONS
# ==============================

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_legal_text(text):
    """Create concise summary focusing on key legal outcomes"""
    text_lower = text.lower()
    summary_parts = []
    
    # Appeal outcomes
    if 'appeal' in text_lower:
        if 'dismissed' in text_lower:
            summary_parts.append("Appeal dismissed")
        elif 'granted' in text_lower:
            summary_parts.append("Appeal granted")
        elif 'affirmed' in text_lower:
            summary_parts.append("Appeal affirmed")
        elif 'reversed' in text_lower:
            summary_parts.append("Appeal reversed")
    
    # Court decisions
    if 'trial court' in text_lower:
        if 'affirmed' in text_lower:
            summary_parts.append("trial court decision affirmed")
        elif 'reversed' in text_lower:
            summary_parts.append("trial court decision reversed")
        else:
            summary_parts.append("trial court decision")
    
    # Other outcomes
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

def predict_with_top3(text, vectorizer, classifier):
    """Get top 3 predictions with probabilities"""
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    probs = classifier.predict_proba(text_tfidf)[0]
    outcomes = classifier.classes_
    top3 = sorted(zip(outcomes, probs), key=lambda x: x[1], reverse=True)[:3]
    return top3

def analyze_legal_case(text):
    """Complete analysis pipeline"""
    if not text.strip():
        return None
    
    # Step 1: Summarization
    summary = summarize_legal_text(text)
    
    # Step 2: Classification
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    prediction = classifier.predict(text_tfidf)[0]
    probabilities = classifier.predict_proba(text_tfidf)[0]
    confidence = max(probabilities)
    
    # Top 3 predictions
    top3 = predict_with_top3(text, vectorizer, classifier)
    
    return {
        'full_text': text,
        'summary': summary,
        'prediction': prediction,
        'confidence': confidence,
        'top3': top3
    }

# ==============================
# OUTPUT FORMATTER
# ==============================

def display_results(result):
    """Display results in the requested format"""
    if result is None:
        print("Please enter some text to analyze.")
        return
    
    output_html = f"""
    <style>
        .result-container {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 20px 0;
        }}
        .section {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .full-text {{
            background: #fff;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-style: italic;
            color: #555;
        }}
        .step-box {{
            background: #fff;
            padding: 15px;
            border: 2px solid #28a745;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .step-header {{
            color: #fff;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 10px 15px;
            border-radius: 5px;
            font-weight: bold;
            margin-bottom: 15px;
            display: inline-block;
        }}
        .prediction-label {{
            background: #28a745;
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 18px;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
        }}
        .confidence {{
            color: #666;
            font-size: 14px;
            margin-left: 10px;
        }}
        .top3-container {{
            background: #f1f3f5;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }}
        .top3-item {{
            background: white;
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            border-left: 3px solid #007bff;
        }}
        .probability-bar {{
            background: #e9ecef;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .probability-fill {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.5s ease;
        }}
    </style>
    
    <div class="result-container">
        <div class="section">
            <div class="section-title">📄 Full Text:</div>
            <div class="full-text">"{result['full_text']}"</div>
        </div>
        
        <div class="section">
            <div class="step-header">Step 1 – Summarization</div>
            <div class="step-box">
                <strong>"{result['summary']}"</strong>
            </div>
        </div>
        
        <div class="section">
            <div class="step-header">Step 2 – Classification</div>
            <div class="step-box">
                <div>Predicted Label: <span class="prediction-label">{result['prediction']}</span></div>
                <div class="confidence">Confidence: {result['confidence']*100:.2f}%</div>
                
                <div class="top3-container">
                    <strong>Top 3 Predictions:</strong>
    """
    
    for i, (outcome, prob) in enumerate(result['top3'], 1):
        output_html += f"""
                    <div class="top3-item">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span><strong>{i}. {outcome}</strong></span>
                            <span>{prob*100:.2f}%</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: {prob*100}%"></div>
                        </div>
                    </div>
        """
    
    output_html += """
                </div>
            </div>
        </div>
    </div>
    """
    
    display(HTML(output_html))
    
    # Also print plain text version
    print("\n" + "="*70)
    print("PLAIN TEXT OUTPUT")
    print("="*70)
    print(f'\nFull Text:\n"{result["full_text"]}"')
    print(f'\nStep 1 – Summarization:\n"{result["summary"]}"')
    print(f'\nStep 2 – Classification:')
    print(f'Predicted Label: `{result["prediction"]}`')
    print(f'Confidence: {result["confidence"]*100:.2f}%')
    print(f'\nTop 3 Predictions:')
    for i, (outcome, prob) in enumerate(result['top3'], 1):
        print(f'  {i}. {outcome}: {prob*100:.2f}%')
    print("="*70)

# ==============================
# INTERACTIVE WIDGET INTERFACE
# ==============================

def create_interactive_analyzer():
    """Create interactive widget for Kaggle notebook"""
    
    if vectorizer is None or classifier is None:
        print("❌ Model not loaded. Cannot create interface.")
        return
    
    # Text input widget
    text_input = widgets.Textarea(
        value='',
        placeholder='Paste your legal case text here...',
        description='Legal Text:',
        layout=Layout(width='100%', height='200px')
    )
    
    # Example cases
    examples = {
        'Example 1: Affirmed': "Ordinarily that discretion will be exercised in the manner indicated by the court in previous judgments. In this case, the appeal was dismissed as the trial court's decision was affirmed.",
        'Example 2: Cited': "This case has been cited in multiple subsequent judgments as a key precedent for intellectual property disputes and trademark law applications.",
        'Example 3: Applied': "The defendant was found liable for breach of contract and the court applied the relevant precedents established in previous commercial litigation cases."
    }
    
    example_dropdown = widgets.Dropdown(
        options=['Select an example...'] + list(examples.keys()),
        description='Examples:',
        layout=Layout(width='400px')
    )
    
    # Buttons
    analyze_button = widgets.Button(
        description='Analyze Case',
        button_style='success',
        icon='check',
        layout=Layout(width='200px', height='40px')
    )
    
    clear_button = widgets.Button(
        description='Clear',
        button_style='warning',
        icon='times',
        layout=Layout(width='150px', height='40px')
    )
    
    # Output area
    output = widgets.Output()
    
    # Event handlers
    def on_analyze_click(b):
        with output:
            clear_output()
            text = text_input.value
            if text.strip():
                print("🔄 Analyzing...")
                result = analyze_legal_case(text)
                display_results(result)
            else:
                print("⚠️ Please enter some text to analyze.")
    
    def on_clear_click(b):
        text_input.value = ''
        with output:
            clear_output()
    
    def on_example_change(change):
        if change['new'] != 'Select an example...':
            text_input.value = examples[change['new']]
    
    analyze_button.on_click(on_analyze_click)
    clear_button.on_click(on_clear_click)
    example_dropdown.observe(on_example_change, names='value')
    
    # Layout
    print("="*70)
    print("LEGAL CASE ANALYZER - INTERACTIVE INTERFACE")
    print("="*70)
    print("\n")
    
    display(widgets.VBox([
        widgets.HTML("<h2 style='color: #007bff;'>📚 Legal Case Classification & Summarization</h2>"),
        example_dropdown,
        text_input,
        widgets.HBox([analyze_button, clear_button]),
        output
    ]))

# ==============================
# SIMPLE FUNCTION FOR DIRECT USE
# ==============================

def analyze_case_text(text):
    """
    Simple function to analyze legal case text
    Usage: analyze_case_text("Your legal text here...")
    """
    if vectorizer is None or classifier is None:
        print("❌ Model not loaded. Please train the model first.")
        return None
    
    result = analyze_legal_case(text)
    display_results(result)
    return result

# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    # Create interactive interface
    create_interactive_analyzer()
    
    print("\n💡 TIP: You can also use: analyze_case_text('your text here') for direct analysis")