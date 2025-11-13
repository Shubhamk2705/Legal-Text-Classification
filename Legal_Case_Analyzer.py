import re
import joblib
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load trained model
try:
    vectorizer = joblib.load('/kaggle/working/vectorizer.joblib')
    classifier = joblib.load('/kaggle/working/legal_classifier.joblib')
    print("✓ Model loaded successfully!\n")
except:
    print("❌ Model not found. Please train the model first.\n")
    vectorizer = None
    classifier = None

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Model classification
def classify_text(text):
    if not vectorizer or not classifier:
        return None, 0.0, {}
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    pred = classifier.predict(tfidf)[0]
    probs = classifier.predict_proba(tfidf)[0]
    conf = max(probs)
    
    # Get all probabilities
    all_probs = {}
    for idx, label in enumerate(classifier.classes_):
        all_probs[label] = probs[idx]
    
    return pred, conf, all_probs

# Improved summarizer (without keyword detection)
def summarize_text(text, max_sentences=3, max_words=50):
    if not text.strip():
        return "⚠ No content to summarize."
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    summary_sentences = []
    total_words = 0
    for sentence in sentences[:max_sentences]:
        words = sentence.split()
        if total_words + len(words) <= max_words:
            summary_sentences.append(sentence)
            total_words += len(words)
        else:
            remaining = max_words - total_words
            if remaining > 5:
                summary_sentences.append(" ".join(words[:remaining]) + "...")
            break

    summary = ". ".join(summary_sentences)
    if not summary.endswith(('.', '!', '?', '...')):
        summary += "..."
    return summary

# Main analysis function
def analyze_legal_case(text):
    if not text.strip():
        return "⚠ No input text provided."
    
    # Get model prediction
    prediction, confidence, all_probs = classify_text(text)

    # Generate summary
    summary = summarize_text(text)

    # Build output
    output = f"📄 LEGAL TEXT SUMMARY:\n{summary}\n\n"
    output += "="*60 + "\n"
    
    if prediction:
        output += f"💡 CLASSIFICATION: {prediction.upper()}\n"
        output += f"📊 Model Confidence: {confidence*100:.2f}%\n\n"
        output += "All Probabilities:\n"
        for label in sorted(all_probs.keys()):
            bar = "█" * int(all_probs[label] * 20)
            output += f"  {label:12s}: {all_probs[label]*100:5.2f}% {bar}\n"

    return output

# Interactive UI
if vectorizer and classifier:
    text_area = widgets.Textarea(
        value='',
        placeholder='Paste your legal case text here...',
        description='Legal Text:',
        layout=widgets.Layout(width='100%', height='200px')
    )

    analyze_button = widgets.Button(
        description='🔍 Analyze Case',
        button_style='success',
        icon='check',
        layout=widgets.Layout(width='200px', height='40px')
    )

    clear_button = widgets.Button(
        description='🗑️ Clear',
        button_style='warning',
        layout=widgets.Layout(width='200px', height='40px')
    )

    output_widget = widgets.Output()

    def on_analyze_click(b):
        with output_widget:
            clear_output()
            print(analyze_legal_case(text_area.value))

    def on_clear_click(b):
        text_area.value = ''
        with output_widget:
            clear_output()

    analyze_button.on_click(on_analyze_click)
    clear_button.on_click(on_clear_click)

    # Sample texts for quick testing
    sample_selector = widgets.Dropdown(
        options={
            'Select a sample...': '',
            'Sample 1: Cited': 'The Supreme Court cited the landmark case of Brown v. Board of Education in its decision to overturn segregation laws.',
            'Sample 2: Applied': 'The court applied the principles of strict liability and found the defendant responsible for damages.',
            'Sample 3: Referred': 'The judgment referred to several constitutional amendments and discussed their applicability to the case.'
        },
        description='Quick Test:',
        layout=widgets.Layout(width='100%')
    )

    def on_sample_select(change):
        if change['new']:
            text_area.value = change['new']

    sample_selector.observe(on_sample_select, names='value')

    display(widgets.VBox([
        widgets.HTML("<h2 style='color: #007bff;'>⚖️ Legal Case Classifier</h2>"),
        widgets.HTML("<p>Paste your legal text below and click 'Analyze Case' to get classification as <b>CITED</b>, <b>APPLIED</b>, or <b>REFERRED</b>.</p>"),
        sample_selector,
        text_area,
        widgets.HBox([analyze_button, clear_button]),
        output_widget
    ]))
else:
    print("❌ Model not loaded. Cannot create interface.")
