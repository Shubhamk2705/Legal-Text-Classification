1️⃣ Open Kaggle

Go to kaggle.com

Log in

Click New Notebook

You now have a Jupyter environment with:

CPU/GPU options

/kaggle/working → model files will be saved here

2️⃣ Upload Your Legal Dataset

Right sidebar → Data → Add Data → Upload

Upload:
legal_text_classification.csv (or your dataset)

After uploading, Kaggle gives a path like:

/kaggle/input/legaltextsumm/legal_text_classification.csv

3️⃣ Copy Dataset Path

In the Data panel, click the dataset name

You will see a folder icon 📁 with the full path

Copy that path

You will paste this inside the training code.

4️⃣ FIRST FILE → Run legal_case_class.py (Training Pipeline)
Create your first Kaggle notebook cell.

Paste the entire code from your file:

legal_case_class.py


This code will:

✔ Load dataset
✔ Clean text
✔ Summarize legal text
✔ Train TF-IDF + RandomForest
✔ Show accuracy + classification report
✔ Save:

/kaggle/working/legal_case_classifier.joblib
/kaggle/working/tfidf_vectorizer.joblib

5️⃣ Add Dataset Path & Run

Look for this line in your code:

DATASET_PATH = '/kaggle/input/legaltextsumm/legal_text_classification.csv'


Replace it with your actual path.

Then Run the entire cell (Shift + Enter).

You will see:
✓ Model saved: /kaggle/working/legal_case_classifier.joblib
✓ Vectorizer saved: /kaggle/working/tfidf_vectorizer.joblib
PIPELINE COMPLETE!

6️⃣ Confirm Training

Check:

✔ Output metrics

Accuracy

Confusion matrix

Classification report

✔ Files tab

You should see:

legal_case_classifier.joblib

tfidf_vectorizer.joblib

7️⃣ SECOND FILE → Run legal_case_analyzer.py (Interactive Analyzer)
Create a new cell below.

Paste the code from your second Python file:

legal_case_analyzer.py


Ensure the last line uses double underscores:

if __name__ == "__main__":
    create_interactive_analyzer()


This script will:

✔ Load your saved model
✔ Load TF-IDF vectorizer
✔ Create an interactive UI using ipywidgets
✔ Handle summarization + prediction

8️⃣ Run the Analyzer Code

Run the cell.

You will now see a live UI:

Text Input Box

Example Dropdown

Analyze Case Button

Clear Button

9️⃣ Test With Legal Text

Enter any paragraph.

Example:

The trial court's decision was affirmed as the defendant was found liable for negligence.


Click Analyze Case.

🔟 View the Final Model Output

The analyzer shows:

📄 Full Text

Your entered paragraph.

🧩 Step 1 — Summarization

Key legal outcome extracted.

⚖️ Step 2 — Classification

Predicted label:

Affirmed

Cited

Applied

Reversed

Dismissed

etc.

📈 Confidence Scores

Top 3 predictions with probability bars.

📝 Plain-text Output

Printed below for easy copying.
