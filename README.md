*1️⃣ Open Kaggle*



*Go to https://www.kaggle.com/*

*.*



*Log in using your account.*



*Click “New Notebook” to create a new Jupyter Notebook environment.*



*You’ll get a coding environment with CPU/GPU options and a /kaggle/working folder where your model files will be saved.*



*2️⃣ Upload the Dataset of Legal Documents*



*In the right sidebar, go to the “Data” section.*



*Click “Add Data” → “Upload”.*



*Select your legal dataset file (.csv or .xlsx) from your computer.*



*Example file: legal\_text\_classification.csv*



*Once it uploads successfully, Kaggle will assign it a dataset path, usually something like:*



*/kaggle/input/legaltextsumm/legal\_text\_classification.csv*



*3️⃣ Copy the Dataset Path*



*After upload, click on the dataset name in the “Data” panel.*



*You’ll see a small folder icon 📁 with a file path (example above).*



*Copy that full path — you’ll need to paste it inside the training code where it says DATASET\_PATH.*



*4️⃣ Paste the “COMPLETE LEGAL CASE CLASSIFICATION PIPELINE” Code*



*In the first cell of your Kaggle notebook, paste the entire training + prediction pipeline code you shared earlier.*



*This script handles:*



*Loading and preprocessing your dataset*



*Cleaning and summarizing text*



*Training a Random Forest classifier*



*Evaluating performance*



*Saving the trained model and TF-IDF vectorizer*



*5️⃣ Add the Dataset Path \& Run*



*Inside that code, find this line:*



*DATASET\_PATH = '/kaggle/input/legaltextsumm/legal\_text\_classification.csv'*





*Replace the path with your actual dataset link if it’s different.*



*Then run the entire cell (Shift + Enter).*



*What happens now:*



*The dataset loads and prepares.*



*The model trains using TF-IDF features.*



*You’ll see accuracy, classification report, and label distribution.*



*Finally, it saves:*



*/kaggle/working/legal\_case\_classifier.joblib*

*/kaggle/working/tfidf\_vectorizer.joblib*





*✅ Once you see messages like:*



*✓ Model saved: /kaggle/working/legal\_case\_classifier.joblib*

*✓ Vectorizer saved: /kaggle/working/tfidf\_vectorizer.joblib*

*PIPELINE COMPLETE!*





*it means your model has been successfully trained.*



*6️⃣ Confirm Model Training*



*Scroll down to the output — you should see model accuracy, classification report, and sample predictions on example cases.*



*Also, check the “Files” tab (on the right of the Kaggle screen) — you’ll find your two .joblib files saved there.*

*That confirms the model was trained and stored properly.*



*7️⃣ Add the “INTERACTIVE LEGAL CASE ANALYZER FOR KAGGLE” Code*



*Now, create a new code cell right below your training code.*



*Paste your second script — “Interactive Legal Case Analyzer for Kaggle.”*



*Make sure it uses:*



*if \_\_name\_\_ == "\_\_main\_\_":*

    *create\_interactive\_analyzer()*





*(with double underscores).*



*This code loads your saved model and vectorizer from /kaggle/working and builds a beautiful interactive interface using ipywidgets.*



*8️⃣ Run the Interactive Analyzer*



*Run the cell.*



*You’ll see an interface appear directly in the notebook with:*



*A text box to enter legal text*



*Example dropdowns (Affirmed, Cited, Applied)*



*Buttons: “Analyze Case” and “Clear”*



*The interface is powered by your trained model.*



*9️⃣ Test with Legal Paragraphs*



*Either:*



*Select an example from the dropdown menu, or*



*Paste your own legal case paragraph into the text box.*

*Example:*



*“The defendant was found liable for negligence and the trial court’s decision was affirmed on appeal.”*



*Click the “Analyze Case” button.*



*🔟 View Final Output*



*The analyzer will instantly show:*



*📄 Full Text (what you entered)*



*🧩 Step 1 – Summarization: key legal outcome extracted (e.g., “Appeal affirmed; defendant liable.”)*



*⚖️ Step 2 – Classification: predicted label (e.g., affirmed, applied, cited, etc.)*



*✅ Confidence score*



*📊 Top 3 predictions with percentage bars*



*You’ll also see a plain text summary printed below the HTML result, useful for copy/paste or logs.*

