import re
from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch.nn.functional as F

app = Flask(__name__)

# --- Load Model ---
model_dir = "model"  # Direktori Model 
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.eval()

# Model Pendukung
roberta_sentiment = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
distilbert_check = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Label Mapping
label_map = {0: "Neutral", 1: "Excited", 2: "Not Excited"}

# --- Preprocessing ---
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s!?]', '', text)  # Hapus karakter khusus kecuali ! dan ?
    return text

# --- Prediksi dengan Ensembling Sederhana ---
def predict_comment(comment):
    processed_text = preprocess_text(comment)
    
    # Prediksi dari BERT (model utama)
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        bert_conf = float(torch.max(probs))
        bert_label = label_map[torch.argmax(probs).item()]
    
    # Model add on
    roberta_result = roberta_sentiment(processed_text[:512])[0]
    roberta_label = "Excited" if roberta_result['label'] == 'POSITIVE' else "Not Excited" if roberta_result['label'] == 'NEGATIVE' else "Neutral"

    distil_result = distilbert_check(processed_text[:512])[0]
    distil_label = "Excited" if distil_result['label'] == 'POSITIVE' else "Not Excited" if distil_result['label'] == 'NEGATIVE' else "Neutral"
    
    # Voting System: Ambil prediksi mayoritas
    votes = [bert_label, roberta_label, distil_label]
    final_label = max(set(votes), key=votes.count)
    
    # Jika hasil BERT dan RoBERTa sama, tingkatkan confidence
    if bert_label == roberta_label:
        bert_conf = min(bert_conf + 0.1, 1.0)
    
    return final_label, bert_conf, "Normal Predicted"

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = confidence = note = None
    if request.method == "POST":
        comment = request.form.get("comment", "")
        if comment.strip():
            prediction, confidence, note = predict_comment(comment)
    return render_template("index.html", prediction=prediction, confidence=confidence, note=note)

if __name__ == "__main__":
    app.run(debug=True)
