import streamlit as st
from transformers import pipeline

# Get transformer model and set up a pipeline
model_ckpt = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt)

labels = {"ar" : "Arabic", "bg" : "Bulgarian", "de" : "German", "el" : "Modern Greek", 
"en" : "English", "es" : "Spanish", "fr" : "French", "hi" : "Hindi", "it" : "Italian", 
"ja" : "Japanese", "nl" : "Dutch", "pl" : "Polish", "pt" : "Portuguese", "ru" : "Russian", 
"sw" : "Swahili", "th" : "Thai", "tr" : "Turkish", "ur" : "Urdu", "vi" : "Vietnamese", "zh" : "Chinese"}


def predict(text: str) -> dict:
    """Compute predictions for text."""
    preds = pipe(text, return_all_scores=True, truncation=True, max_length=128)
    if preds:
        pred = preds[0]
        return {labels.get(p["label"],p["label"]): float(p["score"]) for p in pred}
    else:
        return None

st.title("Language detection with XLM-RoBERTa")
st.write("Determine the language in which your text is written.")
text = st.text_area("Text", "Enter your text here")
if text:
    results = predict(text)
    top_result = max(results, key=results.get)
    st.write(f"Your text is written in {top_result}")
    st.bar_chart(results)
