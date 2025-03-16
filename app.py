import streamlit as st
import nltk
import os
import asyncio
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# ✅ Fix NLTK 'punkt' Error
nltk_data_path = os.path.expanduser("~/.nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)  # Ensure directory exists
nltk.data.path.append(nltk_data_path)  # Add path

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

# ✅ Fix "RuntimeError: no running event loop"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Load Paraphrasing Model
paraphraser = pipeline("text2text-generation", model="t5-small")

# ✅ Streamlit UI
st.title("🔄 AI Paraphrasing Tool")
st.subheader("Enter text to paraphrase")

user_input = st.text_area("Your Text Here", height=150)
input_word_count = len(user_input.split())

# ✅ Function to paraphrase text
def paraphrase_text(text):
    sentences = sent_tokenize(text)
    paraphrased_sentences = []
    
    for sentence in sentences:
        result = paraphraser(f"paraphrase: {sentence}", max_length=100, truncation=True)
        paraphrased_sentences.append(result[0]['generated_text'])
    
    return " ".join(paraphrased_sentences)

# ✅ Generate paraphrased text
if st.button("Paraphrase"):
    if user_input.strip():
        paraphrased_output = paraphrase_text(user_input)
        output_word_count = len(paraphrased_output.split())

        st.subheader("Paraphrased Text:")
        st.write(paraphrased_output)

        # ✅ Show Word Count
        st.info(f"📌 **Original Word Count:** {input_word_count}")
        st.success(f"✅ **Paraphrased Word Count:** {output_word_count}")
    else:
        st.error("❌ Please enter text to paraphrase!")
