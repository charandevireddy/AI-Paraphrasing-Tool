import streamlit as st
import torch
import nltk
from transformers import BartForConditionalGeneration, BartTokenizer
from nltk.tokenize import sent_tokenize

# ✅ Download missing NLTK tokenizer data
nltk.download("punkt")

# Set Streamlit page configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# ✅ Load the paraphrasing model with caching to optimize performance
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device

# Load the model and tokenizer
model, tokenizer, device = load_model()

# Function to paraphrase text
def paraphrase_text(text):
    sentences = sent_tokenize(text)  # ✅ NLTK's punkt tokenizer will now work
    paraphrased_sentences = []

    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence, return_tensors="pt", padding="longest", max_length=256, truncation=True
        ).to(device)

        with torch.no_grad():
            output = model.generate(**encoding, max_length=256, num_return_sequences=1)

        paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)

    return " ".join(paraphrased_sentences)

# Streamlit UI
st.title("📝 AI Paraphrasing Tool")
st.write("Enter text below and click 'Paraphrase' to generate a rewritten version.")

# User input
user_input = st.text_area("Enter text to paraphrase:", height=150)

if st.button("Paraphrase"):
    if user_input.strip():
        with st.spinner("Paraphrasing..."):
            paraphrased_output = paraphrase_text(user_input)
        st.subheader("🔹 Paraphrased Text:")
        st.write(paraphrased_output)
    else:
        st.warning("⚠️ Please enter text before clicking 'Paraphrase'.")

# Footer
st.markdown("---")
st.markdown("🔹 Created with [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/).")
