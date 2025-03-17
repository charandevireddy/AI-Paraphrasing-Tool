import os
import streamlit as st
import torch
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# ✅ Set a fixed directory for NLTK data
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DIR)

# ✅ Download the required NLTK package
nltk.download('punkt', download_dir=NLTK_DIR)

# ✅ Load the Paraphrasing Model
@st.cache_resource
def load_model():
    model_name = "Vamsi/T5_Paraphrase_Paws"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Force model to run on CPU (Streamlit Cloud usually lacks GPU support)
    device = torch.device("cpu")
    model.to(device)

    return model, tokenizer, device

model, tokenizer, device = load_model()

# ✅ Function to Paraphrase Text
def paraphrase_text(text):
    if not text.strip():
        return "Please enter text to paraphrase."

    sentences = sent_tokenize(text)  # ✅ NLTK Sentence Tokenization
    paraphrased_sentences = []

    for sentence in sentences:
        input_text = f"paraphrase: {sentence} </s>"
        encoding = tokenizer.encode_plus(
            input_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True
        )

        # Move tensors to CPU
        input_ids = encoding["input_ids"].to("cpu")
        attention_mask = encoding["attention_mask"].to("cpu")

        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=128, num_return_sequences=1
        )

        paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)

    return " ".join(paraphrased_sentences)

# ✅ Streamlit UI
st.title("AI Paraphrasing Tool")
st.write("Enter text below and click **Paraphrase** to generate a new version.")

user_input = st.text_area("Enter your text here:", height=150)
if st.button("Paraphrase"):
    with st.spinner("Paraphrasing... ⏳"):
        paraphrased_output = paraphrase_text(user_input)
        st.subheader("Paraphrased Text")
        st.write(paraphrased_output)

# ✅ Footer
st.markdown("---")
st.markdown("🚀 Built with [Streamlit](https://streamlit.io/) & [Hugging Face Transformers](https://huggingface.co/)")

