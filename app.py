import os
import streamlit as st
import torch
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# ✅ Set Streamlit Page Configuration (Title & Layout)
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# ✅ Set a fixed directory for NLTK data (Prevents Lookup Errors)
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")  
nltk.data.path.append(NLTK_DIR)

# ✅ Download the required NLTK package (Ensures sentence tokenization works)
nltk.download('punkt', download_dir=NLTK_DIR)

# ✅ Load the Paraphrasing Model (Caches for performance)
@st.cache_resource
def load_model():
    # 🔹 Load the T5-based paraphrasing model from Hugging Face
    model_name = "Vamsi/T5_Paraphrase_Paws"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 🔹 Force model to run on CPU (Streamlit Cloud does not support GPU)
    device = torch.device("cpu")
    model.to(device)

    return model, tokenizer, device

# ✅ Load the model when the app starts
model, tokenizer, device = load_model()

# ✅ Function to Paraphrase Text
def paraphrase_text(text):
    if not text.strip():  # 🔹 Prevents empty input errors
        return "Please enter text to paraphrase."

    sentences = sent_tokenize(text)  # 🔹 Sentence tokenization using NLTK
    paraphrased_sentences = []

    for sentence in sentences:
        # 🔹 Prepares the text in the correct format for the model
        input_text = f"paraphrase: {sentence} </s>"
        encoding = tokenizer.encode_plus(
            input_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True
        )

        # 🔹 Move input tensors to CPU
        input_ids = encoding["input_ids"].to("cpu")
        attention_mask = encoding["attention_mask"].to("cpu")

        # 🔹 Generate paraphrased output using the model
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=128, num_return_sequences=1
        )

        # 🔹 Decode the generated text into a readable format
        paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)

    return " ".join(paraphrased_sentences)  # 🔹 Join sentences into final output

# ✅ Streamlit UI (Frontend)
st.title("AI Paraphrasing Tool")  # 🔹 App Title
st.write("Enter text below and click **Paraphrase** to generate a new version.")  # 🔹 Instructions

# 🔹 User input field for text
user_input = st.text_area("Enter your text here:", height=150)

# 🔹 When the "Paraphrase" button is clicked
if st.button("Paraphrase"):
    with st.spinner("Paraphrasing... ⏳"):  # 🔹 Loading spinner for better UX
        paraphrased_output = paraphrase_text(user_input)  # 🔹 Get paraphrased text
        st.subheader("Paraphrased Text")  # 🔹 Display paraphrased text
        st.write(paraphrased_output)

# ✅ Footer Section
st.markdown("---")
st.markdown("🚀 Built with [Streamlit](https://streamlit.io/) & [Hugging Face Transformers](https://huggingface.co/)")

