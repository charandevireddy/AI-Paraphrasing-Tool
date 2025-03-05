import streamlit as st
import torch
import nltk
import os
import pyperclip
import asyncio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# Ensure an event loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set Streamlit page configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# Fix for Streamlit Watchdog issue
os.environ["STREAMLIT_WATCHDOG"] = "false"

# Ensure NLTK tokenizer (punkt) is downloaded
nltk_data_path = os.path.expanduser("~/.nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)
nltk.data.path.extend([
    "/usr/share/nltk_data",
    "/usr/local/share/nltk_data",
    "/usr/lib/nltk_data",
    "/usr/local/lib/nltk_data"
])

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.data.find('tokenizers/punkt')  # Ensure it is loaded

# Load the paraphrasing model
@st.cache_resource
def load_model():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Function to paraphrase text
def paraphrase_text(text):
    sentences = sent_tokenize(text)
    paraphrased_sentences = []

    for sentence in sentences:
        input_text = f"paraphrase: {sentence} </s>"
        encoding = tokenizer.encode_plus(
            input_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)

    return " ".join(paraphrased_sentences)

# Streamlit app layout
st.title("Paraphrasing Tool")
st.write("Enter a paragraph below to generate a paraphrased version.")

# Sidebar for instructions
st.sidebar.header("Instructions")
st.sidebar.write(
    "1. Enter your text in the box.\n"
    "2. Click 'Paraphrase' to generate a paraphrased version.\n"
    "3. Click 'Clear Text' to reset the input field.\n"
    "4. Click 'Copy Text' to copy the paraphrased text.\n"
    "5. Check the word count before and after paraphrasing."
)

# User input for the paragraph
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Enter Text", height=150)
    word_count = len(user_input.split()) if user_input.strip() else 0
    st.write(f"**Word Count:** {word_count}")

with col2:
    if st.button("Clear Text"):
        st.experimental_rerun()  # Clears the input field

# Generate paraphrased text
if st.button("Paraphrase"):
    if user_input.strip():
        paraphrased_output = paraphrase_text(user_input)
        output_word_count = len(paraphrased_output.split())

        st.subheader("Paraphrased Text:")
        st.text_area("Output", value=paraphrased_output, height=150, key="output")
        st.write(f"**Paraphrased Word Count:** {output_word_count}")

        # Copy text functionality
        def copy_to_clipboard(text):
            pyperclip.copy(text)
            st.success("Text copied to clipboard!")

        # Button to copy paraphrased text to clipboard
        if st.button("Copy Text"):
            copy_to_clipboard(paraphrased_output)
    else:
        st.warning("Please enter some text to paraphrase.")
