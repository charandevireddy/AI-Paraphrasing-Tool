import streamlit as st
import torch
import nltk
import asyncio
import pyperclip
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# ✅ Ensure NLTK Tokenizer (punkt) is Downloaded
nltk.download('punkt')

# ✅ Fix for Asyncio Event Loop Conflict
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Load Paraphrasing Model (with GPU/CPU Support)
@st.cache_resource
def load_model():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device

model, tokenizer, device = load_model()

# ✅ Function to Paraphrase Text
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

# ✅ Streamlit App Layout
st.title("AI Paraphrasing Tool")
st.write("Enter a paragraph below to generate a paraphrased version.")

# 🔹 Sidebar Instructions
st.sidebar.header("Instructions")
st.sidebar.write(
    "1. Enter your text in the box.\n"
    "2. Click 'Paraphrase' to generate a paraphrased version.\n"
    "3. Click 'Clear Text' to reset the input field.\n"
    "4. Click 'Copy Text' to copy the paraphrased text.\n"
    "5. Check the word count before and after paraphrasing."
)

# 🔹 User Input Section
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area("Enter Text", value=st.session_state.user_input, height=150)
word_count = len(user_input.split()) if user_input.strip() else 0
st.write(f"**Word Count:** {word_count}")

# 🔹 Clear Text Button
if st.button("Clear Text"):
    st.session_state.user_input = ""  # Proper way to reset input
    st.rerun()  # Rerun the app to clear input

# 🔹 Paraphrase & Display Results
if st.button("Paraphrase"):
    if user_input.strip():
        paraphrased_output = paraphrase_text(user_input)
        output_word_count = len(paraphrased_output.split())

        st.subheader("Paraphrased Text:")
        st.text_area("Output", value=paraphrased_output, height=150, key="output")
        st.write(f"**Paraphrased Word Count:** {output_word_count}")

        # ✅ Copy to Clipboard Function (with Exception Handling)
        def copy_to_clipboard(text):
            try:
                pyperclip.copy(text)
                st.success("Text copied to clipboard!")
            except pyperclip.PyperclipException:
                st.error("Copying to clipboard is not supported on this system.")

        # 🔹 Copy Button
        if st.button("Copy Text"):
            copy_to_clipboard(paraphrased_output)
    else:
        st.warning("Please enter some text to paraphrase.")
