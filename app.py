import streamlit as st
import torch
import nltk
import pyperclip
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# Set Streamlit page configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# ✅ Download all NLTK modules (avoids missing dependency issues)
nltk.download('all')

# ✅ Load the paraphrasing model
@st.cache_resource
def load_model():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device

model, tokenizer, device = load_model()

# ✅ Function to paraphrase text (Improved for short sentences)
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
                do_sample=True,  # ✅ Enables more diverse outputs
                temperature=1.2,  # ✅ Adds more variation in the response
                top_k=50,  # ✅ Controls randomness (higher = more variety)
                top_p=0.95  # ✅ Nucleus sampling for better diversity
            )

        paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)

    return " ".join(paraphrased_sentences)

# ✅ Streamlit UI
st.title("AI Paraphrasing Tool")
st.write("Enter a paragraph below to generate a paraphrased version.")

# 🔹 Sidebar for instructions
st.sidebar.header("Instructions")
st.sidebar.write(
    "1. Enter your text in the box.\n"
    "2. Click 'Paraphrase' to generate a paraphrased version.\n"
    "3. Click 'Clear Text' to reset the input field.\n"
    "4. Click 'Copy Text' to copy the paraphrased text.\n"
    "5. Check the word count before and after paraphrasing."
)

# 🔹 User Input
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Enter Text", height=150)
    word_count = len(user_input.split()) if user_input.strip() else 0
    st.write(f"**Word Count:** {word_count}")

with col2:
    if st.button("Clear Text"):
        st.experimental_rerun()  # Clears the input field

# 🔹 Paraphrase & Display Results
if st.button("Paraphrase"):
    if user_input.strip():
        paraphrased_output = paraphrase_text(user_input)
        output_word_count = len(paraphrased_output.split())

        st.subheader("Paraphrased Text:")
        st.text_area("Output", value=paraphrased_output, height=150, key="output")
        st.write(f"**Paraphrased Word Count:** {output_word_count}")

        # ✅ Copy to Clipboard Function
        def copy_to_clipboard(text):
            pyperclip.copy(text)
            st.success("Text copied to clipboard!")

        # 🔹 Copy Button
        if st.button("Copy Text"):
            copy_to_clipboard(paraphrased_output)
    else:
        st.warning("Please enter some text to paraphrase.")
