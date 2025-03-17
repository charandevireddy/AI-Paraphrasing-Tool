import streamlit as st
import torch
import pyperclip
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import TreebankWordTokenizer  # Alternative to Punkt

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# ✅ Load Model (Cached for Efficiency)
@st.cache_resource
def load_model():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device

# Load Model
model, tokenizer, device = load_model()

# ✅ Initialize Tokenizer (Replacement for Punkt)
tokenizer_nltk = TreebankWordTokenizer()

# ✅ Function to Paraphrase Text
def paraphrase_text(text):
    sentences = text.split(". ")  # Simple sentence segmentation
    paraphrased_sentences = []

    for sentence in sentences:
        input_text = f"paraphrase: {sentence} </s>"
        encoding = tokenizer.encode_plus(
            input_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)

    return " ".join(paraphrased_sentences)

# ✅ Streamlit UI
st.title("🔄 AI Paraphrasing Tool")
st.write("Enter a paragraph below to generate a paraphrased version.")

# ✅ Sidebar Instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write(
    """
    1️⃣ Enter your text in the box.  
    2️⃣ Click **"Paraphrase"** to generate a paraphrased version.  
    3️⃣ Click **"Clear Text"** to reset the input field.  
    4️⃣ Click **"Copy Text"** to copy the paraphrased output.  
    5️⃣ Check the word count before and after paraphrasing.  
    """
)

# ✅ User Input Section
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("✍ Enter Text:", height=150)
    word_count = len(user_input.split())
    st.write(f"**📝 Word Count:** `{word_count}`")

with col2:
    if st.button("❌ Clear Text"):
        st.experimental_rerun()  # Clears the input field

# ✅ Paraphrase Button and Output
if st.button("🔄 Paraphrase"):
    if user_input.strip():
        paraphrased_output = paraphrase_text(user_input)
        output_word_count = len(paraphrased_output.split())

        st.subheader("🔹 Paraphrased Text:")
        st.text_area("Output", value=paraphrased_output, height=150, key="output")
        st.write(f"**📊 Paraphrased Word Count:** `{output_word_count}`")

        # ✅ Copy to Clipboard Function
        def copy_to_clipboard(text):
            pyperclip.copy(text)
            st.success("✅ Text copied to clipboard!")

        # ✅ Copy Text Button
        if st.button("📋 Copy Text"):
            copy_to_clipboard(paraphrased_output)
    else:
        st.warning("⚠ Please enter some text to paraphrase.")
