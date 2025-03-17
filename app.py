import streamlit as st
import torch
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# ✅ Set Streamlit page configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# ✅ Download NLTK tokenizer (punkt) for sentence segmentation
nltk.download('punkt')

# ✅ Load the paraphrasing model (Cached for efficiency)
@st.cache_resource
def load_model():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cpu")  # Streamlit Cloud does NOT support GPU
    model.to(device)

    return model, tokenizer, device

# Load the model
model, tokenizer, device = load_model()

# ✅ Function to paraphrase text
def paraphrase_text(text):
    if not text.strip():
        return "Please enter text to paraphrase."

    sentences = sent_tokenize(text)  # Sentence segmentation
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

# ✅ Streamlit App UI
st.title("AI Paraphrasing Tool")
st.write("Enter a paragraph below and click **Paraphrase** to generate a reworded version.")

# ✅ Sidebar Instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write(
    "1️⃣ Enter text in the box.\n"
    "2️⃣ Click 'Paraphrase' to generate a paraphrased version.\n"
    "3️⃣ Click 'Clear Text' to reset the input field.\n"
    "4️⃣ Check word count before and after paraphrasing."
)

# ✅ User Input
user_input = st.text_area("Enter Text", value="", height=150, key="user_input")

# ✅ Word Count Display
word_count = len(user_input.split())
st.write(f"**Word Count:** {word_count}")

# ✅ Buttons for Actions
col1, col2 = st.columns([1, 1])

# 🔹 Clear Text Button (Resets Input Field)
with col1:
    if st.button("Clear Text"):
        st.session_state["user_input"] = ""

# 🔹 Paraphrase Button
with col2:
    if st.button("Paraphrase"):
        with st.spinner("Paraphrasing... ⏳"):
            paraphrased_output = paraphrase_text(user_input)
            output_word_count = len(paraphrased_output.split())

            st.subheader("Paraphrased Text:")
            st.text_area("Output", value=paraphrased_output, height=150, key="output")
            st.write(f"**Paraphrased Word Count:** {output_word_count}")

