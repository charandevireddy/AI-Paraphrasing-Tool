import streamlit as st
import torch
import nltk
import os
import ssl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# ✅ Fix SSL issue (for downloading in Streamlit Cloud)
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# ✅ Set up persistent NLTK data path
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")  # Store in project folder
nltk.data.path.append(NLTK_DATA_PATH)

# ✅ Ensure 'punkt' is downloaded
punkt_path = os.path.join(NLTK_DATA_PATH, "tokenizers/punkt")
if not os.path.exists(punkt_path):
    os.makedirs(NLTK_DATA_PATH, exist_ok=True)  # Ensure directory exists
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

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

# ✅ Function to Paraphrase Text
def paraphrase_text(text):
    if not text.strip():
        return "Please enter text to paraphrase."

    # ✅ Manually ensure `punkt` is loaded before calling `sent_tokenize`
    try:
        sent_tokenize("Test sentence.")
    except LookupError:
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)

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
st.title("🔄 AI Paraphrasing Tool")
st.write("Enter a paragraph below and click **Paraphrase** to generate a reworded version.")

# ✅ Sidebar Instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write(
    """
    **How to Use?**
    1️⃣ Enter your text in the box.  
    2️⃣ Click **"Paraphrase"** to generate a paraphrased version.  
    3️⃣ Click **"Clear Text"** to reset the input field.  
    4️⃣ Check the word count before and after paraphrasing.  
    """
)

# ✅ User Input
st.subheader("✍ Enter Text:")
user_input = st.text_area("", value="", height=150, key="user_input")

# ✅ Word Count Display
word_count = len(user_input.split())
st.markdown(f"**📝 Word Count:** `{word_count}`")

# ✅ Buttons for Actions (Aligned Properly)
col1, col2 = st.columns(2)

with col1:
    if st.button("❌ Clear Text"):
        st.session_state["user_input"] = ""

with col2:
    if st.button("🔄 Paraphrase"):
        with st.spinner("Paraphrasing... ⏳"):
            paraphrased_output = paraphrase_text(user_input)
            output_word_count = len(paraphrased_output.split())

            st.subheader("🔹 Paraphrased Text:")
            st.text_area("", value=paraphrased_output, height=150, key="output")
            st.markdown(f"**📊 Paraphrased Word Count:** `{output_word_count}`")
