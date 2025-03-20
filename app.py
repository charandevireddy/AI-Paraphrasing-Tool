import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
import os
import re

# ✅ Set page config as the first Streamlit command
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# ✅ Set a custom NLTK data directory (if needed)
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# ✅ Ensure NLTK punkt and its resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # Force re-download the punkt package
    nltk.download('punkt', download_dir=nltk_data_dir, force=True)

# ✅ Load the paraphrasing model
@st.cache_resource
def load_model():
    model_name = "t5-small"  # Replace with a different model if needed
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model, tokenizer, device

model, tokenizer, device = load_model()

# ✅ Function to preprocess input text
def preprocess_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Keep only alphanumeric, spaces, and basic punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# ✅ Function to paraphrase text
def paraphrase_text(text):
    try:
        sentences = nltk.tokenize.sent_tokenize(text)  # Sentence segmentation
        paraphrased_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue  # Skip empty sentences

            input_text = f"paraphrase: {sentence}"  # Updated input format
            encoding = tokenizer.encode_plus(
                input_text, return_tensors="pt", padding="max_length", max_length=256, truncation=True
            ).to(device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    max_length=256,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,  # Adjust for diversity
                    top_p=0.95,  # Adjust for diversity
                    temperature=0.7,  # Lower for less randomness
                    repetition_penalty=2.5  # Penalize repetition
                )

            paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
            paraphrased_sentences.append(paraphrased_sentence)

            # Debug: Print input and output
            st.write(f"**Input Sentence:** {sentence}")
            st.write(f"**Paraphrased Sentence:** {paraphrased_sentence}")

        return " ".join(paraphrased_sentences)

    except Exception as e:
        st.error(f"❌ Error in paraphrasing: {str(e)}")
        return None

# ✅ Streamlit UI Setup
st.title("🚀 AI Paraphrasing Tool")
st.write("Enter a paragraph below to generate a **paraphrased version** using AI.")

# 📌 Sidebar Instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write(
    "1️⃣ Enter your text in the box.\n"
    "2️⃣ Click **'Paraphrase'** to generate a paraphrased version.\n"
    "3️⃣ Click **'Clear Text'** to reset the input field.\n"
    "4️⃣ Click **'Copy Text'** to copy the paraphrased text.\n"
    "5️⃣ Check the **word count** before and after paraphrasing."
)
st.sidebar.header("🔗 About This App")
st.sidebar.write("This AI-powered paraphrasing tool uses **T5-small** model for high-quality text rewording.")
st.sidebar.markdown("[GitHub Repo](https://github.com/charandevireddy/AI-Paraphrasing-Tool.git)")

# 📌 Session State for Input and Output
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if "paraphrased_output" not in st.session_state:
    st.session_state["paraphrased_output"] = ""

# 📌 User Input
user_input = st.text_area(
    "✍️ Enter Text",
    height=200,
    placeholder="Type or paste your text here...",
    value=st.session_state["user_input"],
    key="input_text_area"
)

# Preprocess input text
cleaned_input = preprocess_text(user_input)

if cleaned_input:
    word_count = len(cleaned_input.split())
    st.write(f"**📝 Word Count:** {word_count}")

# 📌 Buttons for Actions
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🔄 Paraphrase"):
        if cleaned_input.strip():
            with st.spinner("⏳ Generating paraphrase..."):
                paraphrased_text = paraphrase_text(cleaned_input)
                st.session_state["paraphrased_output"] = paraphrased_text
        else:
            st.warning("⚠️ Please enter some text to paraphrase.")

with col2:
    if st.button("🧹 Clear Text"):
        st.session_state["user_input"] = ""
        st.session_state["paraphrased_output"] = ""
        st.experimental_rerun()

with col3:
    if st.button("📋 Copy Text"):
        if st.session_state["paraphrased_output"]:
            st.markdown(
                f"""
                <script>
                navigator.clipboard.writeText(`{st.session_state["paraphrased_output"]}`);
                </script>
                """,
                unsafe_allow_html=True
            )
            st.success("✅ Copied to clipboard!")
        else:
            st.warning("⚠️ No paraphrased text to copy.")

# 📌 Display Paraphrased Output
if st.session_state["paraphrased_output"]:
    st.subheader("✨ Paraphrased Output")
    st.text_area(
        "Paraphrased Text",
        value=st.session_state["paraphrased_output"],
        height=200,
        key="output_text_area"
    )
    output_word_count = len(st.session_state["paraphrased_output"].split())
    st.write(f"**🔢 Paraphrased Word Count:** {output_word_count}")
