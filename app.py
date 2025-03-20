import streamlit as st
import torch
import nltk
from transformers import BartForConditionalGeneration, BartTokenizer
from nltk.tokenize import sent_tokenize

# Set Streamlit page configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# Download NLTK tokenizer
nltk.download("punkt")

# Load the paraphrasing model
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device

model, tokenizer, device = load_model()

# Function to paraphrase text
def paraphrase_text(text):
    sentences = sent_tokenize(text)
    paraphrased_sentences = []

    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence, return_tensors="pt", padding="longest", max_length=256, truncation=True
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                max_length=256,
                min_length=50,  # Ensures meaningful output
                num_return_sequences=1,
                do_sample=True,
                top_k=50,  # Increase for better diversity
                top_p=0.95,
                temperature=0.8,
            )

        paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_sentence)

    return " ".join(paraphrased_sentences)

# Streamlit UI
st.title("📝 AI Paraphrasing Tool")
st.write("Enter a paragraph below to generate a paraphrased version.")

# Sidebar Instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write(
    "1. Enter text in the box.\n"
    "2. Click **'Paraphrase'** to generate a new version.\n"
    "3. Click **'Clear Text'** to reset the input.\n"
    "4. Click **'Copy Text'** to copy the paraphrased output."
)

# User Input
user_input = st.text_area("Enter Text:", height=150)
word_count = len(user_input.split()) if user_input else 0
st.write(f"**Word Count:** {word_count}")

# Buttons Layout
col1, col2, col3 = st.columns([1, 1, 1])

# Clear Text Button
with col1:
    if st.button("Clear Text"):
        st.experimental_rerun()

# Paraphrase Button
paraphrased_output = ""
with col2:
    if st.button("Paraphrase"):
        if user_input.strip():
            paraphrased_output = paraphrase_text(user_input)
            output_word_count = len(paraphrased_output.split())

            st.subheader("🔄 Paraphrased Text:")
            st.text_area("Output", value=paraphrased_output, height=150, key="output")
            st.write(f"**Paraphrased Word Count:** {output_word_count}")
        else:
            st.warning("⚠️ Please enter some text to paraphrase.")

# Copy Text Button (JavaScript-Based Copy)
if paraphrased_output:
    with col3:
        copy_script = f"""
        <script>
        function copyText() {{
            var text = document.getElementById("output").value;
            navigator.clipboard.writeText(text);
            alert("Text copied to clipboard!");
        }}
        </script>
        <button onclick="copyText()">Copy Text</button>
        """
        st.markdown(copy_script, unsafe_allow_html=True)
