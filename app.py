import streamlit as st
import torch
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Fix missing NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')  # Ensure 'punkt_tab' is available

# Set Streamlit page configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

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
    try:
        # ✅ Alternative to NLTK's sent_tokenize (if NLTK fails)
        sentences = nltk.tokenize.sent_tokenize(text) if 'punkt_tab' in nltk.data.find("tokenizers/") else text.split(". ")

        paraphrased_sentences = []

        for sentence in sentences:
            input_text = f"paraphrase this: {sentence}"
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
                    top_k=10,
                    top_p=0.9,
                    temperature=0.7
                )

            paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
            paraphrased_sentences.append(paraphrased_sentence)

        return " ".join(paraphrased_sentences)

    except Exception as e:
        return f"❌ Error in paraphrasing: {str(e)}"

# Streamlit app layout
st.title("🚀 AI Paraphrasing Tool")
st.write("Enter a paragraph below to generate a **paraphrased version** using AI.")

# Sidebar with instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write(
    "1️⃣ Enter your text in the box.\n"
    "2️⃣ Click **'Paraphrase'** to generate a paraphrased version.\n"
    "3️⃣ Click **'Clear Text'** to reset the input field.\n"
    "4️⃣ Click **'Copy Text'** to copy the paraphrased text.\n"
    "5️⃣ Check the **word count** before and after paraphrasing."
)

st.sidebar.header("🔗 About This App")
st.sidebar.write("This AI-powered paraphrasing tool uses **T5-base** model for high-quality text rewording.")
st.sidebar.markdown("[GitHub Repo](https://github.com/charandevireddy/AI-Paraphrasing-Tool.git)")

# User input for text
user_input = st.text_area("✍️ Enter Text", height=200, placeholder="Type or paste your text here...")

if user_input:
    word_count = len(user_input.split())
    st.write(f"**📝 Word Count:** {word_count}")

# Buttons for actions
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🔄 Clear Text"):
        st.experimental_rerun()  # Clears the input field

with col2:
    if st.button("✨ Paraphrase"):
        if user_input.strip():
            with st.spinner("⏳ Generating paraphrase..."):
                paraphrased_output = paraphrase_text(user_input)
                output_word_count = len(paraphrased_output.split())

            st.subheader("📄 Paraphrased Text:")
            st.markdown(f"```{paraphrased_output}```")  # Nicely formatted output
            st.write(f"**🔢 Paraphrased Word Count:** {output_word_count}")

            # JavaScript-based Copy to Clipboard functionality
            copy_code = f"""
            <script>
            function copyText() {{
                navigator.clipboard.writeText("{paraphrased_output}");
                alert("Text copied to clipboard!");
            }}
            </script>
            <button onclick="copyText()">📋 Copy Text</button>
            """
            st.markdown(copy_code, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to paraphrase.")
