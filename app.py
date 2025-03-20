import streamlit as st
import torch
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# Set Streamlit page configuration
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")

# Download NLTK tokenizer (punkt)
nltk.download('punkt')

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
        sentences = sent_tokenize(text)
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

# Streamlit app layout
st.title("🚀 AI Paraphrasing Tool")
st.write("Enter a paragraph below to generate a **paraphrased version** using AI.")

# User input for text (Using session state to manage clearing input)
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

user_input = st.text_area("✍️ Enter Text", height=200, placeholder="Type or paste your text here...", key="user_input")

if user_input:
    word_count = len(user_input.split())
    st.write(f"**📝 Word Count:** {word_count}")

# Buttons for actions
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🔄 Clear Text"):
        st.session_state["user_input"] = ""  # Resets input without rerunning the whole app

with col2:
    if st.button("✨ Paraphrase"):
        if user_input.strip():
            with st.spinner("⏳ Generating paraphrase..."):
                paraphrased_output = paraphrase_text(user_input)
                output_word_count = len(paraphrased_output.split())

            st.subheader("📄 Paraphrased Text:")
            st.code(paraphrased_output, language="text")  # Proper text box for output
            st.write(f"**🔢 Paraphrased Word Count:** {output_word_count}")

            # ✅ Streamlit native Copy button (works without JavaScript)
            st.button("📋 Copy to Clipboard", on_click=lambda: st.session_state.update({"copy_text": paraphrased_output}))

        else:
            st.warning("⚠️ Please enter some text to paraphrase.")
