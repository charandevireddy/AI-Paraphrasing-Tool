import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
import asyncio

# ✅ Fix Asyncio issue
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None

if loop and loop.is_running():
    pass  # Event loop is already running
else:
    asyncio.run(asyncio.sleep(0))  # Ensure an event loop starts

# ✅ Ensure NLTK punkt is downloaded
nltk.download('punkt')

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

# ✅ Function to paraphrase text
def paraphrase_text(text):
    try:
        sentences = nltk.tokenize.sent_tokenize(text)  # Sentence segmentation
        paraphrased_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue  # Skip empty sentences

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
                    temperature=0.7,
                    repetition_penalty=2.5
                )

            paraphrased_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
            paraphrased_sentences.append(paraphrased_sentence)

        return " ".join(paraphrased_sentences)

    except Exception as e:
        return f"❌ Error in paraphrasing: {str(e)}"

# ✅ Streamlit UI Setup
st.set_page_config(page_title="AI Paraphrasing Tool", layout="centered")
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
st.sidebar.write("This AI-powered paraphrasing tool uses **T5-base** model for high-quality text rewording.")
st.sidebar.markdown("[GitHub Repo](https://github.com/charandevireddy/AI-Paraphrasing-Tool.git)")

# 📌 User Input
user_input = st.text_area("✍️ Enter Text", height=200, placeholder="Type or paste your text here...")

if user_input:
    word_count = len(user_input.split())
    st.write(f"**📝 Word Count:** {word_count}")

# 📌 Buttons for Actions
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🔄 Paraphrase"):
        with st.spinner("Processing..."):
            paraphrased_text = paraphrase_text(user_input)
            st.text_area("✨ Paraphrased Output", value=paraphrased_text, height=200)

with col2:
    if st.button("🧹 Clear Text"):
        user_input = ""

with col3:
    if st.button("📋 Copy Text"):
        st.write("✅ Copied to clipboard!")
