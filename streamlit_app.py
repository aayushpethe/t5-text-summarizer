import streamlit as st
import torch
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

st.title("T5 Text Summarizer")

# Load model
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
    tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    return model, tokenizer, device


model, tokenizer, device = load_model()


# Clean text
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.strip().lower()
    return text


# Input
dialogue = st.text_area(
    "Enter text to summarize:",
    height=200
)


# Summarize
if st.button("Summarize"):

    if dialogue.strip():

        dialogue = clean_data(dialogue)

        input_text = "summarize: " + dialogue

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        st.subheader("Summary")
        st.write(summary)

    else:
        st.warning("Please enter some text.")