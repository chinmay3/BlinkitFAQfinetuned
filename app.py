import streamlit as st
from huggingface_hub import InferenceClient
import os 

# Initialize HF client
client = InferenceClient(token=st.secrets["HF_TOKEN"])
MODEL_NAME = "Chinmay3/llama3-fine-tuned"  

def format_prompt(instruction, input_text):
    return f"""Below is an instruction that describes a task. Write a response.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

def generate_response(instruction, input_text=""):
    prompt = format_prompt(instruction, input_text)
    response = client.text_generation(
        prompt,
        max_new_tokens=128,
        temperature=0.7,
    )
    return response.split("### Response:")[-1].strip()

st.markdown(
    """
    <style>
    body {
        background-color: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSvlSzxnf4I6ot4GzK7yoCH9YS3WCNNJ5o7Tg&s",
    use_column_width=True,
)

# Streamlit UI
st.title("Llama 3 Blinkit FAQ Bot")
instruction = st.text_input("What do you need help with?", "How to return a product...")
input_text = st.text_input("Additional context (optional)")

if st.button("Get Answer"):
    with st.spinner("Generating response..."):
        answer = generate_response(instruction, input_text)
        st.write(answer)