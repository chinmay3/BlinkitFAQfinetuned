import streamlit as st
from huggingface_hub import InferenceClient

# Initialize HF client
client = InferenceClient(token=st.secrets["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"])
MODEL_NAME = "Chinmay3/llama3-fine-tuned"  # Replace with your model ID

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

# Streamlit UI
st.title("🦙 Llama 3 Product Assistant")
instruction = st.text_input("What do you need help with?", "How to return a product...")
input_text = st.text_input("Additional context (optional)")

if st.button("Get Answer"):
    with st.spinner("Generating response..."):
        answer = generate_response(instruction, input_text)
        st.write(answer)