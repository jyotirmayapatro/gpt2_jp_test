import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# creating a function to generate response from the GPT-2 model
# Ref: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py
def generate_response(prompt, max_length, temperature):
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    
    # text generation    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True
    )
 
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

### App details ### 
# Ref: https://github.com/dataprofessor/streamlit-freecodecamp-course/blob/main/app.py
# App name
st.title("JP's GPT-2 App")

# user prompt input
prompt = st.text_input("Enter your prompt:", value="")

# token details
token_length = st.number_input("Number of tokens to generate:", min_value=3, max_value=150, value=30)


if st.button("Generate Responses") or (prompt and st.session_state.get("generate_response", False)):
    
    # Flag to detect Enter key press
    st.session_state["generate_response"] = False

    # Generate creative response (high temperature)
    st.subheader("Creative Response :")
    creative_response = generate_response(prompt, token_length, temperature=0.9)
    st.write(creative_response)

    # Generate predictable response (low temperature)
    st.subheader("Predictable Response :")
    predictable_response = generate_response(prompt, token_length, temperature=0.1)
    st.write(predictable_response)


# Automatically trigger generation when Enter key is pressed
if prompt:
    st.session_state["generate_response"] = True


# button
# Ref: https://docs.streamlit.io
# buttonif st.button("Generate Responses"):
    # # # high temperature/creative
# button    st.subheader("Creative Response :")
# button    creative_response = generate_response(prompt, token_length, temperature=0.9)
# button    st.write(creative_response)

    # # # low temperature/predicted
# button    st.subheader("Predictable Response :")
# button    predictable_response = generate_response(prompt, token_length, temperature=0.1)
# button    st.write(predictable_response)
