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

with st.form(key="my_form", clear_on_submit=False):
    # User prompt input
    prompt = st.text_input("Enter your prompt:", value="")
    
    # Number of tokens to generate
    token_length = st.number_input("Number of tokens to generate:", min_value=3, max_value=150, value=30)
    
    # Submit button
    submit_button = st.form_submit_button(label="Generate Responses")

# Check if the form is submitted either by button or Enter key
if submit_button and prompt:
    # High temperature (creative response)
    st.subheader("Creative Response :")
    creative_response = generate_response(prompt, token_length, temperature=0.9)
    st.write(creative_response)

    # Low temperature (predictable response)
    st.subheader("Predictable Response :")
    predictable_response = generate_response(prompt, token_length, temperature=0.1)
    st.write(predictable_response)

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
