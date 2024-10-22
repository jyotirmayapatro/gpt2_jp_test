import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# creating a function to generate response from the GPT-2 model
def generate_response(prompt, max_length, temperature):
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    
    # text generation    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=2
    )
 
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

### App details ### 
# App name
st.title("JP's GPT-2 Text Generation App")

# user prompt input
prompt = st.text_input("Enter your prompt:", value="Once upon a time")

# token details
token_length = st.number_input("Number of tokens to generate:", min_value=3, max_value=150, value=30)

# button
if st.button("Generate Responses"):
    # # # high temperature/creative
    st.subheader("Creative Response :")
    creative_response = generate_response(prompt, token_length, temperature=1.5)
    st.write(creative_response)

    # # # low temperature/predicted
    st.subheader("Predictable Response :")
    predictable_response = generate_response(prompt, token_length, temperature=0.2)
    st.write(predictable_response)
