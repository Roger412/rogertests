from openai import OpenAI
import streamlit as st

openai_key = st.secrets("OPENAI_KEY")

client = OpenAI(api_key=openai_key)

def get_wrapper_message(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming."
            }
        ]
    )

    print(completion.choices[0].message)

