import streamlit as st
from gptWrapper import get_wrapper_message

st.title("Titulo ejemplo")

st.divider()

st.subheader("subheader ejemplo")

st.image("https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRHCCcLRk8wC7I6wOeGJt4d6MPimyZ4h-YlTc2Vlq4KpsVYzAPWGdEwKI4K-jistvqINh8_GKYC_R_ZPkr_JBgbag")

st.text_input("aaaaaaa")

if st.button("enviar"):
    
    st.success("si")
else:
    st.warning("no") 