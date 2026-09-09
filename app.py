import streamlit as st

def validate():
    if data == None:
        st.error("Upload a file")
    else:
        st.write("Running Model")


st.title("NeuroSense",text_alignment="center")
st.header("Early Cognitive Decline Detection using Spiking Neural Networks",text_alignment="center")
data = st.file_uploader("Upload MRI",type="image/*",accept_multiple_files=False)
button = st.button("Start Processing",on_click=validate())