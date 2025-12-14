import streamlit as st
st.set_page_config(page_title='Hello', layout='wide')
st.title('hello streamlit')
st.write('Python:', __import__('platform').python_version())
