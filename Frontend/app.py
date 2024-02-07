import streamlit as st
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import nltk
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import PyPDF2
import pandas as pd
import re  # Added import for regular expressions
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
# initialize list of lists
data = [['alter.jah@gmail.com','10089434.pdf', 0.616698,'java,python'], ['john.joe@gmail.com','10247517.pdf', 0.518026,'engineering']]        
 
# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Email','Resume','Matching Score','Keywords'])

nltk.download('punkt')

st.title('Resume Matching Tool')

def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')

v_spacer(height=3, sb=False)
st.subheader("Choose Files")
# st.write("Choose a Job Description")
uploaded_file = st.file_uploader("Choose a Job Description")

# st.write(uploaded_file)

# Create a button to browse for a folder
v_spacer(height=3, sb=False)
clicked = st.button('Browse Resume Folder')

# Initialize the folder variable
selected_folder = ""

# Check if the button is clicked
if clicked:
    # Create a Tkinter root window (hidden)
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    # Ask the user to select a folder
    selected_folder = filedialog.askdirectory(master=root)

    # Display the selected folder
    st.write(f'Selected Folder: {selected_folder}')

    # If a folder is selected, list all PDF files in the folder and display the table
    if selected_folder:
        pdf_reports = [file for file in os.listdir(selected_folder) if file.endswith('.pdf')]
        output = pd.DataFrame({"File Name": pdf_reports})
        st.table(output)

v_spacer(height=3, sb=False)

calculate=st.button('Match Resumes')
if calculate:
    st.subheader("Matching Results")
    st.table(df)

v_spacer(height=3, sb=False)

results = st.button('Download Results')

if results:
    df.to_excel('ResumeMatched.xlsx', index=True)

v_spacer(height=3, sb=False)
if st.button("Show Visualization"):
    # Load your images
    image1 = "image1.jpg"
    image2 = "image2.jpg"
    image3= "image3.jpg"
    # Create two columns
    col1, col2 = st.columns(2)

    # Display the first image in the first column
    col1.image(image1, caption="Top Matching Skills", use_column_width=True)

    # Display the second image in the second column
    col2.image(image2, caption="Skills Distribution", use_column_width=True)

    st.image(image3,"Skills")