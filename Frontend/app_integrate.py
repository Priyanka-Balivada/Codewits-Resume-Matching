import os
import nltk
from nltk.tokenize import word_tokenize
import PyPDF2
import pandas as pd
import re  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import tkinter as tk
from tkinter import filedialog

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:  # Open the file in binary mode ('rb')
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def preprocess_text(text):
    return word_tokenize(text.lower())

# Function to extract CGPA from text using regular expressions
def extract_cgpa(text):
    cgpa_pattern = r'\b\d\.\d{1,2}\b'  # Pattern for CGPA (e.g., 3.75, 4.0, etc.)
    cgpa_matches = re.findall(cgpa_pattern, text)
    return cgpa_matches

# Title of the application
st.title('Resume Matching Tool')

# Function to create vertical space
def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')

v_spacer(height=3, sb=False)
st.subheader("Choose Files")
uploaded_file = st.file_uploader("Choose a Job Description")

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

# Check if both the job description and resume folder are selected
if uploaded_file and selected_folder:
    # Process job description
    try:
        job_description_text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            job_description_text = uploaded_file.read().decode("latin-1")
        except UnicodeDecodeError:
            job_description_text = uploaded_file.read().decode("utf-16")

    # Process resumes
    resumes_files = [os.path.join(selected_folder, file) for file in os.listdir(selected_folder) if file.endswith(".pdf")]

    # Process job description text
    preprocessed_job_text = preprocess_text(job_description_text)

    # Process resumes
    all_resumes_text = [extract_text_from_pdf(resume_path) for resume_path in resumes_files]
    preprocessed_resumes_text = [preprocess_text(text) for text in all_resumes_text]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)

    # Fit the vectorizer on all resume texts and the selected job text
    vectorizer.fit(all_resumes_text + [job_description_text])

    # Transform the job text and resume texts into TF-IDF vectors
    job_tfidf = vectorizer.transform([job_description_text])
    resume_tfidf = vectorizer.transform(all_resumes_text)

    # Compute cosine similarity between job text and resume texts
    similarity_scores = cosine_similarity(job_tfidf, resume_tfidf)

    # Initialize a dictionary to store results
    results_data = {'Resume': [], 'Similarity Score': [], 'CGPA': [], 'Total Score': []}

    # Extract CGPA from each resume and store results in the dictionary
    for i, resume_file in enumerate(resumes_files):
        cgpa_matches = extract_cgpa(all_resumes_text[i])
        cgpa = ', '.join(cgpa_matches) if cgpa_matches else 'Not found'
        results_data['Resume'].append(os.path.basename(resume_file))
        smScore = similarity_scores[0][i] * 100
        results_data['Similarity Score'].append(smScore)
        results_data['CGPA'].append(cgpa)
        if cgpa == "Not found":
            results_data['Total Score'].append(smScore)
        else:
            sc = float(cgpa) + smScore
            results_data['Total Score'].append(sc)

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results_data)

    # Sort the DataFrame by similarity score in descending order
    results_df = results_df.sort_values(by='Similarity Score', ascending=False)

    # Display the results table
    st.subheader("Matching Results")
    st.table(results_df)

    # Download results
    if st.button('Download Results'):
        results_df.to_excel('ResumeMatched.xlsx', index=False)

    # Visualization
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