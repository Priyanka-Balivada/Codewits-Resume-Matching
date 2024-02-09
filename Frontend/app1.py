import os
import nltk
from nltk.tokenize import word_tokenize
import PyPDF2
import pandas as pd
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def preprocess_text(text):
    return word_tokenize(text.lower())

def extract_cgpa(text):
    cgpa_pattern = r'\b\d\.\d{1,2}\b'  # Pattern for CGPA (e.g., 3.75, 4.0, etc.)
    cgpa_matches = re.findall(cgpa_pattern, text)
    return cgpa_matches

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'

def train_doc2vec_model(documents):
    model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def calculate_similarity(model, text1, text2):
    vector1 = model.infer_vector(preprocess_text(text1))
    vector2 = model.infer_vector(preprocess_text(text2))
    return cosine_similarity([vector1], [vector2])[0][0]

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

    # Preprocess the documents and create TaggedDocuments for resumes
    tagged_resumes = [TaggedDocument(words=preprocess_text(extract_text_from_pdf(resume_path)), tags=[str(i)]) for i, resume_path in enumerate(resumes_files)]
    
    # Train Doc2Vec model for resumes
    model_resumes = train_doc2vec_model(tagged_resumes)

    # Create a DataFrame to store the results
    results_data = {'Resume': [], 'Similarity Score': [], 'CGPA': [], 'Total Score': [], 'Email': [], 'Contact': []}

    # Compare the selected job description with all resumes
    for i, resume_file in enumerate(resumes_files):
        similarity_score = calculate_similarity(model_resumes, extract_text_from_pdf(resume_file), job_description_text)
        cgpa_matches = extract_cgpa(extract_text_from_pdf(resume_file))
        cgpa = ', '.join(cgpa_matches) if cgpa_matches else 'Not found'
        results_data['Resume'].append(os.path.basename(resume_file))
        results_data['Similarity Score'].append(similarity_score*100)
        results_data['CGPA'].append(cgpa)

        emails = re.findall(email_pattern, extract_text_from_pdf(resume_file))
        contacts = re.findall(phone_pattern, extract_text_from_pdf(resume_file))
        if cgpa == "Not found":
            results_data['Total Score'].append(similarity_score*100)
        else:
            total_score = float(cgpa) + similarity_score*100
            results_data['Total Score'].append(total_score)

        results_data['Email'].append(emails)
        results_data['Contact'].append(contacts)

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results_data)

    # Sort the DataFrame by similarity score in descending order
    results_df = results_df.sort_values(by='Similarity Score', ascending=False)

    # Display the results table
    st.table(results_df)

    # Download results
    if st.button('Download Results'):
        results_df.to_excel('ResumeMatched.xlsx', index=False)

    v_spacer(height=3, sb=False)

    # if st.button("Show Visualization"):
    #     # Load your images
    #     image1 = "image1.jpg"
    #     image2 = "image2.jpg"
    #     image3 = "image3.jpg"
    #     # Create two columns
    #     col1, col2 = st.columns(2)

    #     # Display the first image in the first column
    #     col1.image(image1, caption="Top Matching Skills", use_column_width=True)

    #     # Display the second image in the second column
    #     col2.image(image2, caption="Skills Distribution", use_column_width=True)

    #     st.image(image3, "Skills")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results_df['Resume'], results_df['Similarity Score'], color='blue')
    ax.set_xlabel('Resume')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Similarity Scores between Job Description and Resumes')
    ax.set_xticklabels(results_df['Resume'], rotation=45, ha='right')
    plt.tight_layout()

    # if st.button("Show Visualization"):
            # Load your images
            # image1 = "image1.jpg"
            # image2 = "image2.jpg"
            # image3 = "image3.jpg"
            # # Create two columns
            # col1, col2 = st.columns(2)

            # # Display the first image in the first column
            # col1.image(image1, caption="Top Matching Skills", use_column_width=True)

            # # Display the second image in the second column
            # col2.image(image2, caption="Skills Distribution", use_column_width=True)

            # st.image(image3, "Skills") 

            

            # Display the Matplotlib plot in Streamlit
    st.pyplot(fig)