# import streamlit as st;
# import os;
# import tkinter as tk
# from tkinter import filedialog
# import pandas as pd

# st.title('Resume Matching Tool')

# uploaded_file = st.file_uploader("Choose a job description")


# # def file_selector(folder_path='.'):
# #     filenames = os.listdir(folder_path)
# #     selected_filename = st.selectbox('Select a file', filenames)
# #     return os.path.join(folder_path, selected_filename)

# # filename = file_selector()
# # st.write('You selected `%s`' % filename)

# root = tk.Tk()
# root.withdraw()
# root.wm_attributes('-topmost', 1)
# st.write('Please select a folder:')
# clicked = st.button('Browse Folder')
    
# if clicked:
#     dirname = str(filedialog.askdirectory(master=root))
#     pdf_reports = [file for file in os.listdir(dirname) if file.endswith('.pdf')]
#     output = pd.DataFrame({"File Name": pdf_reports})
#     st.table(output)

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

#create your figure and get the figure object returned
# fig = plt.figure() 
# plt.plot(df) 

# st.pyplot(fig)

# st.dataframe(df)

# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page_num in range(len(pdf_reader.pages)):
#             text += pdf_reader.pages[page_num].extract_text()
#     return text

# def preprocess_text(text):
#     return word_tokenize(text.lower())

# def train_doc2vec_model(documents):
#     model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
#     model.build_vocab(documents)
#     model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
#     return model

# def calculate_similarity(model, text1, text2):
#     vector1 = model.infer_vector(preprocess_text(text1))
#     vector2 = model.infer_vector(preprocess_text(text2))
#     return model.dv.cosine_similarities(vector1, [vector2])[0]

# # Folder paths for resumes and job descriptions
# resumes_folder = "resume"
# job_descriptions_folder = "job"

# # List all PDF files in the folders
# resumes_files = [os.path.join(resumes_folder, file) for file in os.listdir(resumes_folder) if file.endswith(".pdf")]
# job_descriptions_files = [os.path.join(job_descriptions_folder, file) for file in os.listdir(job_descriptions_folder) if file.endswith(".pdf")]

# # Load job descriptions from the folder
# print("Available job descriptions:")
# for i, job_desc_file in enumerate(job_descriptions_files):
#     print(f"{i + 1}. {os.path.basename(job_desc_file)}")

# selected_job_index = int(input("Enter the index of the job description you want to compare (1 to {}): ".format(len(job_descriptions_files))))
# selected_job_path = job_descriptions_files[selected_job_index - 1]
# selected_job_text = extract_text_from_pdf(selected_job_path)

# # Load all resumes from the folder
# all_resumes_text = [extract_text_from_pdf(resume_path) for resume_path in resumes_files]

# # Preprocess the documents and create TaggedDocuments for resumes
# tagged_resumes = [TaggedDocument(words=preprocess_text(text), tags=[str(i)]) for i, text in enumerate(all_resumes_text)]

# # Train Doc2Vec model for resumes
# model_resumes = train_doc2vec_model(tagged_resumes)

# # Create a DataFrame to store the results
# results_data = {'Resume': [], 'Similarity Score': []}

# # Compare the selected job description with all resumes
# for i, resume_text in enumerate(all_resumes_text):
#     similarity_score = calculate_similarity(model_resumes, resume_text, selected_job_text)
#     results_data['Resume'].append(os.path.basename(resumes_files[i]))
#     results_data['Similarity Score'].append(similarity_score)

# # Create a DataFrame
# results_df = pd.DataFrame(results_data)

# # Find the index of the highest similarity score
# highest_score_index = results_df['Similarity Score'].idxmax()

# # Get the filename and full path of the resume with the highest score
# highest_score_resume = resumes_files[highest_score_index]
# highest_score_resume_text = all_resumes_text[highest_score_index]

# # Extract name, email addresses, and contact numbers using regular expressions
# name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
# email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
# phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'

# names = re.findall(name_pattern, highest_score_resume_text)
# emails = re.findall(email_pattern, highest_score_resume_text)
# contacts = re.findall(phone_pattern, highest_score_resume_text)

# # Combine extracted information into a single string for TF-IDF analysis
# combined_text = ' '.join([highest_score_resume_text] + names + emails + contacts)

# # Use TF-IDF to get high-weighted keywords
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform([combined_text])

# # Get feature names (words) and their corresponding TF-IDF scores
# feature_names = tfidf_vectorizer.get_feature_names_out()
# tfidf_scores = tfidf_matrix.toarray()[0]

# # Create a DataFrame to store keywords and their TF-IDF scores
# keywords_df = pd.DataFrame({'Keyword': feature_names, 'TF-IDF Score': tfidf_scores})

# # Sort DataFrame by TF-IDF scores in descending order
# keywords_df = keywords_df.sort_values(by='TF-IDF Score', ascending=False)

# # Display the results
# print("\nResume with the Highest Similarity Score:")
# print("Filename:", os.path.basename(highest_score_resume))
# print("Similarity Score:", results_df.loc[highest_score_index, 'Similarity Score'])
# print("Extracted Name:", names)
# print("Extracted Email Addresses:", emails)
# print("Extracted Contact Numbers:", contacts)

# # Display high-weighted keywords
# print("\nHigh-Weighted Keywords:")
# print(keywords_df.head(10))  # Displaying the top 10 keywords