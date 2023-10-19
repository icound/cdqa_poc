# %%
# Anaconda promt
# git clone https://github.com/cdqa-suite/cdQA.git
# cd cdQA
# pip install -e .
import os
import torch
import cdqa
import pandas as pd
from ast import literal_eval
from cdqa.utils.converters import pdf_converter
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model
import streamlit as st

def main():

    st.title("PDF Summarizer through Q&A")
    
    # Select Data Directory
    data_directory_path = st.sidebar.text_input("Enter Data Directory Path")
    def pdfconvert(path):
        try:
            return pdf_converter(directory_path=data_directory_path)
        except Exception as inst:
            print(type(inst))

    if data_directory_path:
        data_directory_path = os.path.abspath(data_directory_path)

        # Validate Data Directory Path
        if not os.path.isdir(data_directory_path):
            st.warning("Invalid data directory path.")
            return

        # Display Files in Data Directory
        st.subheader("Files in Data Directory:")
        files = os.listdir(data_directory_path)
        if files:
            for file in files:
                st.write(file)
        else:
            st.write("No files found in the selected directory.")


    # User Question Section
    st.subheader("Ask Questions")
    num_questions = st.number_input("Number of questions", min_value=1, max_value=10, step=1, value=1)
    user_questions = []
    for i in range(num_questions):
        question = st.text_input(f"Question {i+1}")
        user_questions.append(question)

    if st.button("Ask"):
        # Process the user's questions here
        # You can use any natural language processing library like NLTK or spaCy

        # Dummy responses for demonstration
        df_test=pdfconvert('data_directory_path')
        df_test=df_test[df_test['paragraphs'].notna()]
        # list of data frames to itterate over
        df_list=[v for k, v in df_test.groupby('title')]
        cdqa_pipeline2 = QAPipeline(reader='./models/distilbert_qa.joblib', max_df=1.0)
        def prediction(dflist, questions):
            cdqa_pipeline2.fit_retriever(df=dflist)
            prediction=[]
            # for q in ["what is the agreement #", 'what is the customer name',"how much is the contract value and fee", "what kind of vmware software is customer purchasing?", "what is the schedule"]:
            for q in questions:
                prediction.append(cdqa_pipeline2.predict(q))
                # predict=cdqa_pipeline2.predict(qq)
            return prediction
        out=[prediction(i, user_questions) for i in df_list]
        out_df=pd.DataFrame(out)
        out_df.columns=user_questions
        out_df.to_csv('../out/out_df_test.csv', index=False)
        # st.write(df_test)
        st.write(pd.read_csv('../out/out_df_test.csv', index_col=False))
         # Save as CSV
        




if __name__ == "__main__":
    main()