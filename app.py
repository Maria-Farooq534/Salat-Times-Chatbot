
import os
from dotenv import load_dotenv
import streamlit as st

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Salat Time - Islamabad 2025",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
genai.configure(api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.5)

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = []

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Display the chatbot's title on the page
st.title("Salat Time - Islamabad 2025")

# Display the chat history
for message in st.session_state.chat_session:
    with st.chat_message(translate_role_for_streamlit(message['role'])):
        st.markdown(message['content'])

def get_conversational_chain():
    prompt_template = '''
    Be nice and greet the user if user greets you or say something that can be responded with grace as per user's, Answer the question in a professional way from the given context.The given data is of salat(prayer/namaz) that Muslims offer 5 times a day, users will ask you the time of 5 slats. You will tell the time of that slat or all slats of that day if only date is mentioned as per 2025. use markdown language for responses. If the answer is not available, just reply that "This information is not in your file.", don't provide wrong answers.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer: 
    '''
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    # Create langchain conversational chain
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

# Get user input and generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    
    # Load the stored vectors
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    # Similarity search based on user input
    docs = new_db.similarity_search(user_question)

    # Call the conversational chain function
    chain = get_conversational_chain()
    
    # Respond to user query
    response = chain(
        {'input_documents': docs, 'question': user_question}, return_only_outputs=True
    )
    
    return response['output_text']

# Input field for user's message
user_prompt = st.chat_input("Ask a question from your files...")

if user_prompt:
    # Add user's message to chat and display it
    st.session_state.chat_session.append({'role': 'user', 'content': user_prompt})
    st.chat_message("user").markdown(user_prompt)

    # Send user's message to Google Generative AI and get the response
    response = user_input(user_prompt)
    st.session_state.chat_session.append({'role': 'model', 'content': response})

    # Display the response
    with st.chat_message("assistant"):
        st.markdown(response)
