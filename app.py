import streamlit as st
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
import nltk
import re
import os

# Load environment variables
load_dotenv()

# Ensure necessary NLTK data is downloaded
nltk.download('punkt_tab')  # Downloading the Punkt tokenizer model for tokenization
nltk.download('punkt')  # Ensuring basic tokenizer for sentences is available

# Streamlit App
def main():
    # Set page configuration for title and layout
    st.set_page_config(page_title="Sentiment Analysis with AI", layout="centered")  # Defining the title and layout of the app

    # Add custom CSS for styling
    st.markdown("""
    <style>
        .title {
            font-size: 35px;
            font-weight: bold;
            color: #D3D3D3;  /* Light gray color for the main title */
        }
        .subheader {
            font-size: 20px;
            color: #666;  /* Medium gray for subheaders */
        }
        
        .stButton button {
            background-color: #A9A9A9;  /* Dark gray button background */
            color: black;  /* Black text on the button */
            padding: 10px 20px;  /* Adding padding for a better click area */
            border-radius: 5px;  /* Rounded button edges */
        }
        .reportview-container {
            background-color: #f0f0f5;  /* Light off-white background for the main app area */
        }
        .sidebar .sidebar-content {
            background-color: #f7f7f9;  /* Slightly lighter background for the sidebar */
        }
    </style>
    """, unsafe_allow_html=True)  # Adding custom styling for a cleaner UI

    # Page title
    st.markdown("<h1 class='title'> 🧠 Sentiment Analysis with AI 🤖</h1>", unsafe_allow_html=True)  # Setting the main title with emojis

    # Sidebar instructions
    st.sidebar.header("How to use this tool")  # Sidebar header
    st.sidebar.markdown("""
    1. Enter your text into the text box.
    2. Click the 'Analyze Sentiment' button.
    3. View the sentiment analysis result and explanation.
    """)  # Simple instructions for users

    # User input area in the sidebar
    user_input = st.sidebar.text_area("Enter your text here:", value="", height=150)  # Text box for users to enter input

    # Analyze sentiment button
    if st.sidebar.button("Analyze Sentiment"):  # Button to trigger the sentiment analysis
        if not user_input.strip():  # Check if input is empty or just spaces
            st.error("Please enter some text to analyze.")  # Display error if no input
        else:
            # Tokenize input text
            tokens = word_tokenize(user_input)  # Tokenizing the user input for further processing

            # Initialize AI
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Fetch API key securely from environment variables
            llm = LLM(model="gpt-4o-mini", temperature=0)  # Initializing the LLM with specified settings

            # Define Agent and Task
            agent = Agent(
                role="Analyst",
                goal="Sentiment Analysis",
                backstory="You are an analyst for a large company. Your task is to analyze and classify the sentiment of user-provided text into one of ['positive', 'neutral', 'negative']. And explain why you think it is!",
                llm=llm,
                verbose=True,
            )  # Configuring an AI agent for sentiment analysis

            task = Task(
                expected_output="Sentiment of the input text is, ",
                description="Analyze and classify whether the sentiment of the {input} text is one of ['positive', 'neutral', 'negative']. And explain why you think it is!",
                verbose=True,
                agent=agent,
                llm=llm,
            )  # Creating a task with specific requirements for sentiment classification

            # Initialize Crew and perform task
            my_crew = Crew(agents=[agent], tasks=[task])  # Adding the agent and task to the Crew for execution

            with st.spinner('Analyzing the sentiment...'):  # Display a spinner while the analysis is being performed
                try:
                    result = my_crew.kickoff(inputs={"input": tokens})  # Execute the task with the input tokens
                    st.success('Analysis complete!')  # Notify the user when the analysis is done

                    task_output = result.tasks_output[0]  # Assuming tasks_output is a list
                    final_answer = task_output.raw.lower() if task_output.raw else ''  # Extract and normalize the result

                    # Display the result with color-coded sentiment
                    if re.search(r"('positive'\.|positive\.)", final_answer, re.IGNORECASE):  # Check for positive sentiment
                        st.markdown(f"<p style='color:green;'>{result}</p>", unsafe_allow_html=True)  # Display result in green
                    elif  re.search(r"('negative'\.|negative\.)", final_answer, re.IGNORECASE):  # Check for negative sentiment
                        st.markdown(f"<p style='color:red;'>{result}</p>", unsafe_allow_html=True)  # Display result in red
                    else:
                        st.markdown(f"<p style='color:gray;'>{result}</p>", unsafe_allow_html=True)  # Display result in gray for neutral

                except Exception as e:
                    st.error(f"Error during sentiment analysis: {e}")  # Display any error that occurs during analysis

    # Add progress bar for loading indication
    st.progress(0)  # Placeholder progress bar (not functional, but can be enhanced later)

if __name__ == "__main__":
    main()  # Execute the app
