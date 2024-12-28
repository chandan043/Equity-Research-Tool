# import streamlit as st
# from final import QAModel

# def main():
#     st.title("Equity Research Tool")
#     st.sidebar.title("News Article URLs")

#     # Initialize the model
#     model = QAModel()

#     # Collect URLs from user input
#     urls = []
#     for i in range(3):
#         url = st.sidebar.text_input(f"URL {i+1}")
#         if url:
#             urls.append(url)

#     question_input = st.text_input("Enter your question:", "What is the full form of NRI?")
    
#     if st.button("Get Answer") and urls:
#         # Replace the import_urls and import_question methods with user input
#         model.import_urls = lambda: urls
#         model.import_question = lambda: [question_input]

#         # Process and get the answer
#         score, answer = model.process()
        
#         if score is not None:
#             st.write(f"**Answer:** {answer}")
#             st.write(f"**Score:** {round(score * 100, 2)}%")
#         else:
#             st.write("No answer found.")
#     elif not urls:
#         st.write("Please enter at least one URL.")

# if __name__ == "__main__":
#     main()

import streamlit as st
from final import QAModel
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    # Page configuration
    st.set_page_config(page_title="Equity Research Tool", layout="wide")

    # Add a colorful sidebar for URLs and instructions
    st.sidebar.title("🌐 News Article Inputs")
    st.sidebar.markdown(
        "<div style='color: blue; font-size: 16px;'>Enter the URLs of news articles you want to analyze.</div>",
        unsafe_allow_html=True
    )

    # Initialize the model
    model = QAModel()

    # Collect URLs from user input
    st.sidebar.header("📑 Article URLs")
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}", placeholder=f"Enter URL {i+1}")
        if url:
            urls.append(url)

    # Main panel for question and answers
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Equity Research Question Answering Tool</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size: 18px; text-align: center;'>This tool uses AI to extract relevant answers from news articles.</p>",
        unsafe_allow_html=True
    )

    # Question input
    st.text_input(
        "Enter your question:", placeholder="Type your question here...", key="question_input",
        help="Ask any question related to the articles you provided."
    )

    # Display a button to trigger the process
    if st.button("🔍 Get Answer", help="Click to process your question and find answers."):
        question_input = st.session_state.get("question_input", "")
        if not urls:
            st.error("❌ Please provide at least one valid URL.")
        elif not question_input:
            st.error("❌ Please enter a question.")
        else:
            with st.spinner("⏳ Processing... Please wait."):
                # Replace the import_urls and import_question methods with user input
                model.import_urls = lambda: urls
                model.import_question = lambda: [question_input]

                # Process and get the answer
                score, answer = model.process()

            # Display the result
            if score is not None:
                st.success("✅ Answer found!")
                st.markdown(
                    f"<div style='color: green; font-size: 18px;'><b>Answer:</b> {answer}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='color: orange; font-size: 16px;'><b>Confidence Score:</b> {round(score * 100, 2)}%</div>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ No relevant context found.")

    # Additional footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 14px; color: #6c757d;'>🛈 <b>Tips:</b> Ensure the URLs are accessible and provide factual information for best results.</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
