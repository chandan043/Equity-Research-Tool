import streamlit as st
from final import QAModel

def main():
    st.title("Equity Research Tool")
    st.sidebar.title("News Article URLs")

    # Initialize the model
    model = QAModel()

    # Collect URLs from user input
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        if url:
            urls.append(url)

    question_input = st.text_input("Enter your question:", "What is the full form of NRI?")
    
    if st.button("Get Answer") and urls:
        # Replace the import_urls and import_question methods with user input
        model.import_urls = lambda: urls
        model.import_question = lambda: [question_input]

        # Process and get the answer
        score, answer = model.process()
        
        if score is not None:
            st.write(f"**Answer:** {answer}")
            st.write(f"**Score:** {round(score * 100, 2)}%")
        else:
            st.write("No answer found.")
    elif not urls:
        st.write("Please enter at least one URL.")

if __name__ == "__main__":
    main()
