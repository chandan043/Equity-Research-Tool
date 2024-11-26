
# Equity Research Tool

## Overview

The **Equity Research Tool** is a web-based application designed to assist users in retrieving and analyzing financial information from news articles. By leveraging advanced Natural Language Processing (NLP) techniques, it enables users to input URLs of news articles and ask specific questions to extract relevant answers from the content.

## Features

- **Customizable URL Input**: Users can provide their own list of article URLs.
- **Question-Answering**: Leverages machine learning models to find precise answers to user queries based on the article content.
- **Semantic Search**: Identifies the most relevant parts of the content for accurate answers.
- **Caching**: Uses cached embeddings to improve performance on repeated queries.

## Technologies Used

- **Backend**: Python
- **Frontend**: Streamlit
- **NLP Models**:
  - Sentence Transformers (`all-MiniLM-L6-v2`) for semantic search.
  - DistilBERT (`distilbert-base-cased-distilled-squad`) for question answering.
- **Utilities**: 
  - LangChain for text splitting.
  - Hugging Face Transformers for model pipelines.
  - `pickle` for data caching.

## Installation

### Prerequisites

- Python 3.8 or above
- Streamlit (`pip install streamlit`)
- Required Python libraries (listed in `requirements.txt`)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/equity-research-tool.git
   cd equity-research-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Provide Article URLs**: 
   - Enter up to three article URLs in the sidebar input fields.
   
2. **Enter Your Question**: 
   - Type your query in the main input field (e.g., *What is the full form of NRI?*).

3. **Get the Answer**:
   - Click the "Get Answer" button.
   - The tool retrieves the content, performs semantic search, and displays the most relevant answer along with a confidence score.

## File Structure

- **`final.py`**: Core logic for handling text processing, semantic search, and question answering.
- **`main.py`**: Streamlit interface for user interaction.
- **`requirements.txt`**: List of required Python libraries.

## Example

1. Start the tool using `streamlit run main.py`.
2. Enter a news article URL such as:
   ```
   https://www.moneycontrol.com/news/business/ipo/ntpc-green-shortlists-four-i-banks-for-rs-10000-crore-ipo-12620441.html
   ```
3. Ask a question like:
   ```
   What is the full form of NRI?
   ```
4. View the result:
   - **Answer**: Non-Resident Indian
   - **Score**: 92.3%
   - **TestCase 1:**
     ![Screenshot 2024-08-20 201750](https://github.com/user-attachments/assets/1a6a2f97-c561-4f0b-9111-006454e0b4ca)
   - **TestCase 2:**
     ![Screenshot 2024-08-20 202340](https://github.com/user-attachments/assets/08fd030f-1316-4966-9ae5-de3cec093a33)
   - **TestCase 3:**
     ![Screenshot 2024-08-20 212926](https://github.com/user-attachments/assets/6eee909d-b73a-4dcd-ac3a-753935f6f99f)

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

- **Name**: KOLLOJU CHANDAN 
- **Email**: kollojuchandan123@gmail.com  
- **GitHub**: https://github.com/chandan043





