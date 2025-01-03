# SDLC Automate APP

## Overview
The SDLC Automate App is a Streamlit-based application that automates the generation of user stories, test cases, Cucumber scripts, and Selenium scripts from Business Requirement Documents (BRDs). It leverages Google Gemini AI for natural language processing and FAISS for efficient document retrieval.

## Features
- Upload BRD documents in PDF, DOCX, TXT, XLSX, and PPTX formats
- Extract text from uploaded documents
- Generate user stories from BRD documents
- Convert user stories into test cases
- Generate Cucumber scripts from test cases
- Generate Selenium scripts from test cases

## Installation
To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Environment Setup
Create a `.streamlit/secrets.toml` file and add your Google Gemini API key:

```toml
[GEMINI]
GEMINI_API_KEY = "your_api_key_here"
```

## Usage
Run the Streamlit application using:

```sh
streamlit run app.py
```

## Dependencies
The application requires the following dependencies:

- streamlit==1.38.0
- PyPDF2==3.0.1
- python-docx==0.8.11
- python-pptx==0.6.21
- pandas==2.1.1
- openai
- google-generativeai
- langchain
- langchain_google
- faiss-cpu==1.8.0
- langchain-community==0.3.0
- langchain-core==0.3.5
- scikit-learn
- openpyxl
- sentence-transformers

## Troubleshooting
If you encounter issues with missing dependencies, ensure all required packages are installed by running:

```sh
pip install -r requirements.txt
```

If you see an error related to `sentence-transformers`, install it manually:

```sh
pip install sentence-transformers
```

For issues related to LangChain imports, use the updated import path:

```python
from langchain_community.embeddings import SentenceTransformerEmbeddings
```

## License
This project is licensed under the MIT License.

