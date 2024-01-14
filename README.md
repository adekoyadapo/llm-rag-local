# RAG Chat Assistant

This Python code implements a chat assistant using the RAG (Retrieval-Augmented Generation) architecture. It leverages the Langchain Community libraries for various functionalities, including vector stores, chat models, embeddings, document loaders, and more.

## Setup

### Dependencies

- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or [pyenv](https://github.com/pyenv/pyenv) - Manage python virtual environment (not necessary but recommended localization)
- python

### Conda Environment

Create and activate a Conda environment:

```bash
conda create -p .venv python=3.11
conda activate .venv
conda install pip
```

### Install Dependencies

Install the required dependencies using Conda:

```bash
pip install -r requirements.txt
```

## Usage

1. Download Model from [Huggingface llama](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin) to a directory locally and Set up your model directory by providing the model name with the directory in the `model_path`.

   ```python
   def load_model(
       model_path="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
       model_type="llama",
       max_new_tokens=512,
       content_length=400,
       temperature=0.9,
   ):
   ```

2. Specify the persistence directory for the vector database:

   ```python
   persist_directory = "./db/rag"  # Provide the path for the persistence directory
   ```

3. Run the code.

   ```bash
   streamlit run main.py
   ```

The assistant will start, and if the vector database is not present, it will be created using the provided documents in the `./docs/` directory.

## Chat Interface

The assistant provides a chat interface using Streamlit. Users can interact with the assistant by typing messages in the input box. The assistant processes the queries using the RAG architecture and responds accordingly.


## Dependencies

- torch
- streamlit
- ollama
- Hugginface
- langchain_community

## Notes

- The code supports both PDF and text documents for creating the vector database.
- The assistant's responses are displayed with a simulated typing effect.

Feel free to customize the code according to your specific use case and Ollama model. For more information, refer to the official documentation of [Langchain Community](https://github.com/langchain-ai/langchain).