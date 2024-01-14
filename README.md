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

1. Download Model from [Huggingface llama](https://cdn-lfs.huggingface.co/repos/30/e3/30e3aca7233f7337633262ff6d59dd98559ecd8982e7419b39752c8d0daae1ca/3bfdde943555c78294626a6ccd40184162d066d39774bd2c98dae24943d32cc3?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27llama-2-7b-chat.ggmlv3.q8_0.bin%3B+filename%3D%22llama-2-7b-chat.ggmlv3.q8_0.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1705429180&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwNTQyOTE4MH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy8zMC9lMy8zMGUzYWNhNzIzM2Y3MzM3NjMzMjYyZmY2ZDU5ZGQ5ODU1OWVjZDg5ODJlNzQxOWIzOTc1MmM4ZDBkYWFlMWNhLzNiZmRkZTk0MzU1NWM3ODI5NDYyNmE2Y2NkNDAxODQxNjJkMDY2ZDM5Nzc0YmQyYzk4ZGFlMjQ5NDNkMzJjYzM%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=YwoRKJd8YkiR5cUitrPiLc2OA45RoOtWUgrz7Ks8N6rs5t5agY-kzfqRayBPTGW2EyGV-dEFeXoOYQgdj7J2BQf5GasmQoIumL5kMpi0eJpG7lP38mkGefmgEAus8Bcan07xI6QFIODYtE0I1y4-J8ZcYxUfFRtjAEbXJAFR2kr2anYMUe4MESc5R7xHp6GoHud%7EkCcJ66ddZUyQ5aCYS2CoR3fZUMtQzOlBhexYv10PJPPdV1zjvr9aRQFsqoBNFbdCZNifVR6VP49Hh-zPdckOlXosc9b03LpQhTNTAH8XNvwfyFo8EgyrtQb3xTSdcfSUmycYnVGkIqg2Cu2eVQ__&Key-Pair-Id=KVTP0A1DKRTAX) to a directory locally and Set up your model directory by providing the model name with the directory in the `model_path`.

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