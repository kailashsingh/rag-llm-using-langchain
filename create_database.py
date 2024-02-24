import os.path
import shutil

from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings

DATA_PATH = "data"
CHROMA_PATH = "chroma"


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=1,
        length_function=len,
        add_start_index=True,
        separators=["\n", ","],
        is_separator_regex=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Splitted {len(documents)} documents in {len(chunks)} chunks')

    random_index = 1
    chunk = chunks[random_index]
    print(f'Page content of chunk at index {random_index} is: {chunk.page_content}')
    print(f'Metadata of chunk at index {random_index} is: {chunk.metadata}')

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    # create new DB from documents
    db = Chroma.from_documents(
        chunks,
        # OpenAIEmbeddings(),
        embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f'Saved {len(chunks)} chunks to {CHROMA_PATH}')


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    generate_data_store()
