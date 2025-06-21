import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import ollama
import chromadb
import uuid

# Make sure you have an embedding model downloaded.
# You can run `ollama pull nomic-embed-text`
EMBEDDING_MODEL = 'nomic-embed-text'

# ChromaDB setup
client = chromadb.Client()
summarize_collection = client.get_or_create_collection("summarize")
expand_collection = client.get_or_create_collection("expand")
shorten_collection = client.get_or_create_collection("shorten")

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    result: str


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

def get_or_generate_response(collection, prompt_template: str, text: str):
    # Generate embedding for the input text
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    embedding = response["embedding"]

    # Query the collection for similar documents
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )

    if results['ids'] and results['ids'][0]:
        # Check the distance of the most similar result
        distance = results['distances'][0][0]
        if distance < 0.2:  # Threshold for similarity
            return TextResponse(result=results['documents'][0][0])

    # If no similar document is found, generate a new response
    prompt = prompt_template.format(text=text)
    response = ollama.generate(
        model='gemma3:1b',
        prompt=prompt
    )
    generated_text = response['response']

    # Store the new response in the database
    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[generated_text],
        metadatas=[{'original_text': text}]
    )

    return TextResponse(result=generated_text)


@app.post("/summarize", response_model=TextResponse)
def summarize(request: TextRequest):
    return get_or_generate_response(
        summarize_collection,
        "Please summarize the following text: \n\n{text}",
        request.text
    )

@app.post("/expand", response_model=TextResponse)
def expand(request: TextRequest):
    return get_or_generate_response(
        expand_collection,
        "Please expand the following text: \n\n{text}",
        request.text
    )


@app.post("/shorten", response_model=TextResponse)
def shorten(request: TextRequest):
    return get_or_generate_response(
        shorten_collection,
        "Please shorten the following text: \n\n{text}",
        request.text
    )


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
