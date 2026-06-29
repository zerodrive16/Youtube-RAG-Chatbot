import os
import uuid
import whisper
import yt_dlp
import tempfile
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
COLLECTION_NAME = "youtube"
CHUNK_SIZE = 500 # max tokens / words per chunk 
CHUNK_OVERLAP = 50 # context preservation 

qdrant = QdrantClient(host="localhost", port = 6333)
openai_client = OpenAI()

# Create the Collecition
if not qdrant.collection_exists(COLLECTION_NAME): 
    qdrant.create_collection(
        collection_name=COLLECTION_NAME, 
        vectors_config= VectorParams(size = EMBED_DIM, distance=Distance.COSINE)
    )


# Download the audio from youtube
def download_audio(youtube_url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return os.path.join(tmp_dir, "audio.mp3")


def transcribe(audio_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def ingest(youtube_url: str):
    audio_path = download_audio(youtube_url)
    text = transcribe(audio_path)
    chunks = chunk_text(text)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embed_text(chunk),
            payload={"text": chunk, "source": youtube_url},
        )
        for chunk in chunks
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def search(query: str, top_k: int = 5) -> list[str]:
    vector = embed_text(query)
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=top_k)
    return [hit.payload["text"] for hit in results]


# Prompt Engineering openai api 
def chat(query: str, history: list[dict]) -> str: 
    context = "\n\n".join(search(query))
    system_prompt = (
        "You are a helpful assistant. Answer questions based on the following "
        "context from the uploaded youtube url. "
        "If the answer cannot be found in the context, say so honestly.\n\n"
        f"Context:\n{context}"
    )
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]
    response = openai_client.chat.completions.create(model = "gpt-4o-mini", messages = messages)
    return response.choices[0].message.content