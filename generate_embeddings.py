from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from load_embedding_data import load 
import os 
import pinecone

docs = load(["/home/kaleb/Documents/pico-w-high-level-apis.pdf",
      "/home/kaleb/Documents/pico-w-networking-libs.pdf"])

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
    )


embeddings = embed_model.embed_documents(docs)

print(embeddings[0])
print(f'EMBEDDING COMPLETE')
print(f'TOTAL EMBEDDINGS: {len(embeddings)}')
print(f'EMBEDDING DIMENSIONALITY: {len(embeddings[0])}')

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
        )
