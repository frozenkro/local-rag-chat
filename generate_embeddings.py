from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from load_embedding_data import load 
import os 
import pinecone
import time
import uuid

docs = load(["/home/kaleb/Documents/pico-w-high-level-apis.pdf",
      "/home/kaleb/Documents/pico-w-networking-libs.pdf"])

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
    )

texts = [doc.page_content for doc in docs]
embeddings = embed_model.embed_documents(texts)

print(embeddings[0])
print(f'EMBEDDING COMPLETE')
print(f'TOTAL EMBEDDINGS: {len(embeddings)}')
print(f'EMBEDDING DIMENSIONALITY: {len(embeddings[0])}')

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
    )

index_name = 'local-rag-chat'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
        )
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pinecone.Index(index_name)

batch_size = 32

for i in range(0, len(docs), batch_size):
    i_end = min(len(docs), i+batch_size)
    batch = docs[i:i_end]

    vectors = []
    for doc in batch:
        id = uuid.uuid4()
        text = doc.page_content
        #doc metadata has source and page # already
        metadata = doc.metadata
        metadata['text'] = doc.page_content
        print(id)
        vector = {
            "id": str(id),
            "values": embeddings[i],
            "metadata": metadata
            }
        vectors.append(vector)
    index.upsert(vectors=vectors) 
