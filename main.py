from torch.backends.mkldnn import verbose
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from langchain_community.vectorstores import Pinecone
from torch import cuda
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
import pinecone


def get_index() -> pinecone.Index:

    pinecone.init(
            api_key=os.environ.get('PINECONE_API_KEY') or '',
            environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
            )
    index_name = 'local-rag-chat'
    return pinecone.Index(index_name)

def get_embedding_model() -> HuggingFaceEmbeddings:
    embed_model_id = 'sentence-transformers/all-minilm-l6-v2'

    device = f'cuda:{cuda.current_device()}'

    return HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
        )

def get_vectorstore(index, embed_model) -> Pinecone:
    text_field = 'text' # name of field within the embedding's metadata that contains the text content
    return Pinecone(
        index, embed_model.embed_query, text_field
        )

def get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)

def get_model() -> AutoModelForCausalLM:
    # Had to pip install accellerate per error "ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate"
    params = {
        'low_cpu_mem_usage': True,
        'trust_remote_code': False, 
        'torch_dtype': torch.float16, 
        'use_safetensors': None,
        'device_map': 'auto', 
        'max_memory': None,
        'rope_scaling': {
            'type': 'linear',
            'factor': 4
        }
    }

    return AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", **params).cuda()

def get_llm(tokenizer) -> HuggingFacePipeline:

    params = {
        #'low_cpu_mem_usage': True,
        'trust_remote_code': False, 
        'torch_dtype': torch.float16, 
        'use_safetensors': None,
        'device_map': 'auto', 
        'max_memory': None
        # 'rope_scaling': {
        #     'type': 'linear',
        #     'factor': 4
        # }
    }

    gen_text_ppl = pipeline(
        model='deepseek-ai/deepseek-coder-6.7b-instruct', 
        tokenizer=tokenizer,
        return_full_text=True,
        temperature=0.0,
        max_new_tokens=512,
        repetition_penalty=1.1,
        **params
        )
    return HuggingFacePipeline(pipeline=gen_text_ppl)

def main():
    index = get_index()
    embed_model = get_embedding_model()
    vectorstore = get_vectorstore(index, embed_model)
    
    print('loading tokenizer')
    tokenizer = get_tokenizer()

    print('loading model')
    #model = get_model()

    print('creating llm obj')
    llm = get_llm(tokenizer)

    print('creating rag pipeline')
    rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type='stuff',
            verbose=True,
            retriever=vectorstore.as_retriever()
            )

    print('Enter Query')
    #prompt = input()
    prompt = "Please write a C script for the raspberry pi 'Pico W' that connects to an MQTT broker to publish events. The Pico W is similar to the Pico, but has a built-in wifi adapter called 'cyw43', which uses lwip."

    res = rag_pipeline(prompt)
    print(res)


if __name__ == "__main__":
    main()
