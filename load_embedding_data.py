from langchain_community.document_loaders import PyPDFLoader

def load(paths):
    docs = []
    for path in paths:
        print(path)
        loader = PyPDFLoader(path)
        docs = docs + loader.load_and_split()
    docs = clean(docs)
    return docs


sequences_to_clean = [
    '\n',
    '\xa0'
    ]
def clean(docs):
    for doc in docs:
        for seq in sequences_to_clean:
            doc.page_content = doc.page_content.replace(seq, "")
    return docs


# for debugging
if __name__ == "__main__":
    docs = load(["/home/kaleb/Documents/pico-w-high-level-apis.pdf",
          "/home/kaleb/Documents/pico-w-networking-libs.pdf"])
    print(f'FIRST CHUNK: \n{docs[0]}')
    print(f'MIDDLE CHUNK: \n{docs[(int)(len(docs) / 2)]}')
    print(f'LAST CHUNK: \n{docs[len(docs) - 1]}')
    print(f'TOTAL CHUNKS: {len(docs)}')
    
