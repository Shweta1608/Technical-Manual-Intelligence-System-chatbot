from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader

DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    pdf_documents=loader.load()
    return pdf_documents

pdf_documents=load_pdf_files(data=DATA_PATH)


from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable
from langchain.schema import Document


def fetch_youtube_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item["text"] for item in transcript])
        return Document(
            page_content=transcript_text,
            metadata={"source": video_url}
        )
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video: {video_url}")
        return None
    except VideoUnavailable:
        print(f"Video is unavailable: {video_url}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

video_urls = [
    "https://www.youtube.com/watch?v=a4ebXN4uSWw&list=PLEPeb9u_3PEBFycm2kL8gx1oETetV9Vbe&index=3",
    "https://www.youtube.com/watch?v=QV95jehYnIo&list=PLEPeb9u_3PEBFycm2kL8gx1oETetV9Vbe&index=4"
]

all_documents = []
for url in video_urls:
    doc = fetch_youtube_transcript(url)
    if doc:
        all_documents.append(doc)
        
#print(all_documents)

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader(["https://www.mprofit.in/blog" , "https://www.mprofit.in/features" ,"https://www.mprofit.in/pricing/investors" , "https://www.mprofit.in/pricing/wealth"])
web_documents = loader.load_and_split()

#docs = pdf_documents.load()
combined_documents = pdf_documents + all_documents + web_documents
#print(combined_documents)


def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=combined_documents)
#print("Length of Text Chunks: ", len(text_chunks))

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)