from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os

os.environ["HUGGING_FACE_API_TOKEN"]= "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"

def similarity(reference, prediction):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_1= model.encode(reference, convert_to_tensor=True)
    embedding_2 = model.encode(prediction, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)[0,0]
    value = str(similarity).replace("tensor(","").replace(")","")
    return value

  

