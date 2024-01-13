import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import warnings

warnings.filterwarnings('ignore')

@st.cache_resource
def init_retriever():
    return SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

retriever = init_retriever()

# Load FAISS index and metadata from disk
index = faiss.read_index("faiss.index")
with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

def card(thumbnail, title, url, context):
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
            <div class="col-md-4 col-sm-4">
                 <div class="position-relative">
                     <a href={url}><img src={thumbnail} class="img-fluid" style="width: 192px; height: 106px"></a>
                 </div>
             </div>
             <div  class="col-md-8 col-sm-8">
                 <a href={url}>{title}</a>
                 <br>
                 <span style="color: #808080;">
                     <small>{context[:200].capitalize()+"...."}</small>
                 </span>
             </div>
        </div>
     </div>
        """, unsafe_allow_html=True)
    
st.write("""
# YouTube Q&A
Ask me a question!
""")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

query = st.text_input("Search!", "")

if query != "":
    xq = retriever.encode([query]).tolist()

    D, I = index.search(np.array(xq), k=10)

    count = 1
    displayed_ids = set()
    for i in I[0]:
        if meta[i]['video_id'] not in displayed_ids and count <= 5:
            card(
                meta[i]['thumbnail'],
                meta[i]['title'],
                meta[i]['url'],
                meta[i]['text']
            )
            displayed_ids.add(meta[i]['video_id'])
            count += 1