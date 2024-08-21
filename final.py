import pickle
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

class QAModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_pipeline = pipeline("question-answering", 
                                    model="distilbert/distilbert-base-cased-distilled-squad", 
                                    revision="626af31")
    
    def get_cache_key(self, urls):
        return hashlib.md5(''.join(sorted(urls)).encode()).hexdigest()

    def load_cache(self, cache_key):
        try:
            with open(f"{cache_key}_cache.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def save_cache(self, cache_key, chunks_list, corpus_embeddings):
        with open(f"{cache_key}_cache.pkl", "wb") as f:
            pickle.dump((chunks_list, corpus_embeddings), f)
    
    def import_urls(self):
        return [
            "https://www.chittorgarh.com/ipo_subscription/ola-electric-ipo/1765/",
            "https://www.moneycontrol.com/news/business/ipo/ntpc-green-shortlists-four-i-banks-for-rs-10000-crore-ipo-12620441.html",
            "https://www.moneycontrol.com/news/business/markets/godfrey-phillips-to-exit-retail-business-division-24seven-stock-up-12626491.html",
            "https://www.moneycontrol.com/news/technology/electoral-bonds-it-companies-infosys-cyient-and-zensar-technologies-mentioned-among-donors-12483941.html"
        ]

    def import_question(self):
        return ["What is the full form of NRI?"]

    def load_data(self, urls):
        loader = UnstructuredURLLoader(urls=urls)
        return loader.load()

    def split_text(self, data):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=450,
            chunk_overlap=0
        )
        return text_splitter.split_documents(data)

    def encode_corpus(self, chunks_list):
        return self.model.encode(chunks_list)
    
    def encode_query(self, question):
        return self.model.encode(question)
    
    def semantic_search(self, query_embedding, corpus_embeddings):
        results = util.semantic_search(query_embedding, corpus_embeddings)
        if results:
            return results[0][0]['corpus_id']
        return None

    def get_answer(self, chunks_list, corpus_id, question):
        context = chunks_list[corpus_id]
        return self.qa_pipeline(question=question, context=context)

    def process(self):
        # Step 1: Import URLs and Question
        urls = self.import_urls()
        question = self.import_question()

        # Step 2: Check Cache
        cache_key = self.get_cache_key(urls)
        cached_data = self.load_cache(cache_key)
        
        if cached_data:
            # Use cached data
            chunks_list, corpus_embeddings = cached_data
        else:
            # Step 3: Load Data
            data = self.load_data(urls)
            
            # Step 4: Split Text into Chunks
            chunks = self.split_text(data)
            chunks_list = [str(chunk) for chunk in chunks]
            
            # Step 5: Encode Corpus
            corpus_embeddings = self.encode_corpus(chunks_list)
            
            # Save to cache
            self.save_cache(cache_key, chunks_list, corpus_embeddings)
        
        # Step 6: Encode Query
        query_embedding = self.encode_query(question)
        
        # Step 7: Semantic Search
        corpus_id = self.semantic_search(query_embedding, corpus_embeddings)
        
        if corpus_id is not None:
            # Step 8: Get Answer
            qa_result = self.get_answer(chunks_list, corpus_id, question)
            return qa_result['score'], qa_result['answer']
        else:
            return None, "No relevant context found."
