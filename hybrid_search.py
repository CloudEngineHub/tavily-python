from dotenv import load_dotenv
load_dotenv()

import os
from pymongo import MongoClient
from tavily import TavilyHybridClient

db = MongoClient(os.getenv("MONGO_URI"))["hybrid_search_test"]

hybrid_search = TavilyHybridClient(
    api_key=os.environ["TAVILY_API_KEY"],
    collection=db.get_collection('data'),
    index='vector_search',
    embeddings_field='embeddings',
    content_field='content'
)

results = hybrid_search.search("Who is Leo Messi?", max_results=5, max_local=123123, max_foreign=14, save_foreign=True)

print(results)