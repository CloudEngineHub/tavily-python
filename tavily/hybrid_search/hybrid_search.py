from pymongo import MongoClient
import os
from openai import OpenAI
from tavily import TavilyClient

openai = OpenAI()

def embed_openai(text):
    return openai.embeddings.create(input=[text], model='text-embedding-3-large', dimensions=256).data[0].embedding

class TavilyHybridClient():
    def __init__(self, api_key, collection, index, embeddings_field='embeddings', content_field='content'):
        self.tavily = TavilyClient(api_key)
        self.collection = collection
        self.index = index
        self.embeddings_field = embeddings_field
        self.content_field = content_field
        self.embedding_function = embed_openai
        
        # Check that the index specified by the parameters exists and is a valid vector search index
        index_exists = False
        for index in collection.list_search_indexes():
            if index['name'] != self.index:
                continue
            
            if index['type'] != 'vectorSearch':
                raise ValueError(f"Index '{self.index}' exists but is not of type 'vectorSearch'.")
            
            field_exists = False
            for field in index['latestDefinition']['fields']:
                if field['path'] != self.embeddings_field:
                    continue
                
                if field['type'] != 'vector':
                    raise ValueError(f"Field '{self.embeddings_field}' exists but is not of type 'vector'.")
                elif field['similarity'] != 'cosine':
                    raise ValueError(f"Field '{self.embeddings_field}' exists but has similarity '{field['similarity']}' instead of 'cosine'.")
                
                field_exists = True
                break
            
            if not field_exists:
                raise ValueError(f"Field '{self.embeddings_field}' does not exist in index '{self.index}'.")
            
            index_exists = True
            
        if not index_exists:
            raise ValueError(f"Index '{self.index}' does not exist.")

    def search(self, query, max_results=10, max_local=None, max_foreign=None, save_foreign=False):
        '''
        Return results for the given query from both the tavily API (foreign) and the specified mongo collection (local).
        
        Parameters:
        query (str): The query to search for.
        collection_name (str): The name of the collection to search in.
        max_results (int): The maximum number of results to return.
        max_local (int): The maximum number of local results to return.
        max_foreign (int): The maximum number of foreign results to return.
        '''

        if max_local is None:
            max_local = max_results
        
        if max_foreign is None:
            max_foreign = max_results

        query_embeddings = self.embedding_function(query)

        # Search the local collection
        local_results = list(self.collection.aggregate([
            {
                "$vectorSearch": {
                    "index": self.index,
                    "path": self.embeddings_field,
                    "queryVector": query_embeddings,
                    "numCandidates": 30,
                    "limit": min(max_results, max_local)
                }
            },
            {
                "$project": {
                    "_id": 0,
                    self.content_field: 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]))

        # Search using tavily
        foreign_results = [
            {'content': result['content'], 'score': result['score']} for result in self.tavily.search(query, max_results=max_foreign)['results']
        ] if max_foreign > 0 else []

        # Combine the results
        combined_results = local_results + foreign_results

        # Sort the combined results by score
        combined_results.sort(key=lambda x: x['score'], reverse=True)

        if len(combined_results) > max_results:
            combined_results = combined_results[:max_results]

        if save_foreign:
            for result in foreign_results:
                self.collection.insert_one({
                    self.content_field: result['content'],
                    self.embeddings_field: self.embedding_function(result['content'])
                })

        return combined_results