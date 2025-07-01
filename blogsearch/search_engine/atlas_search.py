import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import re
from collections import defaultdict
import math
import os
from dotenv import load_dotenv
import ssl

# Load environment variables
load_dotenv()

class AtlasBlogSearchEngine:
    def __init__(self):
        # Get connection details from environment
        self.connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        self.db_name = os.getenv('DATABASE_NAME', 'blogsearch')
        
        if not self.connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING not found in environment variables")
        
        # Connect to MongoDB Atlas
        try:
            #self.client = MongoClient(
            #    self.connection_string,
            #    ssl_cert_reqs=ssl.CERT_NONE,
            #    serverSelectionTimeoutMS=5000
            #)
            self.client = MongoClient(
            self.connection_string,
            serverSelectionTimeoutMS=5000
            )

            self.db = self.client[self.db_name]
            self.blogs_collection = self.db.blogs
            print("✅ Search engine connected to MongoDB Atlas")
            
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB Atlas: {e}")
            raise
        
        # Load TF-IDF components
        self.load_tfidf_models()
            
    def load_tfidf_models(self):
        """Load TF-IDF vectorizer and matrix"""
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        vectorizer_path = os.path.normpath(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        matrix_path = os.path.normpath(os.path.join(models_dir, "tfidf_matrix.pkl"))
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(matrix_path, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            print("✅ TF-IDF models loaded successfully")
            
        except FileNotFoundError:
            print("⚠️ TF-IDF models not found. Please run database setup first.")
            self.vectorizer = None
            self.tfidf_matrix = None
            
        except Exception as e:
            print(f"❌ Error loading TF-IDF models: {e}")
            self.vectorizer = None
            self.tfidf_matrix = None
            
    def preprocess_query(self, query):
        """Clean and preprocess search query"""
        # Remove special characters and normalize
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        query = ' '.join(query.split())
        return query
        
    def calculate_pagerank_scores(self):
        """Simple PageRank-like scoring based on content quality metrics"""
        try:
            blogs = list(self.blogs_collection.find({}, {
                '_id': 1, 'content': 1, 'title': 1, 'classification': 1
            }))
            scores = {}
            
            for blog in blogs:
                score = 1.0  # Base score
                
                # Length bonus (longer content often more valuable)
                content_length = len(blog.get('content', ''))
                if content_length > 1000:
                    score += 0.3
                elif content_length > 500:
                    score += 0.2
                    
                # Title quality bonus
                title = blog.get('title', '')
                if len(title) > 10 and len(title.split()) > 3:
                    score += 0.2
                    
                # Personal blog bonus (as per your project focus)
                if blog.get('classification') == 'personal':
                    score += 0.3
                    
                scores[str(blog['_id'])] = score
                
            return scores
            
        except Exception as e:
            print(f"Error calculating PageRank scores: {e}")
            return {}
        
    def tfidf_search(self, query, limit=20):
        """TF-IDF based search"""
        if not self.vectorizer or self.tfidf_matrix is None:
            print("⚠️ TF-IDF models not available, skipping TF-IDF search")
            return []
            
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-limit:][::-1]
            
            # Fetch corresponding documents
            blogs = list(self.blogs_collection.find())
            results = []
            
            for idx in top_indices:
                if idx < len(blogs) and similarities[idx] > 0.1:  # Minimum similarity threshold
                    blog = blogs[idx]
                    results.append({
                        'blog': blog,
                        'tfidf_score': float(similarities[idx])
                    })
                    
            return results
            
        except Exception as e:
            print(f"Error in TF-IDF search: {e}")
            return []
        
    def atlas_text_search(self, query):
        """MongoDB Atlas text search for exact matches"""
        try:
            # Use MongoDB's text search with Atlas
            pipeline = [
                {
                    "$match": {
                        "$text": {"$search": query}
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "textScore"}
                    }
                },
                {
                    "$sort": {"score": {"$meta": "textScore"}}
                },
                {
                    "$limit": 20
                }
            ]
            
            results = list(self.blogs_collection.aggregate(pipeline))
            return results
            
        except Exception as e:
            print(f"Error in Atlas text search: {e}")
            # Fallback to regex search
            return self.fallback_search(query)
            
    def fallback_search(self, query):
        """Fallback search using regex"""
        try:
            # Simple regex search as fallback
            regex_pattern = {"$regex": query, "$options": "i"}
            results = list(self.blogs_collection.find({
                "$or": [
                    {"title": regex_pattern},
                    {"content": regex_pattern}
                ]
            }).limit(20))
            
            # Add artificial score
            for result in results:
                result['score'] = 1.0
                
            return results
            
        except Exception as e:
            print(f"Error in fallback search: {e}")
            return []
        
    def hybrid_search(self, query, classification_filter=None, limit=10):
        """Combine TF-IDF, Atlas text search, and PageRank"""
        try:
            processed_query = self.preprocess_query(query)
            
            # Get TF-IDF results
            tfidf_results = self.tfidf_search(processed_query, limit * 2)
            
            # Get Atlas text search results
            text_results = self.atlas_text_search(query)
            
            # Get PageRank scores
            pagerank_scores = self.calculate_pagerank_scores()
            
            # Combine and score results
            combined_results = {}
            
            # Process TF-IDF results
            for result in tfidf_results:
                blog_id = str(result['blog']['_id'])
                combined_results[blog_id] = {
                    'blog': result['blog'],
                    'tfidf_score': result['tfidf_score'],
                    'text_score': 0,
                    'pagerank_score': pagerank_scores.get(blog_id, 1.0)
                }
                
            # Process text search results
            for blog in text_results:
                blog_id = str(blog['_id'])
                text_score = blog.get('score', 0)
                
                if blog_id in combined_results:
                    combined_results[blog_id]['text_score'] = text_score
                else:
                    combined_results[blog_id] = {
                        'blog': blog,
                        'tfidf_score': 0,
                        'text_score': text_score,
                        'pagerank_score': pagerank_scores.get(blog_id, 1.0)
                    }
                    
            # Calculate final scores and format results
            final_results = []
            for blog_id, data in combined_results.items():
                # Weighted combination of scores
                final_score = (
                    0.5 * data['tfidf_score'] +
                    0.3 * min(data['text_score'] / 10.0, 1.0) +  # Normalize text score
                    0.2 * data['pagerank_score']
                )
                
                blog = data['blog']
                
                # Apply classification filter if specified
                if classification_filter and blog.get('classification') != classification_filter:
                    continue
                    
                # Create snippet
                content = blog.get('content', '')
                snippet = content[:300] + '...' if len(content) > 300 else content
                
                final_results.append({
                    'title': blog.get('title', 'No Title'),
                    'url': blog.get('url', ''),
                    'content': snippet,
                    'classification': blog.get('classification', 'unknown'),
                    'score': final_score
                })
                
            # Sort by final score
            final_results.sort(key=lambda x: x['score'], reverse=True)
            
            return final_results[:limit]
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []
    
    def get_stats(self):
        """Get database statistics"""
        try:
            total_blogs = self.blogs_collection.count_documents({})
            personal_blogs = self.blogs_collection.count_documents({"classification": "personal"})
            non_personal_blogs = self.blogs_collection.count_documents({"classification": "non_personal"})
            
            return {
                "total_blogs": total_blogs,
                "personal_blogs": personal_blogs,
                "non_personal_blogs": non_personal_blogs
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_blogs": 0, "personal_blogs": 0, "non_personal_blogs": 0}

# Test the search engine
if __name__ == "__main__":
    try:
        print("🔍 Testing Atlas Blog Search Engine...")
        engine = AtlasBlogSearchEngine()
        
        # Test search
        test_queries = ["product manager", "machine learning", "startup"]
        
        for query in test_queries:
            print(f"\n🔎 Testing query: '{query}'")
            results = engine.hybrid_search(query, limit=3)
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['title'][:50]}...")
                print(f"      Type: {result['classification']}")
                print(f"      Score: {result['score']:.3f}")
                
        # Test stats
        stats = engine.get_stats()
        print(f"\n📊 Database Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")