import json
import pymongo
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os
from dotenv import load_dotenv
import ssl

# Load environment variables
load_dotenv()

class BlogSearchAtlasDB:
    def __init__(self):
        # Get connection string from environment
        self.connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        self.db_name = os.getenv('DATABASE_NAME', 'blogsearch')
        
        if not self.connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING not found in environment variables")
        
        # Connect to MongoDB Atlas
        try:
            self.client = MongoClient(
                self.connection_string,
                #ssl_cert_reqs=ssl.CERT_NONE,
                serverSelectionTimeoutMS=5000
            )
            # Test the connection
            self.client.admin.command('ping')
            print("✅ Successfully connected to MongoDB Atlas!")
            
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB Atlas: {e}")
            raise
            
        self.db = self.client[self.db_name]
        self.blogs_collection = self.db.blogs
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
    def test_connection(self):
        """Test the Atlas connection"""
        try:
            # Ping the database
            self.client.admin.command('ping')
            
            # Get cluster info
            server_info = self.client.server_info()
            print(f"Connected to MongoDB version: {server_info['version']}")
            
            # Test database operations
            test_doc = {"test": "connection", "timestamp": "2024"}
            result = self.db.test_collection.insert_one(test_doc)
            print(f"Test document inserted with ID: {result.inserted_id}")
            
            # Clean up test document
            self.db.test_collection.delete_one({"_id": result.inserted_id})
            print("Test document cleaned up")
            
            return True
            
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
        
    def create_indexes(self):
        """Create database indexes for faster searches"""
        try:
            # Create text index for full-text search
            self.blogs_collection.create_index([
                ("title", "text"), 
                ("content", "text")
            ])
            
            # Create individual indexes
            self.blogs_collection.create_index("classification")
            self.blogs_collection.create_index("url")
            self.blogs_collection.create_index("label")
            
            print("✅ Database indexes created successfully")
            
        except Exception as e:
            print(f"❌ Failed to create indexes: {e}")
        
    def load_blog_data(self, personal_file, non_personal_file):
        """Load classified blog data into MongoDB Atlas"""
        print("📊 Loading blog data into MongoDB Atlas...")
        
        try:
            # Clear existing data
            delete_result = self.blogs_collection.delete_many({})
            print(f"🗑️ Cleared {delete_result.deleted_count} existing documents")
            
            blogs_data = []
            all_texts = []
            
            # Load personal blogs
            if os.path.exists(personal_file):
                with open(personal_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            blog = json.loads(line.strip())
                            blog['classification'] = 'personal'
                            blog['label'] = 1
                            blogs_data.append(blog)
                            
                            # Combine title and content for TF-IDF
                            text = f"{blog.get('title', '')} {blog.get('content', '')}"
                            all_texts.append(text)
                            
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipped line {line_num} in personal file: {e}")
                            continue
                            
                print(f"📝 Loaded {len([b for b in blogs_data if b['classification'] == 'personal'])} personal blogs")
            else:
                print(f"❌ Personal blogs file not found: {personal_file}")
                    
            # Load non-personal blogs
            if os.path.exists(non_personal_file):
                personal_count = len(blogs_data)
                with open(non_personal_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            blog = json.loads(line.strip())
                            blog['classification'] = 'non_personal'
                            blog['label'] = 0
                            blogs_data.append(blog)
                            
                            # Combine title and content for TF-IDF
                            text = f"{blog.get('title', '')} {blog.get('content', '')}"
                            all_texts.append(text)
                            
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipped line {line_num} in non-personal file: {e}")
                            continue
                            
                non_personal_count = len(blogs_data) - personal_count
                print(f"🏢 Loaded {non_personal_count} non-personal blogs")
            else:
                print(f"❌ Non-personal blogs file not found: {non_personal_file}")
            
            # Insert into MongoDB Atlas (in batches for better performance)
            if blogs_data:
                batch_size = 1000
                total_inserted = 0
                
                for i in range(0, len(blogs_data), batch_size):
                    batch = blogs_data[i:i + batch_size]
                    try:
                        result = self.blogs_collection.insert_many(batch, ordered=False)
                        total_inserted += len(result.inserted_ids)
                        print(f"📤 Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} documents")
                    except Exception as e:
                        print(f"⚠️ Error inserting batch {i//batch_size + 1}: {e}")
                        
                print(f"✅ Total documents inserted: {total_inserted}")
                
                # Create and save TF-IDF vectors
                print("🔧 Creating TF-IDF vectors...")
                if all_texts:
                    tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                    
                    # Save vectorizer and matrix
                    os.makedirs('models', exist_ok=True)
                    
                    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
                        pickle.dump(self.vectorizer, f)
                        
                    with open('models/tfidf_matrix.pkl', 'wb') as f:
                        pickle.dump(tfidf_matrix, f)
                        
                    print("✅ TF-IDF vectors created and saved")
                    
                return total_inserted
            else:
                print("❌ No blog data to insert")
                return 0
                
        except Exception as e:
            print(f"❌ Error loading blog data: {e}")
            raise
            
    def get_collection_stats(self):
        """Get statistics about the collections"""
        try:
            total_blogs = self.blogs_collection.count_documents({})
            personal_blogs = self.blogs_collection.count_documents({"classification": "personal"})
            non_personal_blogs = self.blogs_collection.count_documents({"classification": "non_personal"})
            
            stats = {
                "total_blogs": total_blogs,
                "personal_blogs": personal_blogs,
                "non_personal_blogs": non_personal_blogs,
                "database_name": self.db_name
            }
            
            print(f"\n📊 Database Statistics:")
            print(f"   Total blogs: {total_blogs}")
            print(f"   Personal blogs: {personal_blogs}")
            print(f"   Non-personal blogs: {non_personal_blogs}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}

# Test script
if __name__ == "__main__":
    try:
        print("🚀 Initializing MongoDB Atlas connection...")
        db = BlogSearchAtlasDB()
        
        # Test connection
        if db.test_connection():
            print("✅ Connection test passed!")
            
            # Create indexes
            db.create_indexes()
            #new added code----------------------------
            script_dir = os.path.dirname(__file__)
            personal_path = os.path.join(script_dir, "..", "data", "processed", "final", "personal_blogs_fixed.jsonl")
            non_personal_path = os.path.join(script_dir, "..", "data", "processed", "final", "non_personal_blogs_fixed.jsonl")
            #personal_path = os.path.normpath(personal_path)
            #non_personal_path = os.path.normpath(non_personal_path)

            # Load data from your processed files
            #personal_file = "..data/processed/final/personal_blogs_fixed.jsonl"
            #non_personal_file = "..data/processed/final/non_personal_blogs_fixed.jsonl"
            
            if os.path.exists(personal_path) or os.path.exists(non_personal_path):
                count = db.load_blog_data(personal_path, non_personal_path)
                print(f"✅ Database setup complete! {count} blogs indexed.")
                
                # Show stats
                db.get_collection_stats()
            else:
                print("⚠️ Blog data files not found. Please check file paths:")
                print(f"   Looking for: {personal_file}")
                print(f"   Looking for: {non_personal_file}")
        else:
            print("❌ Connection test failed!")
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check your .env file contains the correct MONGODB_CONNECTION_STRING")
        print("2. Ensure your IP address is whitelisted in Atlas Network Access")
        print("3. Verify your database user credentials are correct")
