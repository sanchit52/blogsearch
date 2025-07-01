from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

try:
    # Get connection string from environment
    connection_string = os.getenv('MONGODB_URI')
    
    # Create client
    client = MongoClient(connection_string)
    
    # Test connection
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
    
    # List databases
    print("Available databases:", client.list_database_names())
    
    # Test creating a collection
    db = client['blogsearch']
    collection = db['test']
    
    # Insert a test document
    test_doc = {"message": "Hello MongoDB Atlas!", "test": True}
    result = collection.insert_one(test_doc)
    print(f"✅ Test document inserted with ID: {result.inserted_id}")
    
    # Retrieve the document
    found_doc = collection.find_one({"test": True})
    print(f"✅ Retrieved document: {found_doc}")
    
    # Clean up test document
    collection.delete_one({"_id": result.inserted_id})
    print("✅ Test document cleaned up")
    
    client.close()
    print("✅ Connection test completed successfully!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting tips:")
    print("1. Check your username and password")
    print("2. Verify your IP is whitelisted in Network Access")
    print("3. Make sure you replaced <password> with your actual password")
    print("4. Check if your connection string is properly formatted")