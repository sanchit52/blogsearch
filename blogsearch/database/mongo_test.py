from pymongo import MongoClient

uri = "mongodb+srv://bloguser:Pratyaksha%4009@cluster0.k2vwhyk.mongodb.net/blogsearch?retryWrites=true&w=majority&appName=Cluster0"

try:
    client = MongoClient(uri)
    print("✅ Connected! Databases:", client.list_database_names())
except Exception as e:
    print("❌ ERROR:", e)