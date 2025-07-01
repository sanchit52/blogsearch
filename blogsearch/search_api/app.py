from flask import Flask, request, jsonify
from flask_cors import CORS
#for search engine
import sys
import os
from dotenv import load_dotenv
#till 
# Load environment variables
load_dotenv()
# Add the project root to Python path -----------------NEED TO ADD FILE
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from search_engine.atlas_search import AtlasBlogSearchEngine
    search_engine = AtlasBlogSearchEngine()
    print("✅ Search engine initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize search engine: {e}")
    search_engine = None
app = Flask(__name__)
CORS(app)
#@app.route("/search")
#def search():
#    query = request.args.get("q")
#    dummy_result = [
#        {
 #           "title": "How I became a Product Manager",
  #          "url": "https://manassaloi.com/2018/03/30/how-i-became-pm.html",
   #         "snippet": "A career retrospective..."
    #    }
    #]
    #return jsonify(dummy_result)
#new code added
@app.route("/search")
def search():
    try:
        if not search_engine:
            return jsonify({"error": "Search engine not available"}), 503
            
        query = request.args.get("q", "").strip()
        classification = request.args.get("type", None)  # 'personal' or 'non_personal'
        limit = int(request.args.get("limit", 10))
        
        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
            
        print(f"🔍 Search request: '{query}' (type: {classification}, limit: {limit})")
        
        # Perform search
        results = search_engine.hybrid_search(
            query=query,
            classification_filter=classification,
            limit=limit
        )
        
        # Format results for frontend
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result["title"],
                "url": result["url"],
                "snippet": result["content"],
                "type": result["classification"],
                "score": round(result["score"], 3)
            })
            
        print(f"✅ Found {len(formatted_results)} results")
        
        return jsonify({
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        })
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/stats")
def stats():
    """Get database statistics"""
    try:
        if not search_engine:
            return jsonify({"error": "Search engine not available"}), 503
            
        stats_data = search_engine.get_stats()
        return jsonify(stats_data)
        
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        if not search_engine:
            return jsonify({
                "status": "unhealthy", 
                "message": "Search engine not initialized"
            }), 503
            
        # Test database connection
        stats = search_engine.get_stats()
        
        return jsonify({
            "status": "healthy", 
            "message": "Blog search API is running",
            "database_connected": True,
            "total_blogs": stats.get("total_blogs", 0)
        })
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }), 503
#till here 
# 👇 This part actually starts the server
if __name__ == "__main__":
    print("🚀 Starting Blog Search API with MongoDB Atlas...")
    print("📍 Available endpoints:")
    print("   - GET /search?q=<query>&type=<personal|non_personal>&limit=<number>")
    print("   - GET /stats")
    print("   - GET /health")
    print(f"🌐 Server will run on: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)