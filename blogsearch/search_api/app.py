from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/search")
def search():
    query = request.args.get("q")
    dummy_result = [
        {
            "title": "How I became a Product Manager",
            "url": "https://manassaloi.com/2018/03/30/how-i-became-pm.html",
            "snippet": "A career retrospective..."
        }
    ]
    return jsonify(dummy_result)

# 👇 This part actually starts the server
if __name__ == "__main__":
    app.run(debug=True)
