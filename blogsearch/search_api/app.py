from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables all origins — fine for dev
