from flask import Flask
from flask_cors import CORS
from routes.emotion import bp
import model
from dotenv import load_dotenv
import os
import pymongo

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MongoDB Atlas setup
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["emotiondb"]
app.config["DB"] = db  # make DB accessible in routes if needed

# Register blueprint for API routes
app.register_blueprint(bp, url_prefix='/api')

# Train model on startup (or load pre-trained model if preferred)
if __name__ == '__main__':
    model.train()
    app.run(host='0.0.0.0', port=5000)

