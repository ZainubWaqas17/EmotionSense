from flask import Blueprint, request, jsonify
from model import predict

bp = Blueprint('emotion', __name__)

@bp.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    result = predict(text)
    return jsonify({'emotion': result})
