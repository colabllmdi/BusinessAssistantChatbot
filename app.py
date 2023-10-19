from flask import Flask, request, jsonify, render_template
from model import llama_2_model_response
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['user_message']
    print(f'Received user message: {user_message}')
    bot_response = llama_2_model_response(user_message)
    print(f'Generated bot response: {bot_response}')
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)