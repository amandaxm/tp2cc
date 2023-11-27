from flask import Flask,redirect, request,url_for,jsonify,render_template,make_response,json
import pickle
import pandas as pd
from flask_cors import CORS
import os
import time

app = Flask(__name__)
CORS(app)
version = os.environ['version']

with open("data/rules.pickle", "rb") as handle:
    app.model = pickle.load(handle)

def generate_recommendations(model, songs):
    recommendation = set()
    for rule in model.iterrows():
        antecedents = rule[1]['antecedents']
        confidence = rule[1]['confidence']
        for musicaReq in songs:
            if musicaReq in antecedents and confidence > 0.5:
                recommendation.update(rule[1]['consequents'])
                break

    return list(recommendation)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommender', methods=['POST'])
def recommend():
    try:
        data = request.get_json(force=True)
        songs = data['songs']

        recommendations = generate_recommendations(app.model, songs)
        
        return jsonify({
            'playlist': recommendations,
            'version': version,
            'model_date': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=32165, debug=True)
