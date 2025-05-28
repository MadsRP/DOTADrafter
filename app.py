# app.py - Fixed imports for working integration
import json
import os

from flask import Flask, render_template, request, jsonify

from data_collector import fetch_hero_data, fetch_hero_portraits
from dynamic_winrates_integration import get_draft_win_probabilities, get_all_hero_winrates

app = Flask(__name__)


def get_hero_data():
    if os.path.exists('data/heroes.json'):
        with open('data/heroes.json', 'r') as f:
            return json.load(f)
    else:
        os.makedirs('data', exist_ok=True)
        heroes = fetch_hero_data()
        fetch_hero_portraits(heroes)
        with open('data/heroes.json', 'w') as f:
            json.dump(heroes, f)
        return heroes


@app.route('/')
def index():
    hero_data = get_hero_data()
    return render_template('index.html', heroes=hero_data)


@app.route('/api/analysis', methods=['POST'])
def get_analysis():
    draft_state = request.json
    analysis = get_draft_win_probabilities(draft_state)
    return jsonify({'analysis': analysis})


@app.route('/api/hero-winrates', methods=['POST'])
def get_hero_winrates():
    try:
        draft_state = request.json
        hero_winrates = get_all_hero_winrates(draft_state)

        return jsonify({'hero_winrates': hero_winrates})

    except Exception as e:
        return jsonify({'error': str(e), 'hero_winrates': {}}), 500


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Starting Dota 2 Draft Predictor Flask App...")
    print("Available endpoints:")
    print("- GET  /: Main application")
    print("- POST /api/counterpicks: Get hero recommendations")
    print("- GET  /api/model-status: Check ML model status")

    app.run(debug=True)
