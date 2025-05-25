# app.py - Simple fix to get back to working state
import json
import os

from flask import Flask, render_template, request, jsonify

from data_collector import fetch_hero_data, fetch_hero_portraits

# Go back to the working integration
try:
    from improved_lightweight_integration import get_hero_recommendations, get_draft_win_probabilities
    print("✅ Using improved_lightweight_integration")
except ImportError:
    try:
        from lightweight_integration import get_hero_recommendations, get_draft_win_probabilities
        print("✅ Using lightweight_integration")
    except ImportError:
        print("❌ No integration found - check your files")

app = Flask(__name__)

# Load or fetch hero data
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

# Home page route
@app.route('/')
def index():
    hero_data = get_hero_data()
    return render_template('index.html', heroes=hero_data)

# API endpoint to get counter picks - SIMPLIFIED
@app.route('/api/counterpicks', methods=['POST'])
def get_counterpicks():
    try:
        data = request.json

        # Extract draft state from the request
        draft_state = {
            'radiant': {
                'picks': data.get('radiant', {}).get('picks', []),
                'bans': data.get('radiant', {}).get('bans', [])
            },
            'dire': {
                'picks': data.get('dire', {}).get('picks', []),
                'bans': data.get('dire', {}).get('bans', [])
            },
            'currentTeam': data.get('currentTeam', 'radiant'),
            'currentAction': data.get('currentAction', 'pick')
        }

        print(f"Received draft state: {draft_state}")

        # Get ML-powered recommendations
        recommendations = get_hero_recommendations(draft_state)

        # Get draft analysis
        analysis = get_draft_win_probabilities(draft_state)

        return jsonify({
            'recommendations': recommendations,
            'analysis': analysis
        })

    except Exception as e:
        print(f"Error in counterpicks endpoint: {e}")
        return jsonify({
            'recommendations': [],
            'analysis': {
                'radiant_win_probability': 0.5,
                'dire_win_probability': 0.5,
                'analysis': 'Error getting recommendations'
            }
        }), 500

# Model status endpoint
@app.route('/api/model-status', methods=['GET'])
def model_status():
    try:
        model_exists = os.path.exists('models/dota_draft_predictor.pkl')

        training_files = [f for f in os.listdir('data') if f.startswith('training_matches_')]
        has_training_data = len(training_files) > 0

        return jsonify({
            'model_trained': model_exists,
            'has_training_data': has_training_data,
            'recommendations_available': model_exists or has_training_data
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'model_trained': False,
            'has_training_data': False
        })

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Starting Dota 2 Draft Predictor Flask App...")
    print("Available endpoints:")
    print("- GET  /: Main application")
    print("- POST /api/counterpicks: Get hero recommendations")
    print("- GET  /api/model-status: Check ML model status")

    app.run(debug=True)