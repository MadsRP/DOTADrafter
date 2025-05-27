# app.py - Fixed imports for working integration
import json
import os

from flask import Flask, render_template, request, jsonify

from data_collector import fetch_hero_data, fetch_hero_portraits

try:
    from dynamic_winrates_integration import get_hero_recommendations, get_draft_win_probabilities

    print("✅ Using dynamic_winrates_integration")
except ImportError:
    try:
        from lightweight_integration import get_hero_recommendations, get_draft_win_probabilities

        print("✅ Using lightweight_integration")
    except ImportError:
        print("❌ No integration found - creating fallback functions")


        def get_hero_recommendations(draft_data):
            return []


        def get_draft_win_probabilities(draft_data):
            return {
                'radiant_win_probability': 0.5,
                'dire_win_probability': 0.5,
                'analysis': 'No ML model available'
            }

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


# API endpoint to get counter picks - FIXED
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
        print(f"Got {len(recommendations) if recommendations else 0} recommendations")

        # Get draft analysis
        analysis = get_draft_win_probabilities(draft_state)
        print(f"Got analysis: {analysis}")

        return jsonify({
            'recommendations': recommendations,
            'analysis': analysis
        })

    except Exception as e:
        print(f"Error in counterpicks endpoint: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'recommendations': [],
            'analysis': {
                'radiant_win_probability': 0.5,
                'dire_win_probability': 0.5,
                'analysis': f'Error getting recommendations: {str(e)}'
            }
        }), 500


# Model status endpoint
@app.route('/api/model-status', methods=['GET'])
def model_status():
    try:
        model_exists = os.path.exists('models/dota_draft_predictor.pkl')

        training_files = []
        if os.path.exists('data'):
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


# Add this NEW endpoint to your app.py

@app.route('/api/hero-winrates', methods=['POST'])
def get_hero_winrates():
    """
    NEW: Calculate win rates for ALL available heroes in one request
    This prevents the infinite loop issue
    """
    try:
        data = request.json

        # Extract draft state
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

        print(f"Calculating win rates for all heroes in draft state: {draft_state}")

        # Get all available heroes (not picked/banned)
        selected_hero_ids = set(
            draft_state['radiant']['picks'] + draft_state['radiant']['bans'] +
            draft_state['dire']['picks'] + draft_state['dire']['bans']
        )

        # Assume heroes 1-150 (adjust based on your data)
        all_hero_ids = list(range(1, 151))
        available_heroes = [hid for hid in all_hero_ids if hid not in selected_hero_ids]

        print(f"Calculating win rates for {len(available_heroes)} available heroes...")

        # Calculate win rates for all available heroes
        hero_winrates = {}
        current_team = draft_state['currentTeam']

        # Get baseline probability (current draft state)
        try:
            baseline_analysis = get_draft_win_probabilities(draft_state)
            baseline_prob = (baseline_analysis['radiant_win_probability'] if current_team == 'radiant'
                             else baseline_analysis['dire_win_probability'])
        except:
            baseline_prob = 0.5

        # Calculate win rate for each available hero
        for hero_id in available_heroes:
            try:
                # Create test draft state with this hero added
                test_draft = {
                    'radiant': {
                        'picks': draft_state['radiant']['picks'].copy(),
                        'bans': draft_state['radiant']['bans'].copy()
                    },
                    'dire': {
                        'picks': draft_state['dire']['picks'].copy(),
                        'bans': draft_state['dire']['bans'].copy()
                    },
                    'currentTeam': current_team,
                    'currentAction': 'pick'
                }

                # Add hero to current team
                if current_team == 'radiant':
                    test_draft['radiant']['picks'].append(hero_id)
                else:
                    test_draft['dire']['picks'].append(hero_id)

                # Get ML prediction with this hero
                prediction = get_draft_win_probabilities(test_draft)
                win_prob = (prediction['radiant_win_probability'] if current_team == 'radiant'
                            else prediction['dire_win_probability'])

                # Convert to percentage and calibrate
                win_rate = win_prob * 100

                # Optional: Calibrate relative to baseline
                improvement = win_prob - baseline_prob
                calibrated_rate = 50 + (improvement * 100)  # Center around 50%
                final_rate = max(25, min(75, calibrated_rate))

                hero_winrates[hero_id] = round(final_rate, 1)

            except Exception as e:
                print(f"Error calculating win rate for hero {hero_id}: {e}")
                hero_winrates[hero_id] = 50.0  # Fallback

        # Set unavailable heroes to 0
        for hero_id in selected_hero_ids:
            hero_winrates[hero_id] = 0.0

        print(f"✅ Calculated win rates for {len(hero_winrates)} heroes")

        return jsonify({
            'hero_winrates': hero_winrates,
            'baseline_probability': baseline_prob,
            'available_heroes': len(available_heroes)
        })

    except Exception as e:
        print(f"❌ Error in hero-winrates endpoint: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'error': str(e),
            'hero_winrates': {}
        }), 500


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Starting Dota 2 Draft Predictor Flask App...")
    print("Available endpoints:")
    print("- GET  /: Main application")
    print("- POST /api/counterpicks: Get hero recommendations")
    print("- GET  /api/model-status: Check ML model status")

    app.run(debug=True)
