# app.py - Main Flask application
import json
import os

from flask import Flask, render_template, request, jsonify

from data_collector import fetch_hero_data, fetch_hero_portraits

app = Flask(__name__)


# Load or fetch hero data
def get_hero_data():
    # Check if we have cached hero data
    if os.path.exists('data/heroes.json'):
        with open('data/heroes.json', 'r') as f:
            return json.load(f)
    else:
        # Fetch fresh hero data
        os.makedirs('data', exist_ok=True)
        heroes = fetch_hero_data()

        # Download hero portraits
        fetch_hero_portraits(heroes)
        with open('data/heroes.json', 'w') as f:
            json.dump(heroes, f)

        return heroes


# Home page route
@app.route('/')
def index():
    hero_data = get_hero_data()
    print(hero_data)
    return render_template('index.html', heroes=hero_data)


# API endpoint to get counter picks
@app.route('/api/counterpicks', methods=['POST'])
def get_counterpicks():
    data = request.json
    selected_heroes = data.get('selected', [])
    banned_heroes = data.get('banned', [])

    # This is just a placeholder - we'll implement the actual counter-pick logic
    # based on our collected data and analysis
    return jsonify({
        'recommendations': [
            {'id': 1, 'name': 'Anti-Mage', 'winRate': 53.2, 'reasons': ['Counter to enemy cores']}
        ]
    })


if __name__ == '__main__':
    app.run(debug=True)
