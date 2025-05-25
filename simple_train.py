#!/usr/bin/env python3
# simple_train.py - Train ML model without TensorFlow

import os
import json
import random
from lightweight_ml_model import DotaDraftPredictorLight

def create_quick_training_data():
    """Create training data quickly"""
    print("📊 Creating quick training data...")

    # Load heroes
    if not os.path.exists('data/heroes.json'):
        print("❌ No hero data found! Run 'python data_collector.py' first.")
        return None

    with open('data/heroes.json', 'r') as f:
        heroes = json.load(f)

    hero_ids = [hero['id'] for hero in heroes]
    print(f"Using {len(hero_ids)} heroes")

    training_data = []

    # Generate training matches
    for i in range(800):
        # Select 10 unique heroes for picks
        selected_heroes = random.sample(hero_ids, 10)
        radiant_picks = selected_heroes[:5]
        dire_picks = selected_heroes[5:]

        # Select heroes for bans
        available_for_bans = [h for h in hero_ids if h not in selected_heroes]
        num_bans = random.randint(4, 12)
        if len(available_for_bans) >= num_bans:
            banned_heroes = random.sample(available_for_bans, num_bans)
        else:
            banned_heroes = available_for_bans

        # Split bans between teams
        mid_point = len(banned_heroes) // 2
        radiant_bans = banned_heroes[:mid_point]
        dire_bans = banned_heroes[mid_point:]

        # Determine winner with some logic
        radiant_score = calculate_team_strength(radiant_picks, heroes)
        dire_score = calculate_team_strength(dire_picks, heroes)

        # Add randomness
        radiant_score += random.uniform(-10, 10)
        dire_score += random.uniform(-10, 10)

        radiant_won = 1 if radiant_score > dire_score else 0

        training_sample = {
            "match_id": f"quick_{i}",
            "radiant_picks": sorted(radiant_picks),
            "dire_picks": sorted(dire_picks),
            "radiant_bans": sorted(radiant_bans),
            "dire_bans": sorted(dire_bans),
            "radiant_won": radiant_won,
            "duration_seconds": random.randint(1200, 3600),
            "average_rank": random.randint(70, 95),
            "source": "quick_mock"
        }

        training_data.append(training_sample)

    # Save training data
    os.makedirs('data', exist_ok=True)
    filename = 'data/training_matches_quick.json'
    with open(filename, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"✅ Created {len(training_data)} training samples")
    print(f"Saved to {filename}")

    return training_data

def calculate_team_strength(hero_picks, heroes):
    """Calculate team strength for realistic win prediction"""
    hero_dict = {hero['id']: hero for hero in heroes}

    # Count attributes
    str_count = sum(1 for hid in hero_picks if hero_dict.get(hid, {}).get('primaryAttribute') == 'str')
    agi_count = sum(1 for hid in hero_picks if hero_dict.get(hid, {}).get('primaryAttribute') == 'agi')
    int_count = sum(1 for hid in hero_picks if hero_dict.get(hid, {}).get('primaryAttribute') == 'int')

    # Balanced teams get bonus
    max_attr = max(str_count, agi_count, int_count)
    balance_penalty = max_attr - 2  # Penalty for too many of one attribute

    # Base score
    base_score = 50 - balance_penalty * 3

    # Add some hero-specific variance
    for hero_id in hero_picks:
        base_score += random.uniform(-2, 2)

    return base_score

def main():
    print("🚀 Quick ML Training (No TensorFlow Required)")
    print("=" * 50)

    # Step 1: Create training data
    print("\n📊 Step 1: Creating training data...")
    training_data = create_quick_training_data()

    if not training_data:
        print("❌ Failed to create training data")
        return

    # Step 2: Train model
    print("\n🧠 Step 2: Training machine learning model...")
    predictor = DotaDraftPredictorLight()

    # Try neural network first, fallback to random forest if needed
    try:
        print("Trying Neural Network...")
        results = predictor.train(training_data, model_type='neural_network')
    except Exception as e:
        print(f"Neural network failed: {e}")
        print("Using Random Forest instead...")
        results = predictor.train(training_data, model_type='random_forest')

    print(f"✅ Model trained successfully!")
    print(f"Test Accuracy: {results['test_accuracy']:.3f}")

    # Step 3: Save model
    print("\n💾 Step 3: Saving model...")
    os.makedirs('models', exist_ok=True)
    predictor.save_model('models/dota_draft_predictor')

    # Step 4: Test model
    print("\n🧪 Step 4: Testing model...")
    test_sample = training_data[0]

    prediction = predictor.predict_win_probability(
        test_sample['radiant_picks'],
        test_sample['dire_picks'],
        test_sample['radiant_bans'],
        test_sample['dire_bans']
    )

    print(f"Test prediction:")
    print(f"  Radiant win probability: {prediction['radiant_win_probability']:.3f}")
    print(f"  Actual result: {'Radiant Won' if test_sample['radiant_won'] else 'Dire Won'}")

    # Test recommendations
    recommendations = predictor.recommend_heroes(
        'radiant',
        test_sample['radiant_picks'][:3],  # Partial draft
        test_sample['dire_picks'][:3],
        test_sample['radiant_bans'],
        test_sample['dire_bans'],
        top_k=3
    )

    print(f"\nSample recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['name']}: {rec['winRate']}% win rate")

    print("\n🎉 Training completed successfully!")
    print("\nYour model files:")
    print(f"  - models/dota_draft_predictor.pkl")
    print(f"  - models/dota_draft_predictor_scaler.pkl")

    print("\nNext steps:")
    print("1. Update your Flask app to use the lightweight model")
    print("2. Run: python app.py")
    print("3. Go to http://localhost:5000")
    print("4. Start drafting with ML recommendations!")

if __name__ == "__main__":
    main()