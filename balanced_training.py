# balanced_training.py - Not too simple, not too complex, just right
import os
import json
import random
from lightweight_ml_model import DotaDraftPredictorLight

def create_balanced_training_data():
    """Create training data that's better than simple but not over-complicated"""
    print("📊 Creating balanced training data...")

    # Load heroes
    if not os.path.exists('data/heroes.json'):
        print("❌ No hero data found!")
        return None

    with open('data/heroes.json', 'r') as f:
        heroes = json.load(f)

    hero_ids = [hero['id'] for hero in heroes]
    hero_dict = {hero['id']: hero for hero in heroes}
    print(f"Using {len(hero_ids)} heroes")

    training_data = []

    # Generate 1200 balanced training matches
    for i in range(1200):
        # Create realistic draft
        draft = create_balanced_draft(hero_ids, hero_dict, i)
        if draft:
            training_data.append(draft)

    # Save training data
    os.makedirs('data', exist_ok=True)
    filename = 'data/training_matches_balanced.json'
    with open(filename, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"✅ Created {len(training_data)} balanced training samples")

    return training_data

def create_balanced_draft(hero_ids, hero_dict, match_id):
    """Create a realistic but not over-complicated draft"""

    # Select 10 unique heroes for picks
    selected_heroes = random.sample(hero_ids, 10)
    radiant_picks = selected_heroes[:5]
    dire_picks = selected_heroes[5:]

    # Select heroes for bans (realistic number)
    available_for_bans = [h for h in hero_ids if h not in selected_heroes]
    num_bans = random.randint(6, 12)  # Realistic ban count
    banned_heroes = random.sample(available_for_bans, min(num_bans, len(available_for_bans)))

    # Split bans between teams
    mid_point = len(banned_heroes) // 2
    radiant_bans = banned_heroes[:mid_point]
    dire_bans = banned_heroes[mid_point:]

    # Calculate winner with BALANCED logic (not too simple, not too complex)
    radiant_score = calculate_balanced_team_score(radiant_picks, dire_picks, hero_dict)
    dire_score = calculate_balanced_team_score(dire_picks, radiant_picks, hero_dict)

    # Add realistic variance (important for learning)
    radiant_score += random.uniform(-8, 8)
    dire_score += random.uniform(-8, 8)

    radiant_won = 1 if radiant_score > dire_score else 0

    return {
        "match_id": f"balanced_{match_id}",
        "radiant_picks": sorted(radiant_picks),
        "dire_picks": sorted(dire_picks),
        "radiant_bans": sorted(radiant_bans),
        "dire_bans": sorted(dire_bans),
        "radiant_won": radiant_won,
        "duration_seconds": random.randint(1500, 3600),
        "average_rank": random.randint(70, 90),
        "source": "balanced_mock"
    }

def calculate_balanced_team_score(team_picks, enemy_picks, hero_dict):
    """Calculate team score with balanced complexity"""
    base_score = 50

    # 1. Attribute balance (simple but effective)
    attr_counts = {'str': 0, 'agi': 0, 'int': 0, 'all': 0}
    for hero_id in team_picks:
        attr = hero_dict.get(hero_id, {}).get('primaryAttribute', 'all')
        attr_counts[attr] += 1

    # Penalty for too much of one attribute
    max_attr = max(attr_counts.values())
    balance_penalty = max(0, (max_attr - 2) * 3)

    # 2. Simple "counter" logic based on attributes
    counter_bonus = 0
    for my_hero in team_picks:
        my_attr = hero_dict.get(my_hero, {}).get('primaryAttribute', 'all')
        for enemy_hero in enemy_picks:
            enemy_attr = hero_dict.get(enemy_hero, {}).get('primaryAttribute', 'all')

            # Simple rock-paper-scissors style counters
            if ((my_attr == 'str' and enemy_attr == 'agi') or
                    (my_attr == 'agi' and enemy_attr == 'int') or
                    (my_attr == 'int' and enemy_attr == 'str')):
                counter_bonus += 1

    # 3. Team size bonus (full team is stronger)
    size_bonus = len(team_picks) * 2

    # 4. Small random factor for variety
    variety_factor = random.uniform(-3, 3)

    total_score = base_score - balance_penalty + counter_bonus + size_bonus + variety_factor
    return total_score

def main():
    print("🎯 Balanced ML Training (Just Right Complexity)")
    print("=" * 55)

    # Create balanced training data
    training_data = create_balanced_training_data()

    if not training_data:
        return

    # Train with balanced approach
    print("\n🧠 Training balanced model...")
    predictor = DotaDraftPredictorLight()

    try:
        # Try neural network with good parameters
        results = predictor.train(training_data, model_type='neural_network')
        print(f"✅ Neural Network trained! Test Accuracy: {results['test_accuracy']:.3f}")

        # If accuracy is still low, try random forest
        if results['test_accuracy'] < 0.52:
            print("🔄 Neural network accuracy low, trying Random Forest...")
            predictor = DotaDraftPredictorLight()  # Fresh instance
            results = predictor.train(training_data, model_type='random_forest')
            print(f"✅ Random Forest trained! Test Accuracy: {results['test_accuracy']:.3f}")

    except Exception as e:
        print(f"Neural network failed: {e}")
        print("🔄 Using Random Forest...")
        results = predictor.train(training_data, model_type='random_forest')
        print(f"✅ Random Forest trained! Test Accuracy: {results['test_accuracy']:.3f}")

    # Save the model
    os.makedirs('models', exist_ok=True)
    predictor.save_model('models/dota_draft_predictor')

    # Test predictions
    print("\n🧪 Testing balanced model...")
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

    # Test recommendations variety
    recommendations = predictor.recommend_heroes(
        'radiant',
        test_sample['radiant_picks'][:2],
        test_sample['dire_picks'][:2],
        test_sample['radiant_bans'][:3],
        test_sample['dire_bans'][:3],
        top_k=5
    )

    print(f"\nSample recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['name']}: {rec['winRate']}% win rate")

    # Show expected results
    if results['test_accuracy'] >= 0.55:
        print(f"\n🎉 Excellent! {results['test_accuracy']:.1%} accuracy is very good!")
    elif results['test_accuracy'] >= 0.52:
        print(f"\n✅ Good! {results['test_accuracy']:.1%} accuracy is solid for this problem.")
    else:
        print(f"\n📊 {results['test_accuracy']:.1%} accuracy - acceptable for draft prediction complexity.")

    print(f"\n🎯 Next steps:")
    print(f"1. Run: python app.py")
    print(f"2. Test your web interface")
    print(f"3. Model should give better recommendations than before!")

if __name__ == "__main__":
    main()