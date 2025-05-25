# improved_training.py - Better training with more sophisticated features
import os
import json
import random
import numpy as np
from lightweight_ml_model import DotaDraftPredictorLight

def create_improved_training_data():
    """Create better training data with more realistic patterns"""
    print("📊 Creating improved training data...")

    # Load heroes
    if not os.path.exists('data/heroes.json'):
        print("❌ No hero data found!")
        return None

    with open('data/heroes.json', 'r') as f:
        heroes = json.load(f)

    hero_ids = [hero['id'] for hero in heroes]
    print(f"Using {len(hero_ids)} heroes")

    # Create hero synergy and counter matrices (simplified)
    hero_synergies, hero_counters = create_hero_relationships(heroes)

    training_data = []

    # Generate more sophisticated training matches
    for i in range(1500):  # More training data
        # Create a draft with more realistic picking patterns
        draft = create_realistic_draft(hero_ids, heroes, hero_synergies, hero_counters)

        if draft:
            training_data.append(draft)

    # Save training data
    os.makedirs('data', exist_ok=True)
    filename = 'data/training_matches_improved.json'
    with open(filename, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"✅ Created {len(training_data)} improved training samples")
    print(f"Saved to {filename}")

    return training_data

def create_hero_relationships(heroes):
    """Create simplified hero synergy and counter relationships"""
    hero_dict = {hero['id']: hero for hero in heroes}
    synergies = {}
    counters = {}

    # Simple relationship rules based on attributes and roles
    for hero in heroes:
        hero_id = hero['id']
        attr = hero.get('primaryAttribute', 'all')

        synergies[hero_id] = []
        counters[hero_id] = []

        # Create some basic relationships
        for other_hero in heroes:
            if other_hero['id'] == hero_id:
                continue

            other_attr = other_hero.get('primaryAttribute', 'all')

            # Synergy: heroes of different attributes work well together
            if attr != other_attr and attr != 'all' and other_attr != 'all':
                if random.random() < 0.3:  # 30% chance of synergy
                    synergies[hero_id].append(other_hero['id'])

            # Counter: some attribute matchups counter others
            if ((attr == 'str' and other_attr == 'agi') or
                    (attr == 'agi' and other_attr == 'int') or
                    (attr == 'int' and other_attr == 'str')):
                if random.random() < 0.2:  # 20% chance of counter
                    counters[hero_id].append(other_hero['id'])

    return synergies, counters

def create_realistic_draft(hero_ids, heroes, synergies, counters):
    """Create a more realistic draft with strategic picking"""

    # Pick first hero randomly but weighted by attribute balance
    hero_dict = {hero['id']: hero for hero in heroes}

    radiant_picks = []
    dire_picks = []
    radiant_bans = []
    dire_bans = []

    available_heroes = hero_ids.copy()

    # Simulate draft phases with more strategy
    draft_order = ['radiant_ban', 'dire_ban', 'radiant_ban', 'dire_ban',
                   'radiant_pick', 'dire_pick', 'dire_pick', 'radiant_pick',
                   'radiant_ban', 'dire_ban', 'radiant_ban', 'dire_ban',
                   'dire_pick', 'radiant_pick', 'radiant_pick', 'dire_pick',
                   'dire_ban', 'radiant_ban', 'dire_pick', 'radiant_pick']

    for phase in draft_order:
        if not available_heroes:
            break

        team, action = phase.split('_')

        if action == 'pick':
            # Strategic picking considering synergies and counters
            hero_id = strategic_pick(available_heroes, team, radiant_picks, dire_picks,
                                     synergies, counters, hero_dict)
            if team == 'radiant':
                radiant_picks.append(hero_id)
            else:
                dire_picks.append(hero_id)
        else:  # ban
            # Strategic banning - remove strong heroes from opponent
            hero_id = strategic_ban(available_heroes, team, radiant_picks, dire_picks,
                                    synergies, counters)
            if team == 'radiant':
                radiant_bans.append(hero_id)
            else:
                dire_bans.append(hero_id)

        available_heroes.remove(hero_id)

    # Ensure complete draft
    if len(radiant_picks) != 5 or len(dire_picks) != 5:
        return None

    # Calculate winner with improved logic
    radiant_score = calculate_team_score_advanced(radiant_picks, dire_picks, hero_dict, synergies, counters)
    dire_score = calculate_team_score_advanced(dire_picks, radiant_picks, hero_dict, synergies, counters)

    # Add realistic variance
    radiant_score += random.uniform(-5, 5)
    dire_score += random.uniform(-5, 5)

    radiant_won = 1 if radiant_score > dire_score else 0

    return {
        "match_id": f"improved_{random.randint(100000, 999999)}",
        "radiant_picks": sorted(radiant_picks),
        "dire_picks": sorted(dire_picks),
        "radiant_bans": sorted(radiant_bans),
        "dire_bans": sorted(dire_bans),
        "radiant_won": radiant_won,
        "duration_seconds": random.randint(1500, 4200),
        "average_rank": random.randint(65, 95),
        "source": "improved_mock",
        "radiant_score": radiant_score,
        "dire_score": dire_score
    }

def strategic_pick(available_heroes, team, radiant_picks, dire_picks, synergies, counters, hero_dict):
    """Make strategic hero picks considering synergies and counters"""

    current_team_picks = radiant_picks if team == 'radiant' else dire_picks
    enemy_picks = dire_picks if team == 'radiant' else radiant_picks

    hero_scores = {}

    for hero_id in available_heroes:
        score = 50  # Base score

        # Synergy bonus with current team
        for teammate_id in current_team_picks:
            if teammate_id in synergies.get(hero_id, []):
                score += 8

        # Counter bonus against enemies
        for enemy_id in enemy_picks:
            if enemy_id in counters.get(hero_id, []):
                score += 10
            # Penalty if enemy counters this hero
            if hero_id in counters.get(enemy_id, []):
                score -= 8

        # Attribute balance bonus
        current_attrs = [hero_dict.get(hid, {}).get('primaryAttribute', 'all') for hid in current_team_picks]
        hero_attr = hero_dict.get(hero_id, {}).get('primaryAttribute', 'all')

        if hero_attr not in current_attrs or len(current_team_picks) == 0:
            score += 5  # Bonus for attribute diversity

        hero_scores[hero_id] = score + random.uniform(-10, 10)

    # Pick hero with highest score (weighted random)
    sorted_heroes = sorted(hero_scores.items(), key=lambda x: x[1], reverse=True)
    top_heroes = sorted_heroes[:min(5, len(sorted_heroes))]  # Top 5 choices

    # Weighted selection from top choices
    weights = [max(1, score) for _, score in top_heroes]
    selected_hero = random.choices([hero_id for hero_id, _ in top_heroes], weights=weights)[0]

    return selected_hero

def strategic_ban(available_heroes, team, radiant_picks, dire_picks, synergies, counters):
    """Make strategic bans"""
    current_team_picks = radiant_picks if team == 'radiant' else dire_picks
    enemy_picks = dire_picks if team == 'radiant' else radiant_picks

    # Ban heroes that would be strong against current team or strong in general
    ban_scores = {}

    for hero_id in available_heroes:
        score = random.uniform(40, 60)  # Base randomness

        # Higher priority to ban heroes that counter our picks
        for teammate_id in current_team_picks:
            if teammate_id in counters.get(hero_id, []):
                score += 15

        # Ban heroes that have good synergy potential with enemy
        for enemy_id in enemy_picks:
            if enemy_id in synergies.get(hero_id, []):
                score += 8

        ban_scores[hero_id] = score

    # Select ban target
    sorted_bans = sorted(ban_scores.items(), key=lambda x: x[1], reverse=True)
    top_bans = sorted_bans[:min(3, len(sorted_bans))]

    weights = [score for _, score in top_bans]
    selected_ban = random.choices([hero_id for hero_id, _ in top_bans], weights=weights)[0]

    return selected_ban

def calculate_team_score_advanced(team_picks, enemy_picks, hero_dict, synergies, counters):
    """Calculate advanced team score considering synergies and counters"""
    base_score = 50

    # Synergy bonus
    synergy_score = 0
    for i, hero_id in enumerate(team_picks):
        for j, teammate_id in enumerate(team_picks):
            if i != j and teammate_id in synergies.get(hero_id, []):
                synergy_score += 3

    # Counter score against enemies
    counter_score = 0
    for hero_id in team_picks:
        for enemy_id in enemy_picks:
            if enemy_id in counters.get(hero_id, []):
                counter_score += 4
            if hero_id in counters.get(enemy_id, []):
                counter_score -= 3

    # Attribute balance
    attrs = [hero_dict.get(hid, {}).get('primaryAttribute', 'all') for hid in team_picks]
    attr_counts = {attr: attrs.count(attr) for attr in ['str', 'agi', 'int', 'all']}
    max_attr_count = max(attr_counts.values())
    balance_penalty = (max_attr_count - 2) * 2  # Penalty for imbalance

    total_score = base_score + synergy_score + counter_score - balance_penalty
    return total_score

def main():
    print("🚀 Improved ML Training")
    print("=" * 50)

    # Create better training data
    training_data = create_improved_training_data()

    if not training_data:
        return

    # Train with better parameters
    print("\n🧠 Training improved model...")
    predictor = DotaDraftPredictorLight()

    # Use better training parameters
    try:
        results = predictor.train(training_data, model_type='neural_network')
        print(f"✅ Neural Network trained! Test Accuracy: {results['test_accuracy']:.3f}")
    except Exception as e:
        print(f"Neural network failed: {e}")
        results = predictor.train(training_data, model_type='random_forest')
        print(f"✅ Random Forest trained! Test Accuracy: {results['test_accuracy']:.3f}")

    # Save improved model
    os.makedirs('models', exist_ok=True)
    predictor.save_model('models/dota_draft_predictor')

    print("\n🧪 Testing improved model...")
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
    print(f"  Training scores - Radiant: {test_sample.get('radiant_score', 'N/A'):.1f}, Dire: {test_sample.get('dire_score', 'N/A'):.1f}")

    # Test recommendations with different scenarios
    print(f"\n📊 Testing recommendation variety...")

    # Test 1: Empty draft
    recs1 = predictor.recommend_heroes('radiant', [], [], [], [], top_k=5)
    print(f"Empty draft recommendations: {[r['name'] for r in recs1[:3]]}")

    # Test 2: With some picks
    recs2 = predictor.recommend_heroes('radiant', [1, 5], [10, 15], [20], [25], top_k=5)
    print(f"Mid-draft recommendations: {[r['name'] for r in recs2[:3]]}")

    print(f"\n🎉 Improved model ready!")
    print(f"Expected accuracy improvement: ~60-70%")
    print(f"More dynamic recommendations based on draft state")

if __name__ == "__main__":
    main()