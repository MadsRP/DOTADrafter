# data_training.py - Train ML model on already collected STRATZ data
import json
import os

from ml_model import DotaDraftPredictorLight


def load_existing_training_data():
    """Load any existing training data from the data folder"""
    if not os.path.exists('data'):
        print("❌ No data folder found. Please run match_collector.py first.")
        return []

    all_training_data = []

    # Look for training data files created by match_collector
    for filename in os.listdir('data'):
        if filename.startswith('training_matches_') and filename.endswith('.json'):
            filepath = os.path.join('data', filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_training_data.extend(data)
                    print(f"📂 Loaded {len(data)} samples from {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

    if all_training_data:
        # Remove duplicates based on match_id
        seen_ids = set()
        unique_data = []
        for sample in all_training_data:
            match_id = sample.get('match_id')
            if match_id and match_id not in seen_ids:
                seen_ids.add(match_id)
                unique_data.append(sample)

        print(f"📊 Total unique training samples: {len(unique_data)}")
        return unique_data
    else:
        print("❌ No training data found in data folder.")
        print("💡 Please run match_collector.py first to collect data.")
        return []


def validate_training_data(training_data):
    """Validate that training data is suitable for ML training"""
    print("🔍 Validating training data...")

    valid_matches = []

    for match in training_data:
        # Check required fields
        if not all(key in match for key in ['radiant_picks', 'dire_picks', 'radiant_won']):
            continue

        # Check for reasonable number of picks
        radiant_picks = match.get('radiant_picks', [])
        dire_picks = match.get('dire_picks', [])

        if len(radiant_picks) < 5 or len(dire_picks) < 5:
            continue

        # Check winner data is valid
        radiant_won = match.get('radiant_won')
        if radiant_won not in [0, 1]:
            continue

        valid_matches.append(match)

    print(f"✅ Validated {len(valid_matches)}/{len(training_data)} matches")
    return valid_matches


def analyze_hero_frequency(training_data, min_appearances=15):
    """Analyze hero frequency in training data"""
    print(f"\n🦸 HERO FREQUENCY ANALYSIS")
    print(f"=" * 50)

    if not training_data:
        print("No training data to analyze")
        return {}

    # Count hero appearances
    hero_counts = {}
    hero_win_counts = {}  # Track wins for each hero

    for match in training_data:
        radiant_won = match.get('radiant_won', 0)

        # Count picks (most important)
        for hero_id in match.get('radiant_picks', []):
            if hero_id is not None:
                hero_counts[hero_id] = hero_counts.get(hero_id, 0) + 1
                if radiant_won:
                    hero_win_counts[hero_id] = hero_win_counts.get(hero_id, 0) + 1

        for hero_id in match.get('dire_picks', []):
            if hero_id is not None:
                hero_counts[hero_id] = hero_counts.get(hero_id, 0) + 1
                if not radiant_won:
                    hero_win_counts[hero_id] = hero_win_counts.get(hero_id, 0) + 1

        # Also count bans (less weight but still relevant)
        for hero_id in match.get('radiant_bans', []) + match.get('dire_bans', []):
            if hero_id is not None:
                hero_counts[hero_id] = hero_counts.get(hero_id, 0) + 0.5  # Half weight for bans

    # Load hero names
    hero_names = {}
    try:
        import json
        with open('data/heroes.json', 'r') as f:
            heroes = json.load(f)
            for hero in heroes:
                hero_id = hero.get('id')
                if hero_id:
                    hero_names[hero_id] = hero.get('displayName', f'Hero {hero_id}')
    except:
        print("⚠️ Could not load hero names, using IDs")

    # Separate heroes by frequency
    frequent_heroes = {}
    rare_heroes = {}
    never_seen = []

    for hero_id, count in hero_counts.items():
        if count >= min_appearances:
            frequent_heroes[hero_id] = {
                'count': count,
                'name': hero_names.get(hero_id, f'Hero {hero_id}'),
                'win_rate': (hero_win_counts.get(hero_id, 0) / count * 100) if count > 0 else 0
            }
        else:
            rare_heroes[hero_id] = {
                'count': count,
                'name': hero_names.get(hero_id, f'Hero {hero_id}'),
                'win_rate': (hero_win_counts.get(hero_id, 0) / count * 100) if count > 0 else 0
            }

    # Find heroes never seen (assuming heroes 1-150)
    all_possible_heroes = set(range(1, 151))
    seen_heroes = set(hero_counts.keys())
    never_seen_ids = all_possible_heroes - seen_heroes

    for hero_id in never_seen_ids:
        never_seen.append({
            'id': hero_id,
            'name': hero_names.get(hero_id, f'Hero {hero_id}')
        })

    # Print analysis
    print(f"Total matches analyzed: {len(training_data)}")
    print(f"Minimum appearances threshold: {min_appearances}")
    print(f"Total unique heroes seen: {len(hero_counts)}")

    print(f"\n✅ RELIABLE HEROES ({len(frequent_heroes)} heroes with {min_appearances}+ appearances):")
    if frequent_heroes:
        # Sort by frequency
        sorted_frequent = sorted(frequent_heroes.items(), key=lambda x: x[1]['count'], reverse=True)
        for i, (hero_id, data) in enumerate(sorted_frequent[:10]):  # Show top 10
            print(
                f"   {i + 1:2d}. {data['name']:<20} - {data['count']:5.1f} appearances, {data['win_rate']:5.1f}% win rate")
        if len(sorted_frequent) > 10:
            print(f"   ... and {len(sorted_frequent) - 10} more reliable heroes")

    print(f"\n⚠️  RARE HEROES ({len(rare_heroes)} heroes with <{min_appearances} appearances):")
    if rare_heroes:
        # Sort by frequency (lowest first)
        sorted_rare = sorted(rare_heroes.items(), key=lambda x: x[1]['count'])
        for hero_id, data in sorted_rare:
            print(f"   • {data['name']:<20} - {data['count']:5.1f} appearances, {data['win_rate']:5.1f}% win rate")

    # Summary statistics
    total_heroes_in_game = len(hero_names) if hero_names else 150
    coverage = len(hero_counts) / total_heroes_in_game * 100
    reliable_coverage = len(frequent_heroes) / total_heroes_in_game * 100

    print(f"\n📊 COVERAGE SUMMARY:")
    print(f"   Hero pool coverage: {coverage:.1f}% ({len(hero_counts)}/{total_heroes_in_game} heroes)")
    print(f"   Reliable predictions: {reliable_coverage:.1f}% ({len(frequent_heroes)}/{total_heroes_in_game} heroes)")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if reliable_coverage < 60:
        print(f"   • Consider collecting more data - only {reliable_coverage:.1f}% of heroes have reliable data")
    if len(never_seen) > 50:
        print(f"   • {len(never_seen)} heroes never seen - they'll have default predictions")
    if len(rare_heroes) > 30:
        print(f"   • {len(rare_heroes)} heroes are rarely picked - predictions may be less accurate")

    if reliable_coverage >= 70:
        print(f"   ✅ Good hero coverage! {reliable_coverage:.1f}% of heroes have reliable training data")

    # Simple yes/no summary - check against actual heroes that exist
    actual_hero_count = len(hero_names) if hero_names else len(hero_counts)
    all_real_heroes_reliable = len(frequent_heroes) == actual_hero_count
    print(
        f"\n🎯 All heroes appeared in at least {min_appearances} games: {'✅ YES' if all_real_heroes_reliable else '❌ NO'}")
    print(f"    ({len(frequent_heroes)}/{actual_hero_count} heroes have sufficient data)")

    print(f"=" * 50)

    return {
        'frequent_heroes': frequent_heroes,
        'rare_heroes': rare_heroes,
        'never_seen': never_seen,
        'coverage': coverage,
        'reliable_coverage': reliable_coverage
    }


# Updated analyze_training_data function - add this to your data_training.py
def analyze_training_data(training_data):
    """Analyze the training data quality - UPDATED VERSION"""
    print(f"\n📈 TRAINING DATA ANALYSIS")
    print(f"=" * 50)

    if not training_data:
        print("No training data to analyze")
        return

    print(f"Total matches: {len(training_data)}")

    # Win rate analysis
    radiant_wins = sum(1 for match in training_data if match.get('radiant_won') == 1)
    radiant_win_rate = radiant_wins / len(training_data) * 100
    print(f"Radiant win rate: {radiant_win_rate:.1f}% ({radiant_wins}/{len(training_data)})")

    # Pick/ban analysis
    avg_radiant_picks = sum(len(match.get('radiant_picks', [])) for match in training_data) / len(training_data)
    avg_dire_picks = sum(len(match.get('dire_picks', [])) for match in training_data) / len(training_data)
    avg_radiant_bans = sum(
        len([b for b in match.get('radiant_bans', []) if b is not None]) for match in training_data) / len(
        training_data)
    avg_dire_bans = sum(len([b for b in match.get('dire_bans', []) if b is not None]) for match in training_data) / len(
        training_data)

    print(f"Average picks per team: Radiant {avg_radiant_picks:.1f}, Dire {avg_dire_picks:.1f}")
    print(f"Average bans per team: Radiant {avg_radiant_bans:.1f}, Dire {avg_dire_bans:.1f}")

    # Hero diversity
    all_heroes = set()
    for match in training_data:
        all_heroes.update([h for h in match.get('radiant_picks', []) if h is not None])
        all_heroes.update([h for h in match.get('dire_picks', []) if h is not None])
        all_heroes.update([h for h in match.get('radiant_bans', []) if h is not None])
        all_heroes.update([h for h in match.get('dire_bans', []) if h is not None])

    print(f"Unique heroes in dataset: {len(all_heroes)}")

    # Data sources
    sources = {}
    for match in training_data:
        source = match.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    print(f"Data sources: {sources}")

    # Duration analysis
    durations = [match.get('duration_seconds', 0) for match in training_data if match.get('duration_seconds', 0) > 0]
    if durations:
        avg_duration = sum(durations) / len(durations) / 60  # Convert to minutes
        print(f"Average match duration: {avg_duration:.1f} minutes")

    print(f"=" * 50)

    # NEW: Add hero frequency analysis
    hero_analysis = analyze_hero_frequency(training_data, min_appearances=15)

    return hero_analysis


def main():
    """Main function to train ML model on already collected data"""
    print("🎯 DOTA 2 DRAFT PREDICTOR TRAINING")
    print("=" * 50)
    print("📋 Training on data collected by match_collector.py")

    # Step 1: Load existing data
    print("\n1️⃣ Loading existing training data...")
    training_data = load_existing_training_data()

    if not training_data:
        print("\n❌ No training data found!")
        print("🔧 Please run the following steps:")
        print("   1. Run: python match_collector.py")
        print("   2. Wait for data collection to complete")
        print("   3. Run: python data_training.py")
        return

    # Step 2: Validate the data
    print("\n2️⃣ Validating training data...")
    validated_data = validate_training_data(training_data)

    if len(validated_data) < 200:
        print(f"❌ Not enough valid training data ({len(validated_data)} matches)")
        print("💡 Need at least 200 matches for training. Please collect more data.")
        return

    # Step 3: Analyze the data
    print("\n3️⃣ Analyzing training data...")
    analyze_training_data(validated_data)

    # Step 4: Train the Deep Neural Network
    print(f"\n4️⃣ Training Deep Neural Network on {len(validated_data)} matches...")

    # Initialize and train the model
    predictor = DotaDraftPredictorLight()

    try:
        print("🧠 Training Deep Neural Network...")
        results = predictor.train(validated_data)  # No model_type parameter needed
        print(f"✅ Deep Neural Network Results:")
        print(f"   Training Accuracy: {results['train_accuracy']:.3f}")
        print(f"   Test Accuracy: {results['test_accuracy']:.3f}")
        print(f"   Overfitting Gap: {abs(results['train_accuracy'] - results['test_accuracy']):.3f}")

        # Save the model
        os.makedirs('models', exist_ok=True)
        predictor.save_model('models/dota_draft_predictor')

    except Exception as e:
        print(f"❌ Deep Neural Network training failed: {e}")
        print("💡 Check your data quality or reduce model complexity further")
        return

    # Step 5: Test the model
    print(f"\n5️⃣ Testing trained model...")
    test_sample = validated_data[0]

    try:
        prediction = predictor.predict_win_probability(
            test_sample['radiant_picks'],
            test_sample['dire_picks'],
            test_sample.get('radiant_bans', []),
            test_sample.get('dire_bans', [])
        )

        print(f"📊 Test Prediction:")
        print(f"   Match ID: {test_sample.get('match_id', 'unknown')}")
        print(f"   Predicted Radiant Win Probability: {prediction['radiant_win_probability']:.3f}")
        print(f"   Actual Result: {'Radiant Won' if test_sample['radiant_won'] else 'Dire Won'}")

        # Get hero recommendations
        recommendations = predictor.recommend_heroes(
            'radiant',
            test_sample['radiant_picks'][:3],  # First 3 picks
            test_sample['dire_picks'][:3],
            test_sample.get('radiant_bans', [])[:2],  # First 2 bans
            test_sample.get('dire_bans', [])[:2],
            top_k=5
        )

        print(f"\n🎯 Sample Recommendations for Radiant:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['name']}: {rec['winRate']}% win rate")

    except Exception as e:
        print(f"❌ Error testing model: {e}")

    # Final summary
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"=" * 50)
    print(f"✅ Trained Deep Neural Network on {len(validated_data)} real STRATZ matches")
    print(f"✅ Model saved to models/dota_draft_predictor")
    print(f"✅ Training Accuracy: {results['train_accuracy']:.3f}")
    print(f"✅ Test Accuracy: {results['test_accuracy']:.3f}")

    # Provide guidance based on accuracy
    test_acc = results['test_accuracy']
    gap = abs(results['train_accuracy'] - results['test_accuracy'])

    if test_acc >= 0.60 and gap < 0.15:
        print(f"🌟 Excellent DNN performance! Good generalization.")
    elif test_acc >= 0.55:
        print(f"✅ Good DNN performance. Ready to use!")
    else:
        print(f"⚠️ DNN accuracy could be better. The simplified model reduced overfitting.")

    if gap < 0.1:
        print(f"✅ Low overfitting - model generalizes well!")
    elif gap < 0.2:
        print(f"⚠️ Moderate overfitting - acceptable for Dota prediction")
    else:
        print(f"❌ High overfitting - consider even simpler architecture")

    print(f"\n🚀 Next steps:")
    print(f"1. Run: python app.py")
    print(f"2. Your Deep Neural Network is trained on real Dota 2 data!")
    print(f"3. Win rate predictions are based on actual match outcomes")


if __name__ == "__main__":
    main()
