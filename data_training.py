# real_data_training.py - Use real STRATZ data for ML training
import json
import os
import time

from lightweight_ml_model import DotaDraftPredictorLight
from match_collector import fetch_real_match_data, fetch_hero_data


def collect_training_matches(num_matches=500):
    """Collect real matches from STRATZ API for training"""
    print(f"🔍 Collecting {num_matches} real matches from STRATZ...")

    # Collect matches in batches to respect API limits
    all_matches = []
    batch_size = 100
    batches = (num_matches + batch_size - 1) // batch_size  # Ceiling division

    for batch_num in range(batches):
        remaining_matches = min(batch_size, num_matches - len(all_matches))
        print(f"📥 Fetching batch {batch_num + 1}/{batches} ({remaining_matches} matches)...")

        try:
            # Vary the rank requirement to get diverse data
            if batch_num % 3 == 0:
                min_rank = 95  # Immortal players
            elif batch_num % 3 == 1:
                min_rank = 85  # Ancient players
            else:
                min_rank = 80  # Legend+ players

            batch_matches = fetch_real_match_data(limit=remaining_matches)

            if batch_matches:
                all_matches.extend(batch_matches)
                print(f"✅ Got {len(batch_matches)} matches (Total: {len(all_matches)})")
            else:
                print("⚠️ No matches returned from API")

            # Respect API rate limits - wait between batches
            if batch_num < batches - 1:  # Don't wait after last batch
                print("⏳ Waiting 5 seconds to respect API limits...")
                time.sleep(5)

        except Exception as e:
            print(f"❌ Error fetching batch {batch_num + 1}: {e}")
            continue

    print(f"📊 Successfully collected {len(all_matches)} matches total")
    return all_matches


def convert_stratz_to_training_data(stratz_matches):
    """Convert STRATZ match format to our training data format"""
    print("🔄 Converting STRATZ data to training format...")

    training_data = []
    skipped_matches = 0

    for match in stratz_matches:
        try:
            # Extract basic match info
            match_id = match.get('id')
            radiant_won = match.get('didRadiantWin', False)

            # Initialize pick/ban lists
            radiant_picks = []
            dire_picks = []
            radiant_bans = []
            dire_bans = []

            # Process pick/ban data
            pick_bans = match.get('pickBans', [])

            for pb in pick_bans:
                hero_id = pb.get('heroId')
                is_pick = pb.get('isPick', False)
                is_radiant = pb.get('isRadiant', False)

                if not hero_id:  # Skip if no hero ID
                    continue

                if is_pick:
                    if is_radiant:
                        radiant_picks.append(hero_id)
                    else:
                        dire_picks.append(hero_id)
                else:  # is ban
                    if is_radiant:
                        radiant_bans.append(hero_id)
                    else:
                        dire_bans.append(hero_id)

            # Validate match has reasonable pick/ban data
            if len(radiant_picks) < 3 or len(dire_picks) < 3:
                skipped_matches += 1
                continue

            # Create training sample
            training_sample = {
                "match_id": str(match_id),
                "radiant_picks": radiant_picks,
                "dire_picks": dire_picks,
                "radiant_bans": radiant_bans,
                "dire_bans": dire_bans,
                "radiant_won": 1 if radiant_won else 0,
                "duration_seconds": match.get('durationSeconds', 0),
                "game_mode": match.get('gameMode', 0),
                "lobby_type": match.get('lobbyType', 0),
                "source": "real_stratz_data"
            }

            training_data.append(training_sample)

        except Exception as e:
            print(f"⚠️ Error processing match {match.get('id', 'unknown')}: {e}")
            skipped_matches += 1
            continue

    print(f"✅ Converted {len(training_data)} matches to training format")
    if skipped_matches > 0:
        print(f"⚠️ Skipped {skipped_matches} matches due to insufficient data")

    return training_data


def save_training_data(training_data, filename=None):
    """Save training data to file"""
    if filename is None:
        timestamp = int(time.time())
        filename = f'data/training_matches_real_{timestamp}.json'

    os.makedirs('data', exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"💾 Saved {len(training_data)} training samples to {filename}")
    return filename


def load_existing_real_data():
    """Load any existing real training data"""
    if not os.path.exists('data'):
        return []

    all_training_data = []

    # Look for real training data files
    for filename in os.listdir('data'):
        if filename.startswith('training_matches_real_') and filename.endswith('.json'):
            filepath = os.path.join('data', filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_training_data.extend(data)
                    print(f"📂 Loaded {len(data)} samples from {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

    # Also check for raw match files and convert them
    for filename in os.listdir('data'):
        if filename.startswith('matches_') and filename.endswith('.json'):
            filepath = os.path.join('data', filename)
            try:
                with open(filepath, 'r') as f:
                    raw_matches = json.load(f)
                    converted_data = convert_stratz_to_training_data(raw_matches)
                    all_training_data.extend(converted_data)
                    print(f"📂 Converted {len(converted_data)} samples from {filename}")
            except Exception as e:
                print(f"❌ Error converting {filename}: {e}")

    if all_training_data:
        # Remove duplicates based on match_id
        seen_ids = set()
        unique_data = []
        for sample in all_training_data:
            match_id = sample.get('match_id')
            if match_id not in seen_ids:
                seen_ids.add(match_id)
                unique_data.append(sample)

        print(f"📊 Total unique training samples: {len(unique_data)}")
        return unique_data

    return []


def analyze_training_data(training_data):
    """Analyze the training data quality"""
    print(f"\n📈 TRAINING DATA ANALYSIS")
    print(f"=" * 50)

    if not training_data:
        print("No training data to analyze")
        return

    print(f"Total matches: {len(training_data)}")

    # Win rate analysis
    radiant_wins = sum(1 for match in training_data if match['radiant_won'] == 1)
    radiant_win_rate = radiant_wins / len(training_data) * 100
    print(f"Radiant win rate: {radiant_win_rate:.1f}% ({radiant_wins}/{len(training_data)})")

    # Pick/ban analysis
    avg_radiant_picks = sum(len(match['radiant_picks']) for match in training_data) / len(training_data)
    avg_dire_picks = sum(len(match['dire_picks']) for match in training_data) / len(training_data)
    avg_radiant_bans = sum(len(match['radiant_bans']) for match in training_data) / len(training_data)
    avg_dire_bans = sum(len(match['dire_bans']) for match in training_data) / len(training_data)

    print(f"Average picks per team: Radiant {avg_radiant_picks:.1f}, Dire {avg_dire_picks:.1f}")
    print(f"Average bans per team: Radiant {avg_radiant_bans:.1f}, Dire {avg_dire_bans:.1f}")

    # Hero diversity
    all_heroes = set()
    for match in training_data:
        all_heroes.update(match['radiant_picks'])
        all_heroes.update(match['dire_picks'])
        all_heroes.update(match['radiant_bans'])
        all_heroes.update(match['dire_bans'])

    print(f"Unique heroes in dataset: {len(all_heroes)}")

    # Data sources
    sources = {}
    for match in training_data:
        source = match.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    print(f"Data sources: {sources}")
    print(f"=" * 50)


def main():
    """Main function to collect and train on real STRATZ data"""
    print("🎯 REAL STRATZ DATA TRAINING")
    print("=" * 50)

    # Step 1: Load existing data
    print("1️⃣ Loading existing training data...")
    existing_data = load_existing_real_data()

    # Step 2: Collect more data if needed
    target_matches = 150  # Reasonable target for good training

    if len(existing_data) < target_matches:
        needed_matches = target_matches - len(existing_data)
        print(f"2️⃣ Need {needed_matches} more matches to reach target of {target_matches}")

        # Ensure we have hero data
        hero_data = fetch_hero_data(save_to_file=True)

        # Collect new matches
        new_matches = collect_training_matches(needed_matches)

        if new_matches:
            # Convert to training format
            new_training_data = convert_stratz_to_training_data(new_matches)

            # Combine with existing data
            all_training_data = existing_data + new_training_data

            # Save combined data
            save_training_data(all_training_data)
        else:
            print("⚠️ No new matches collected, using existing data")
            all_training_data = existing_data
    else:
        print(f"2️⃣ Already have {len(existing_data)} matches, sufficient for training")
        all_training_data = existing_data

    # Step 3: Analyze the data
    analyze_training_data(all_training_data)

    # Step 4: Train the model
    if len(all_training_data) < 50:
        print("❌ Not enough training data (need at least 50 matches)")
        print("💡 Try running data collection again or check your API token")
        return

    print(f"\n3️⃣ Training ML model on {len(all_training_data)} real matches...")

    # Initialize and train the model
    predictor = DotaDraftPredictorLight()

    try:
        # Try neural network first
        print("🧠 Training Neural Network...")
        results = predictor.train(all_training_data, model_type='neural_network')
        print(f"✅ Neural Network Results:")
        print(f"   Training Accuracy: {results['train_accuracy']:.3f}")
        print(f"   Test Accuracy: {results['test_accuracy']:.3f}")

        # Save the model
        os.makedirs('models', exist_ok=True)
        predictor.save_model('models/dota_draft_predictor')

        # If accuracy is low, also try Random Forest
        if results['test_accuracy'] < 0.55:
            print("\n🔄 Neural network accuracy could be better, also training Random Forest...")
            rf_predictor = DotaDraftPredictorLight()
            rf_results = predictor.train(all_training_data, model_type='random_forest')
            print(f"🌲 Random Forest Results:")
            print(f"   Training Accuracy: {rf_results['train_accuracy']:.3f}")
            print(f"   Test Accuracy: {rf_results['test_accuracy']:.3f}")

            # Save the better model
            if rf_results['test_accuracy'] > results['test_accuracy']:
                rf_predictor.save_model('models/dota_draft_predictor')
                print("💾 Saved Random Forest model (better accuracy)")
            else:
                print("💾 Kept Neural Network model (better accuracy)")

    except Exception as e:
        print(f"❌ Neural Network training failed: {e}")
        print("🔄 Falling back to Random Forest...")

        try:
            predictor = DotaDraftPredictorLight()
            results = predictor.train(all_training_data, model_type='random_forest')
            print(f"🌲 Random Forest Results:")
            print(f"   Training Accuracy: {results['train_accuracy']:.3f}")
            print(f"   Test Accuracy: {results['test_accuracy']:.3f}")

            predictor.save_model('models/dota_draft_predictor')

        except Exception as e2:
            print(f"❌ Random Forest also failed: {e2}")
            return

    # Step 5: Test the model
    print(f"\n4️⃣ Testing model with real data...")
    test_sample = all_training_data[0]

    try:
        prediction = predictor.predict_win_probability(
            test_sample['radiant_picks'],
            test_sample['dire_picks'],
            test_sample['radiant_bans'],
            test_sample['dire_bans']
        )

        print(f"📊 Test Prediction:")
        print(f"   Match ID: {test_sample['match_id']}")
        print(f"   Predicted Radiant Win Probability: {prediction['radiant_win_probability']:.3f}")
        print(f"   Actual Result: {'Radiant Won' if test_sample['radiant_won'] else 'Dire Won'}")

        # Get hero recommendations
        recommendations = predictor.recommend_heroes(
            'radiant',
            test_sample['radiant_picks'][:3],  # First 3 picks
            test_sample['dire_picks'][:3],
            test_sample['radiant_bans'][:2],  # First 2 bans
            test_sample['dire_bans'][:2],
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
    print(f"✅ Trained on {len(all_training_data)} real STRATZ matches")
    print(f"✅ Model saved to models/dota_draft_predictor")
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"Training Data: {len(all_training_data)} matches")
    print(f"   Training Accuracy: {results['train_accuracy']:.3f}")
    print(f"   Test Accuracy: {results['test_accuracy']:.3f}")

    print(f"\n🚀 Next steps:")
    print(f"1. Run: python app.py")
    print(f"2. Your model now uses REAL professional Dota 2 data!")
    print(f"3. Win rate predictions are based on actual match outcomes")


if __name__ == "__main__":
    main()
