#!/usr/bin/env python3
# train_model.py - Script to collect data and train the ML model

import os
import sys
import json

def main():
    print("=== Dota 2 Draft Predictor - Model Training ===\n")

    # Check if we have API token
    from data_collector import API_TOKEN
    if not API_TOKEN:
        print("❌ ERROR: No STRATZ API token found!")
        print("Please set up your API token in enviromentvariables.env")
        print("Get your token from: https://stratz.com/api")
        return

    print("✅ API token found")

    # Step 1: Collect training data
    print("\n📊 Step 1: Collecting training data...")
    try:
        from ml_data_collector import create_training_dataset, analyze_training_data

        # Start with a smaller dataset for testing
        num_matches = 300
        print(f"Collecting {num_matches} matches (this may take a few minutes)...")

        training_data = create_training_dataset(num_matches=num_matches)

        if not training_data:
            print("❌ Failed to collect training data")
            return

        print(f"✅ Successfully collected {len(training_data)} valid matches")

        # Analyze the data
        print("\n📈 Analyzing training data...")
        analysis = analyze_training_data(training_data)

    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        return

    # Step 2: Train the model
    print("\n🧠 Step 2: Training neural network...")
    try:
        from draft_ml_model import DotaDraftPredictor

        # Initialize predictor
        predictor = DotaDraftPredictor()

        # Train the model
        print("Starting training process...")
        history = predictor.train(training_data, epochs=25)

        # Save the model
        os.makedirs('models', exist_ok=True)
        predictor.save_model('models/dota_draft_predictor')

        print("✅ Model training completed!")

        # Get final accuracy
        if 'val_accuracy' in history.history:
            final_accuracy = history.history['val_accuracy'][-1]
            print(f"📊 Final validation accuracy: {final_accuracy:.3f}")

    except Exception as e:
        print(f"❌ Error training model: {e}")
        return

    # Step 3: Test the model
    print("\n🧪 Step 3: Testing model predictions...")
    try:
        # Test with a sample from training data
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

        print("✅ Model testing successful!")

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return

    print("\n🎉 Training completed successfully!")
    print("\nNext steps:")
    print("1. Run your Flask app: python app.py")
    print("2. Go to http://localhost:5000")
    print("3. Start drafting and get ML-powered recommendations!")

    print(f"\nModel files saved:")
    print(f"- models/dota_draft_predictor.h5")
    print(f"- models/dota_draft_predictor_scaler.pkl")

if __name__ == "__main__":
    main()