# ml_model.py - Fixed to handle None values and data issues
import json
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class DotaDraftPredictor:
    def __init__(self, max_hero_id=150):
        """
        Lightweight version using scikit-learn
        """
        self.max_hero_id = max_hero_id
        self.scaler = StandardScaler()
        self.model = None
        self.hero_data = None

    def load_hero_data(self):
        """Load hero information with proper None handling"""
        try:
            with open('data/heroes.json', 'r') as f:
                heroes = json.load(f)
                self.hero_data = {}

                for hero in heroes:
                    # Ensure hero_id is valid
                    hero_id = hero.get('id')
                    if hero_id is None or not isinstance(hero_id, int):
                        print(f"⚠️ Skipping hero with invalid ID: {hero}")
                        continue

                    self.hero_data[hero_id] = hero

                print(f"Loaded {len(self.hero_data)} heroes")
        except Exception as e:
            print(f"Could not load hero data: {e}")
            self.hero_data = {}

    def create_feature_vector(self, radiant_picks, dire_picks, radiant_bans, dire_bans):
        """Convert draft state into feature vector with proper None handling"""
        features = np.zeros(self.max_hero_id + 1)

        # Clean and validate input lists
        def clean_hero_list(hero_list):
            """Remove None values and invalid hero IDs"""
            if not hero_list:
                return []
            cleaned = []
            for hero_id in hero_list:
                if hero_id is not None and isinstance(hero_id, (int, float)):
                    hero_id = int(hero_id)
                    if 1 <= hero_id <= self.max_hero_id:
                        cleaned.append(hero_id)
                    else:
                        print(f"⚠️ Hero ID {hero_id} out of range (1-{self.max_hero_id})")
            return cleaned

        # Clean all input lists
        radiant_picks = clean_hero_list(radiant_picks)
        dire_picks = clean_hero_list(dire_picks)
        radiant_bans = clean_hero_list(radiant_bans)
        dire_bans = clean_hero_list(dire_bans)

        # Encode picks: +1 for radiant, -1 for dire
        for hero_id in radiant_picks:
            features[hero_id] = 1.0

        for hero_id in dire_picks:
            features[hero_id] = -1.0

        # Encode bans: 0.5 (neutral but unavailable)
        for hero_id in radiant_bans + dire_bans:
            if features[hero_id] == 0:  # Only if not already picked
                features[hero_id] = 0.5

        # Add composition features
        composition_features = self.calculate_composition_features(
            radiant_picks, dire_picks
        )

        # Combine features
        full_features = np.concatenate([features, composition_features])

        return full_features

    def calculate_composition_features(self, radiant_picks, dire_picks):
        """Calculate team composition features with None handling"""
        if not self.hero_data:
            return np.zeros(8)

        def get_attribute_counts(picks):
            str_count = agi_count = int_count = all_count = 0

            for hero_id in picks:
                if hero_id is None or hero_id not in self.hero_data:
                    continue

                hero = self.hero_data.get(hero_id, {})
                attr = hero.get('primaryAttribute', 'all')

                if attr == 'str':
                    str_count += 1
                elif attr == 'agi':
                    agi_count += 1
                elif attr == 'int':
                    int_count += 1
                else:
                    all_count += 1

            return [str_count, agi_count, int_count, all_count]

        radiant_attrs = get_attribute_counts(radiant_picks)
        dire_attrs = get_attribute_counts(dire_picks)

        return np.array(radiant_attrs + dire_attrs, dtype=float)

    def prepare_training_data(self, training_data):
        """Convert training data into feature matrices with validation"""
        print("Preparing training data...")

        X = []
        y = []
        skipped_matches = 0

        for i, match in enumerate(training_data):
            try:
                # Validate match data
                if not isinstance(match, dict):
                    print(f"⚠️ Match {i}: Not a dictionary")
                    skipped_matches += 1
                    continue

                # Extract required fields with defaults
                radiant_picks = match.get('radiant_picks', []) or []
                dire_picks = match.get('dire_picks', []) or []
                radiant_bans = match.get('radiant_bans', []) or []
                dire_bans = match.get('dire_bans', []) or []
                radiant_won = match.get('radiant_won')

                # Validate radiant_won
                if radiant_won is None or radiant_won not in [0, 1]:
                    print(f"⚠️ Match {i}: Invalid radiant_won value: {radiant_won}")
                    skipped_matches += 1
                    continue

                # Validate we have some picks
                if len(radiant_picks) < 3 or len(dire_picks) < 3:
                    print(f"⚠️ Match {i}: Insufficient picks (R:{len(radiant_picks)}, D:{len(dire_picks)})")
                    skipped_matches += 1
                    continue

                # Create feature vector
                features = self.create_feature_vector(
                    radiant_picks, dire_picks, radiant_bans, dire_bans
                )

                # Validate feature vector
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print(f"⚠️ Match {i}: Invalid feature vector")
                    skipped_matches += 1
                    continue

                X.append(features)
                y.append(int(radiant_won))

            except Exception as e:
                print(f"⚠️ Match {i}: Error processing - {e}")
                skipped_matches += 1
                continue

        if skipped_matches > 0:
            print(f"⚠️ Skipped {skipped_matches} matches due to data issues")

        if len(X) == 0:
            raise ValueError("No valid training samples after data cleaning!")

        X = np.array(X)
        y = np.array(y)

        print(f"Created feature matrix: {X.shape}")
        print(f"Valid training samples: {len(y)}")
        print(f"Radiant win rate in training data: {np.mean(y):.3f}")

        return X, y

    def train(self, training_data, test_size=0.2):
        print("Starting DNN training process...")

        # Load hero data
        self.load_hero_data()

        # Prepare data
        try:
            X, y = self.prepare_training_data(training_data)
        except Exception as e:
            print(f"❌ Data preparation failed: {e}")
            raise

        # Validate we have enough data
        if len(X) < 10:
            raise ValueError(f"Not enough valid training samples: {len(X)} (need at least 10)")

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}")
            print("Trying split without stratification...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training radiant win rate: {np.mean(y_train):.3f}")
        print(f"Test radiant win rate: {np.mean(y_test):.3f}")

        # Scale features
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        except Exception as e:
            print(f"❌ Feature scaling failed: {e}")
            raise

        # Create simplified Deep Neural Network
        print("🧠 Creating simplified Deep Neural Network...")

        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),  # Simple architecture to prevent overfitting
            activation='relu',
            solver='adam',
            alpha=0.01,                   # Strong regularization
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,      # 20% for validation
            n_iter_no_change=10
        )

        print(f"Network architecture: 2 hidden layers (32, 16 neurons)")
        print(f"Regularization strength: 0.01")
        print(f"Early stopping enabled with 20% validation data")

        # Train the model
        print("Training Deep Neural Network...")
        try:
            self.model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"❌ DNN training failed: {e}")
            raise

        # Evaluate
        try:
            train_predictions = self.model.predict(X_train_scaled)
            test_predictions = self.model.predict(X_test_scaled)

            train_accuracy = accuracy_score(y_train, train_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)

            print(f"\n📊 DEEP NEURAL NETWORK RESULTS:")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Overfitting Gap: {abs(train_accuracy - test_accuracy):.4f}")

            # Interpret results
            gap = abs(train_accuracy - test_accuracy)
            if gap < 0.1:
                print("✅ Excellent generalization! Low overfitting.")
            elif gap < 0.2:
                print("⚠️ Moderate overfitting, but acceptable for Dota prediction.")
            else:
                print("❌ High overfitting detected. Model may be memorizing patterns.")

            print("\nClassification Report:")
            print(classification_report(y_test, test_predictions,
                                        target_names=['Dire Win', 'Radiant Win']))

            return {'test_accuracy': test_accuracy, 'train_accuracy': train_accuracy}

        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")
            raise

    def predict_win_probability(self, radiant_picks, dire_picks, radiant_bans, dire_bans):
        """Predict win probability with error handling"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        try:
            # Create feature vector
            features = self.create_feature_vector(radiant_picks, dire_picks, radiant_bans, dire_bans)
            features = features.reshape(1, -1)

            # Validate features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print("⚠️ Invalid features detected, using default probability")
                return {
                    'radiant_win_probability': 0.5,
                    'dire_win_probability': 0.5
                }

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get probability estimates
            if hasattr(self.model, 'predict_proba'):
                prob_estimates = self.model.predict_proba(features_scaled)[0]
                radiant_win_prob = prob_estimates[1] if len(prob_estimates) > 1 else 0.5
            else:
                # Fallback for models without probability estimates
                prediction = self.model.predict(features_scaled)[0]
                radiant_win_prob = 0.7 if prediction == 1 else 0.3

            return {
                'radiant_win_probability': float(radiant_win_prob),
                'dire_win_probability': float(1 - radiant_win_prob)
            }

        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return {
                'radiant_win_probability': 0.5,
                'dire_win_probability': 0.5
            }

    def save_model(self, filepath):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save!")

        try:
            # Save model
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(self.model, f)

            # Save scaler
            with open(f"{filepath}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)

            print(f"Model saved to {filepath}.pkl")
            print(f"Scaler saved to {filepath}_scaler.pkl")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise

    def load_model(self, filepath):
        """Load a trained model"""
        try:
            with open(f"{filepath}.pkl", 'rb') as f:
                self.model = pickle.load(f)

            with open(f"{filepath}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)

            self.load_hero_data()
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Test data validation
    print("Testing DotaDraftPredictorLight...")

    # Sample test data
    test_data = [
        {
            'match_id': 'test1',
            'radiant_picks': [1, 2, 3, 4, 5],
            'dire_picks': [10, 11, 12, 13, 14],
            'radiant_bans': [20, 21],
            'dire_bans': [30, 31],
            'radiant_won': 1
        }
    ]

    predictor = DotaDraftPredictor()
    try:
        X, y = predictor.prepare_training_data(test_data)
        print(f"✅ Test passed: Feature shape {X.shape}, Labels shape {y.shape}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
