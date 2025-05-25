# lightweight_ml_model.py - ML model using scikit-learn instead of TensorFlow
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class DotaDraftPredictorLight:
    def __init__(self, max_hero_id=150):
        """
        Lightweight version using scikit-learn
        """
        self.max_hero_id = max_hero_id
        self.scaler = StandardScaler()
        self.model = None
        self.hero_data = None

    def load_hero_data(self):
        """Load hero information"""
        try:
            with open('data/heroes.json', 'r') as f:
                heroes = json.load(f)
                self.hero_data = {hero['id']: hero for hero in heroes}
                print(f"Loaded {len(heroes)} heroes")
        except Exception as e:
            print(f"Could not load hero data: {e}")
            self.hero_data = {}

    def create_feature_vector(self, radiant_picks, dire_picks, radiant_bans, dire_bans):
        """Convert draft state into feature vector"""
        features = np.zeros(self.max_hero_id + 1)

        # Encode picks: +1 for radiant, -1 for dire
        for hero_id in radiant_picks:
            if hero_id <= self.max_hero_id:
                features[hero_id] = 1.0

        for hero_id in dire_picks:
            if hero_id <= self.max_hero_id:
                features[hero_id] = -1.0

        # Encode bans: 0.5 (neutral but unavailable)
        for hero_id in radiant_bans + dire_bans:
            if hero_id <= self.max_hero_id and features[hero_id] == 0:
                features[hero_id] = 0.5

        # Add composition features
        composition_features = self._calculate_composition_features(
            radiant_picks, dire_picks
        )

        # Combine features
        full_features = np.concatenate([features, composition_features])

        return full_features

    def _calculate_composition_features(self, radiant_picks, dire_picks):
        """Calculate team composition features"""
        if not self.hero_data:
            return np.zeros(8)

        def get_attribute_counts(picks):
            str_count = agi_count = int_count = all_count = 0

            for hero_id in picks:
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
        """Convert training data into feature matrices"""
        print("Preparing training data...")

        X = []
        y = []

        for match in training_data:
            features = self.create_feature_vector(
                match['radiant_picks'],
                match['dire_picks'],
                match['radiant_bans'],
                match['dire_bans']
            )

            X.append(features)
            y.append(match['radiant_won'])

        X = np.array(X)
        y = np.array(y)

        print(f"Created feature matrix: {X.shape}")

        return X, y

    def train(self, training_data, test_size=0.2, model_type='neural_network'):
        """Train the model"""
        print("Starting training process...")

        # Load hero data
        self.load_hero_data()

        # Prepare data
        X, y = self.prepare_training_data(training_data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Choose model
        if model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            print("Using Neural Network (MLP)")
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            print("Using Random Forest")

        # Train the model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, test_predictions,
                                    target_names=['Dire Win', 'Radiant Win']))

        return {'test_accuracy': test_accuracy, 'train_accuracy': train_accuracy}

    def predict_win_probability(self, radiant_picks, dire_picks, radiant_bans, dire_bans):
        """Predict win probability"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Create feature vector
        features = self.create_feature_vector(radiant_picks, dire_picks, radiant_bans, dire_bans)
        features = features.reshape(1, -1)

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

    def recommend_heroes(self, current_team, radiant_picks, dire_picks, radiant_bans, dire_bans,
                         available_heroes=None, top_k=5):
        """Recommend heroes"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if available_heroes is None:
            picked_banned = set(radiant_picks + dire_picks + radiant_bans + dire_bans)
            available_heroes = [i for i in range(1, self.max_hero_id + 1)
                                if i not in picked_banned]

        recommendations = []

        for hero_id in available_heroes:
            # Test adding this hero
            test_radiant_picks = radiant_picks.copy()
            test_dire_picks = dire_picks.copy()

            if current_team == 'radiant':
                test_radiant_picks.append(hero_id)
            else:
                test_dire_picks.append(hero_id)

            # Get win probability
            prediction = self.predict_win_probability(
                test_radiant_picks, test_dire_picks, radiant_bans, dire_bans
            )

            win_prob = (prediction['radiant_win_probability'] if current_team == 'radiant'
                        else prediction['dire_win_probability'])

            hero_name = "Unknown Hero"
            if self.hero_data and hero_id in self.hero_data:
                hero_name = self.hero_data[hero_id]['displayName']

            recommendations.append({
                'id': hero_id,
                'name': hero_name,
                'winRate': round(win_prob * 100, 1),
                'reasons': f'Increases {current_team} win probability'
            })

        # Sort and return top k
        recommendations.sort(key=lambda x: x['winRate'], reverse=True)
        return recommendations[:top_k]

    def save_model(self, filepath):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save!")

        # Save model
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.model, f)

        # Save scaler
        with open(f"{filepath}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"Model saved to {filepath}.pkl")
        print(f"Scaler saved to {filepath}_scaler.pkl")

    def load_model(self, filepath):
        """Load a trained model"""
        with open(f"{filepath}.pkl", 'rb') as f:
            self.model = pickle.load(f)

        with open(f"{filepath}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)

        self.load_hero_data()
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Load training data
    print("Loading training data...")
    training_files = [f for f in os.listdir('data') if f.startswith('training_matches_') and f.endswith('.json')]

    if not training_files:
        print("No training data found! Run working_data_collector.py first.")
        exit()

    latest_file = sorted(training_files)[-1]
    print(f"Using training data: {latest_file}")

    with open(f'data/{latest_file}', 'r') as f:
        training_data = json.load(f)

    print(f"Loaded {len(training_data)} training samples")

    # Initialize and train model
    predictor = DotaDraftPredictorLight()
    results = predictor.train(training_data, model_type='neural_network')

    # Save the model
    os.makedirs('models', exist_ok=True)
    predictor.save_model('models/dota_draft_predictor')

    # Test prediction
    print("\n=== Testing Predictions ===")
    test_match = training_data[0]
    prediction = predictor.predict_win_probability(
        test_match['radiant_picks'],
        test_match['dire_picks'],
        test_match['radiant_bans'],
        test_match['dire_bans']
    )

    print(f"Predicted Radiant win probability: {prediction['radiant_win_probability']:.3f}")
    print(f"Actual result: {'Radiant Won' if test_match['radiant_won'] else 'Dire Won'}")

    # Test recommendations
    print("\n=== Testing Recommendations ===")
    recommendations = predictor.recommend_heroes(
        'radiant',
        test_match['radiant_picks'][:4],
        test_match['dire_picks'][:4],
        test_match['radiant_bans'],
        test_match['dire_bans']
    )

    print("Top 5 hero recommendations:")
    for rec in recommendations:
        print(f"- {rec['name']}: {rec['winRate']}% win rate")