# lightweight_integration.py - Integration using scikit-learn model
import os
import json
from lightweight_ml_model import DotaDraftPredictorLight

class LightweightRecommendationEngine:
    def __init__(self, model_path='models/dota_draft_predictor'):
        self.predictor = DotaDraftPredictorLight()
        self.model_loaded = False
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load the trained ML model"""
        try:
            if os.path.exists(f"{self.model_path}.pkl"):
                self.predictor.load_model(self.model_path)
                self.model_loaded = True
                print("✅ Lightweight ML model loaded successfully!")
            else:
                print(f"❌ No trained model found at {self.model_path}")
                print("Run 'python simple_train.py' first")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False

    def get_recommendations(self, draft_state):
        """Get hero recommendations"""
        if not self.model_loaded:
            return self._get_fallback_recommendations(draft_state)

        try:
            # Extract draft information
            radiant_picks = draft_state['radiant']['picks']
            dire_picks = draft_state['dire']['picks']
            radiant_bans = draft_state['radiant']['bans']
            dire_bans = draft_state['dire']['bans']
            current_team = draft_state['currentTeam']
            current_action = draft_state['currentAction']

            if current_action == 'pick':
                # Get hero recommendations for picks
                recommendations = self.predictor.recommend_heroes(
                    current_team=current_team,
                    radiant_picks=radiant_picks,
                    dire_picks=dire_picks,
                    radiant_bans=radiant_bans,
                    dire_bans=dire_bans,
                    top_k=10
                )
            else:
                # For bans, recommend heroes that would be strong for the enemy
                enemy_team = 'dire' if current_team == 'radiant' else 'radiant'

                enemy_recommendations = self.predictor.recommend_heroes(
                    current_team=enemy_team,
                    radiant_picks=radiant_picks,
                    dire_picks=dire_picks,
                    radiant_bans=radiant_bans,
                    dire_bans=dire_bans,
                    top_k=10
                )

                # Convert to ban recommendations
                recommendations = []
                for rec in enemy_recommendations:
                    recommendations.append({
                        'id': rec['id'],
                        'name': rec['name'],
                        'winRate': round(100 - rec['winRate'], 1),  # Only 1 decimal place
                        'reasons': f'Deny strong pick from {enemy_team}'
                    })

            return recommendations

        except Exception as e:
            print(f"❌ Error getting ML recommendations: {e}")
            return self._get_fallback_recommendations(draft_state)

    def get_draft_analysis(self, draft_state):
        """Analyze current draft"""
        if not self.model_loaded:
            return {
                'radiant_win_probability': 0.5,
                'dire_win_probability': 0.5,
                'analysis': 'ML model not available'
            }

        try:
            prediction = self.predictor.predict_win_probability(
                draft_state['radiant']['picks'],
                draft_state['dire']['picks'],
                draft_state['radiant']['bans'],
                draft_state['dire']['bans']
            )

            radiant_prob = prediction['radiant_win_probability']

            if radiant_prob > 0.6:
                analysis = "Radiant has a strong draft advantage"
            elif radiant_prob < 0.4:
                analysis = "Dire has a strong draft advantage"
            else:
                analysis = "Draft is relatively balanced"

            return {
                'radiant_win_probability': radiant_prob,
                'dire_win_probability': prediction['dire_win_probability'],
                'analysis': analysis
            }

        except Exception as e:
            print(f"❌ Error in draft analysis: {e}")
            return {
                'radiant_win_probability': 0.5,
                'dire_win_probability': 0.5,
                'analysis': 'Error analyzing draft'
            }

    def _get_fallback_recommendations(self, draft_state):
        """Fallback recommendations when ML model isn't available"""
        print("Using fallback recommendations (no ML model)")

        try:
            with open('data/heroes.json', 'r') as f:
                heroes = json.load(f)
        except:
            heroes = []

        # Get already selected heroes
        selected_hero_ids = set()
        selected_hero_ids.update(draft_state['radiant']['picks'])
        selected_hero_ids.update(draft_state['dire']['picks'])
        selected_hero_ids.update(draft_state['radiant']['bans'])
        selected_hero_ids.update(draft_state['dire']['bans'])

        # Filter available heroes
        available_heroes = [h for h in heroes if h['id'] not in selected_hero_ids]

        recommendations = []

        for hero in available_heroes[:15]:  # Consider top 15 available
            # Simple scoring
            base_winrate = 50 + (hash(hero['displayName']) % 10) - 5  # 45-55%

            recommendations.append({
                'id': hero['id'],
                'name': hero['displayName'],
                'winRate': round(base_winrate, 1),  # Only 1 decimal place
                'reasons': 'Basic recommendation (no ML model)'
            })

        # Sort by win rate
        recommendations.sort(key=lambda x: x['winRate'], reverse=True)
        return recommendations[:10]

# Global instance
recommendation_engine = None

def initialize_lightweight_engine():
    """Initialize the lightweight recommendation engine"""
    global recommendation_engine
    if recommendation_engine is None:
        recommendation_engine = LightweightRecommendationEngine()
    return recommendation_engine

def get_hero_recommendations(draft_data):
    """Get recommendations using lightweight model"""
    engine = initialize_lightweight_engine()
    return engine.get_recommendations(draft_data)

def get_draft_win_probabilities(draft_data):
    """Get win probability analysis using lightweight model"""
    engine = initialize_lightweight_engine()
    return engine.get_draft_analysis(draft_data)