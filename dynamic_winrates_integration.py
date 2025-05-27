# dynamic_winrates_integration.py - Update all hero win rates dynamically
import json
import os
import time

from ml_model import DotaDraftPredictorLight


class DynamicWinRateEngine:
    def __init__(self, model_path='models/dota_draft_predictor'):
        self.predictor = DotaDraftPredictorLight()
        self.model_loaded = False
        self.model_path = model_path
        self.hero_cache = {}  # Cache hero data
        self.load_model()
        self.load_hero_cache()

    def load_hero_cache(self):
        """Load and cache hero data for faster lookups"""
        try:
            with open('data/heroes.json', 'r') as f:
                heroes = json.load(f)
                self.hero_cache = {hero['id']: hero for hero in heroes}
                print(f"Cached {len(heroes)} heroes for dynamic updates")
        except Exception as e:
            print(f"Error loading hero cache: {e}")
            self.hero_cache = {}

    def load_model(self):
        """Load the trained ML model"""
        try:
            if os.path.exists(f"{self.model_path}.pkl"):
                self.predictor.load_model(self.model_path)
                self.model_loaded = True
                print("✅ Dynamic win rate model loaded successfully!")
            else:
                print(f"❌ No trained model found at {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False

    def get_all_hero_winrates(self, draft_state):
        """
        Calculate win rates for ALL available heroes given the current draft state
        This is the main function that updates all heroes at once
        """
        if not self.model_loaded:
            return self._get_fallback_winrates(draft_state)

        start_time = time.time()

        try:
            radiant_picks = draft_state['radiant']['picks']
            dire_picks = draft_state['dire']['picks']
            radiant_bans = draft_state['radiant']['bans']
            dire_bans = draft_state['dire']['bans']
            current_team = draft_state['currentTeam']
            current_action = draft_state['currentAction']

            # Get all picked/banned heroes
            unavailable_hero_ids = set(radiant_picks + dire_picks + radiant_bans + dire_bans)

            # Get all available heroes
            all_hero_ids = list(self.hero_cache.keys())
            available_heroes = [hid for hid in all_hero_ids if hid not in unavailable_hero_ids]

            print(f"Calculating win rates for {len(available_heroes)} available heroes...")

            # Calculate win rates for all available heroes
            hero_winrates = {}
            baseline_prob = self._get_baseline_probability(radiant_picks, dire_picks, radiant_bans, dire_bans,
                                                           current_team)

            # Batch process heroes for efficiency
            for hero_id in available_heroes:
                # Test adding this hero to current team
                test_radiant_picks = radiant_picks.copy()
                test_dire_picks = dire_picks.copy()

                if current_team == 'radiant':
                    test_radiant_picks.append(hero_id)
                else:
                    test_dire_picks.append(hero_id)

                # Get win probability with this hero
                try:
                    prediction = self.predictor.predict_win_probability(
                        test_radiant_picks, test_dire_picks, radiant_bans, dire_bans
                    )

                    win_prob = (prediction['radiant_win_probability'] if current_team == 'radiant'
                                else prediction['dire_win_probability'])

                    # Calibrate win rate
                    calibrated_rate = self._calibrate_win_rate(win_prob, baseline_prob)
                    hero_winrates[hero_id] = round(calibrated_rate, 1)

                except Exception as e:
                    # Fallback for individual hero errors
                    hero_winrates[hero_id] = 50.0

            # Set unavailable heroes to 0% (or some indicator)
            for hero_id in unavailable_hero_ids:
                hero_winrates[hero_id] = 0.0  # Unavailable

            elapsed_time = time.time() - start_time
            print(f"✅ Calculated {len(hero_winrates)} hero win rates in {elapsed_time:.2f} seconds")

            return hero_winrates

        except Exception as e:
            print(f"❌ Error calculating dynamic win rates: {e}")
            return self._get_fallback_winrates(draft_state)

    def _get_baseline_probability(self, radiant_picks, dire_picks, radiant_bans, dire_bans, current_team):
        """Get baseline win probability for current draft state"""
        if not radiant_picks and not dire_picks:
            return 0.5

        try:
            prediction = self.predictor.predict_win_probability(radiant_picks, dire_picks, radiant_bans, dire_bans)
            return (prediction['radiant_win_probability'] if current_team == 'radiant'
                    else prediction['dire_win_probability'])
        except:
            return 0.5

    def _calibrate_win_rate(self, raw_prob, baseline_prob):
        """Calibrate win rate for better spread"""
        improvement = raw_prob - baseline_prob
        calibrated = 50 + (improvement * 80)  # Scale improvement
        return max(25, min(75, calibrated))  # Bound between 25-75%

    def _get_fallback_winrates(self, draft_state):
        """Fallback win rates when ML model isn't available"""
        print("Using fallback win rates (no ML model)")

        hero_winrates = {}
        unavailable_hero_ids = set(
            draft_state['radiant']['picks'] +
            draft_state['dire']['picks'] +
            draft_state['radiant']['bans'] +
            draft_state['dire']['bans']
        )

        for hero_id in self.hero_cache.keys():
            if hero_id in unavailable_hero_ids:
                hero_winrates[hero_id] = 0.0  # Unavailable
            else:
                # Simple fallback calculation
                base_rate = 45 + (hash(str(hero_id)) % 20)  # 45-65% range
                hero_winrates[hero_id] = round(base_rate, 1)

        return hero_winrates

    def get_recommendations(self, draft_state):
        """Get top recommendations (separate from all win rates)"""
        if not self.model_loaded:
            return self._get_simple_recommendations(draft_state)

        try:
            # Get all win rates
            all_winrates = self.get_all_hero_winrates(draft_state)

            # Filter to available heroes and sort by win rate
            available_heroes = [(hid, rate) for hid, rate in all_winrates.items() if rate > 0]
            available_heroes.sort(key=lambda x: x[1], reverse=True)

            # Create recommendation format
            recommendations = []
            for hero_id, win_rate in available_heroes[:10]:  # Top 10
                hero_name = self.hero_cache.get(hero_id, {}).get('displayName', 'Unknown Hero')

                recommendations.append({
                    'id': hero_id,
                    'name': hero_name,
                    'winRate': win_rate,
                    'reasons': self._get_recommendation_reason(win_rate, draft_state['currentTeam'])
                })

            return recommendations

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return self._get_simple_recommendations(draft_state)

    def _get_recommendation_reason(self, win_rate, current_team):
        """Generate reason based on win rate"""
        if win_rate >= 65:
            return f"Excellent synergy for {current_team}"
        elif win_rate >= 55:
            return f"Strong pick for {current_team}"
        elif win_rate >= 45:
            return f"Balanced choice for {current_team}"
        else:
            return f"Situational pick for {current_team}"

    def _get_simple_recommendations(self, draft_state):
        """Simple recommendations fallback"""
        recommendations = []
        unavailable_ids = set(
            draft_state['radiant']['picks'] +
            draft_state['dire']['picks'] +
            draft_state['radiant']['bans'] +
            draft_state['dire']['bans']
        )

        count = 0
        for hero_id, hero_data in self.hero_cache.items():
            if hero_id not in unavailable_ids and count < 10:
                win_rate = 45 + (hash(hero_data['displayName']) % 20)
                recommendations.append({
                    'id': hero_id,
                    'name': hero_data['displayName'],
                    'winRate': round(win_rate, 1),
                    'reasons': 'Estimated recommendation'
                })
                count += 1

        return recommendations

    def get_draft_analysis(self, draft_state):
        """Get draft analysis"""
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
                analysis = "Radiant has a strong draft advantage!"
            elif radiant_prob > 0.55:
                analysis = "Radiant has a slight edge in draft"
            elif radiant_prob < 0.4:
                analysis = "Dire has a strong draft advantage!"
            elif radiant_prob < 0.45:
                analysis = "Dire has a slight edge in draft"
            else:
                analysis = "Draft is well balanced between teams"

            return {
                'radiant_win_probability': radiant_prob,
                'dire_win_probability': prediction['dire_win_probability'],
                'analysis': analysis
            }

        except Exception as e:
            print(f"Error in draft analysis: {e}")
            return {
                'radiant_win_probability': 0.5,
                'dire_win_probability': 0.5,
                'analysis': 'Error analyzing draft'
            }


# Global instance
dynamic_engine = None


def initialize_dynamic_engine():
    """Initialize the dynamic win rate engine"""
    global dynamic_engine
    if dynamic_engine is None:
        dynamic_engine = DynamicWinRateEngine()
    return dynamic_engine


def get_all_hero_winrates(draft_data):
    """Get win rates for all heroes - NEW FUNCTION"""
    engine = initialize_dynamic_engine()
    return engine.get_all_hero_winrates(draft_data)


def get_hero_recommendations(draft_data):
    """Get top hero recommendations"""
    engine = initialize_dynamic_engine()
    return engine.get_recommendations(draft_data)


def get_draft_win_probabilities(draft_data):
    """Get draft analysis"""
    engine = initialize_dynamic_engine()
    return engine.get_draft_analysis(draft_data)
