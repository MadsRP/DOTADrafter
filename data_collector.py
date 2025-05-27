# data_collector.py - Only check for 10 picks and winner
import json
import os
import random
import time

import requests
from dotenv import load_dotenv

load_dotenv()


def load_env_file(file_path):
    """Load environment variables from any file path"""
    if not os.path.exists(file_path):
        print(f"Environment file not found: {file_path}")
        return False

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        return True
    except Exception as e:
        print(f"Error loading environment file: {e}")
        return False


# Load environment
env_file_path = '.env'
load_env_file(env_file_path)
API_TOKEN = os.environ.get("STRATZ_API_TOKEN")
API_URL = "https://api.stratz.com/graphql"


def run_graphql_query(query, variables=None):
    """Execute a GraphQL query to the STRATZ API"""
    if not API_TOKEN:
        print("ERROR: No API token available")
        return None

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "STRATZ_API"
    }

    request_data = {"query": query}
    if variables:
        request_data["variables"] = variables

    try:
        response = requests.post(API_URL, json=request_data, headers=headers)
        response.raise_for_status()
        result = response.json()

        if "errors" in result:
            print(f"GraphQL errors: {result['errors']}")
            return None

        return result
    except Exception as e:
        print(f"API request failed: {e}")
        return None


def generate_recent_match_ids(latest_match_id=8307478229, count=100):
    """Generate realistic recent match IDs"""
    print(f"🎯 Generating {count} recent match IDs from latest: {latest_match_id}")

    match_ids = []

    # Generate IDs going backwards from the latest match
    for i in range(count):
        offset = random.randint(50, 500)  # Reasonable gaps
        match_id = latest_match_id - (i * offset)
        match_ids.append(match_id)

    # Remove duplicates and sort
    match_ids = sorted(list(set(match_ids)), reverse=True)

    print(f"✅ Generated {len(match_ids)} match IDs from {max(match_ids)} to {min(match_ids)}")
    return match_ids[:count]


def fetch_match_details_batch(candidate_ids, max_valid_matches=50):
    """Fetch match details from candidate IDs - STEP 2"""
    print(f"📥 Fetching details for {len(candidate_ids)} candidate match IDs...")
    print(f"🎯 Target: {max_valid_matches} valid matches")

    valid_matches = []
    attempted = 0

    for match_id in candidate_ids:
        if len(valid_matches) >= max_valid_matches:
            print(f"✅ Got {max_valid_matches} valid matches, stopping")
            break

        attempted += 1
        print(f"Fetching match {attempted}/{len(candidate_ids)}: {match_id}")

        # Simple STRATZ query - just get picks and winner
        query = """
        query GetMatch($matchId: Long!) {
          match(id: $matchId) {
            id
            didRadiantWin
            durationSeconds
            pickBans {
              heroId
              isPick
              isRadiant
            }
          }
        }
        """

        variables = {"matchId": match_id}
        result = run_graphql_query(query, variables)

        if result and "data" in result and "match" in result["data"] and result["data"]["match"]:
            match_data = result["data"]["match"]

            # SIMPLE validation: just check for 10 picks and winner
            pick_bans = match_data.get('pickBans') or []
            picks = [pb for pb in pick_bans if pb and pb.get('isPick', False)]

            has_10_picks = len(picks) == 10
            has_winner = match_data.get('didRadiantWin') is not None

            if has_10_picks and has_winner:
                # Count picks per team to make sure it's 5v5
                radiant_picks = [p for p in picks if p.get('isRadiant', False)]
                dire_picks = [p for p in picks if not p.get('isRadiant', True)]

                if len(radiant_picks) == 5 and len(dire_picks) == 5:
                    valid_matches.append(match_data)
                    duration_min = (match_data.get('durationSeconds') or 0) // 60
                    winner = "Radiant" if match_data.get('didRadiantWin') else "Dire"
                    print(f"✅ Match {match_id}: Valid! {winner} won, {duration_min} min")
                else:
                    print(f"⚠️ Match {match_id}: Uneven teams (R:{len(radiant_picks)} D:{len(dire_picks)})")
            else:
                reasons = []
                if not has_10_picks:
                    reasons.append(f"only {len(picks)} picks")
                if not has_winner:
                    reasons.append("no winner data")
                print(f"⚠️ Match {match_id}: Skipped ({', '.join(reasons)})")
        else:
            print(f"❌ Match {match_id}: Not found or no data")

        # Respect API rate limits
        time.sleep(0.05)  # 50ms delay

        # Progress update every 10 matches
        if attempted % 10 == 0:
            success_rate = len(valid_matches) / attempted * 100
            print(f"📊 Progress: {len(valid_matches)}/{attempted} valid ({success_rate:.1f}% success rate)")

    print(f"✅ Final result: {len(valid_matches)} valid matches from {attempted} attempts")
    return valid_matches


def convert_to_training_format(raw_matches):
    """Convert raw STRATZ matches to training format - STEP 3"""
    print(f"🔄 Converting {len(raw_matches)} raw matches to training format...")

    training_data = []

    for match in raw_matches:
        try:
            # Extract picks and bans
            pick_bans = match.get('pickBans', [])
            picks = [pb for pb in pick_bans if pb and pb.get('isPick', False)]
            bans = [pb for pb in pick_bans if pb and not pb.get('isPick', True)]

            # Separate by team
            radiant_picks = [p['heroId'] for p in picks if p.get('isRadiant', False)]
            dire_picks = [p['heroId'] for p in picks if not p.get('isRadiant', True)]
            radiant_bans = [b['heroId'] for b in bans if b.get('isRadiant', False)]
            dire_bans = [b['heroId'] for b in bans if not b.get('isRadiant', True)]

            # Create training sample
            training_sample = {
                "match_id": str(match['id']),
                "radiant_picks": radiant_picks,
                "dire_picks": dire_picks,
                "radiant_bans": radiant_bans,  # Might be empty for some matches
                "dire_bans": dire_bans,  # Might be empty for some matches
                "radiant_won": 1 if match.get('didRadiantWin', False) else 0,
                "duration_seconds": match.get('durationSeconds', 0),
                "source": "real_stratz_api"
            }

            # Validate we have 5 picks per team
            if len(radiant_picks) == 5 and len(dire_picks) == 5:
                training_data.append(training_sample)

        except Exception as e:
            print(f"❌ Error converting match {match.get('id', 'unknown')}: {e}")
            continue

    print(f"✅ Converted {len(training_data)} matches to training format")
    return training_data


def process_and_save_matches(raw_matches):
    """Process raw matches and save to files - STEP 4"""
    print(f"💾 Processing and saving {len(raw_matches)} matches...")

    if not raw_matches:
        print("❌ No matches to process")
        return []

    # Convert to training format
    training_data = convert_to_training_format(raw_matches)

    # Save training data
    timestamp = int(time.time())
    os.makedirs('data', exist_ok=True)

    # Save training data (this is what we actually need)
    training_filename = f'data/training_matches_general_{timestamp}.json'
    with open(training_filename, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"💾 Saved {len(training_data)} training samples to {training_filename}")
    print(f"🗑️ Raw match data not saved (only training format needed)")

    # Show statistics
    if training_data:
        radiant_wins = sum(1 for m in training_data if m['radiant_won'] == 1)
        win_rate = radiant_wins / len(training_data) * 100
        avg_duration = sum(m['duration_seconds'] for m in training_data) / len(training_data) / 60

        print(f"📊 Training Data Quality:")
        print(f"   Total matches: {len(training_data)}")
        print(f"   Radiant win rate: {win_rate:.1f}%")
        print(f"   Average duration: {avg_duration:.1f} minutes")
        print(f"   All matches have complete 5v5 hero picks")

        # Check ban data availability
        matches_with_bans = sum(1 for m in training_data if len(m['radiant_bans']) > 0 or len(m['dire_bans']) > 0)
        print(f"   Matches with ban data: {matches_with_bans}/{len(training_data)} ({matches_with_bans / len(training_data) * 100:.1f}%)")

    return training_data


def fetch_hero_data():
    """Fetch hero data"""
    print("📋 Fetching hero data...")

    query = """
    {
      constants {
        heroes {
          id
          displayName
          shortName
          stats {
            primaryAttribute
          }
        }
      }
    }
    """

    result = run_graphql_query(query)
    heroes = []

    if result and "data" in result and "constants" in result["data"] and "heroes" in result["data"]["constants"]:
        api_heroes = result["data"]["constants"]["heroes"]

        for hero in api_heroes:
            processed_hero = {
                "id": hero["id"],
                "displayName": hero["displayName"],
                "shortName": hero["shortName"],
                "winRate": 50
            }

            if "stats" in hero and hero["stats"] and "primaryAttribute" in hero["stats"]:
                processed_hero["primaryAttribute"] = hero["stats"]["primaryAttribute"]
            else:
                processed_hero["primaryAttribute"] = "all"

            heroes.append(processed_hero)

        print(f"✅ Fetched {len(heroes)} heroes")

    # Save heroes
    os.makedirs('data', exist_ok=True)
    with open('data/heroes.json', 'w') as f:
        json.dump(heroes, f, indent=2)

    return heroes


def fetch_hero_portraits(heroes):
    """Download hero portraits from Valve's CDN"""
    import os
    import requests
    from time import sleep

    # Create directory for hero images if it doesn't exist
    os.makedirs('static/images/heroes', exist_ok=True)

    for hero in heroes:
        hero_id = hero['id']
        hero_name = hero['shortName']

        # Path to save the image
        image_path = f"static/images/heroes/{hero_id}.jpg"

        # Skip if file already exists
        if os.path.exists(image_path):
            continue

        # Format the hero name for Valve's API (lowercase, no spaces or hyphens)
        valve_name = hero_name.lower().replace(' ', '_').replace('-', '')

        try:
            # Use Valve's CDN for hero portraits
            image_url = f"https://cdn.dota2.com/apps/dota2/images/heroes/{valve_name}_vert.jpg"

            print(f"Trying to download from: {image_url}")

            # Download the image
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses

            # Save the image
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded portrait for {hero_name} (ID: {hero_id})")

            # Sleep a bit to avoid hitting rate limits
            sleep(0.1)

        except Exception as e:
            print(f"Failed to download portrait for {hero_name}: {e}")
            try:
                # Try alternative URL format (horizontal version)
                alt_image_url = f"https://cdn.dota2.com/apps/dota2/images/heroes/{valve_name}_hphover.png"
                print(f"Trying alternative URL: {alt_image_url}")

                response = requests.get(alt_image_url, stream=True)
                response.raise_for_status()

                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Downloaded portrait using alternative URL for {hero_name}")
                sleep(0.1)

            except Exception as alt_e:
                print(f"Also failed with alternative URL: {alt_e}")
                try:
                    # Try one more format from OpenDota
                    opendota_url = f"https://api.opendota.com/apps/dota2/images/heroes/{valve_name}_full.png"
                    print(f"Trying OpenDota URL: {opendota_url}")

                    response = requests.get(opendota_url, stream=True)
                    response.raise_for_status()

                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print(f"Downloaded portrait using OpenDota URL for {hero_name}")
                    sleep(0.1)

                except Exception as od_e:
                    print(f"All download attempts failed for {hero_name}")


def main():
    """Main data collection orchestrator - CLEAR FLOW"""
    print("🎮 DOTA 2 DRAFT PREDICTOR - DATA COLLECTOR")
    print("=" * 50)
    print("🎯 Strategy: Multi-step data collection process")
    print("📋 Steps: Generate IDs → Fetch Matches → Process → Save")

    if not API_TOKEN:
        print("❌ No STRATZ API token found!")
        print("Please set STRATZ_API_TOKEN in your environment file")
        return

    print(f"✅ API token loaded: {API_TOKEN[:10]}...")

    # STEP 0: Fetch heroes first
    print("\n0️⃣ Fetching hero data...")
    heroes = fetch_hero_data()
    fetch_hero_portraits(heroes)

    # STEP 1: Generate candidate match IDs
    print(f"\n1️⃣ Generating candidate match IDs...")
    latest_match_id = 8307478229  # Known recent match
    target_matches = 400  # How many matches we want
    candidate_ids = generate_recent_match_ids(latest_match_id, count=target_matches * 2)

    # STEP 2: Fetch real match data from candidates
    print(f"\n2️⃣ Fetching match details from candidates...")
    raw_matches = fetch_match_details_batch(candidate_ids, max_valid_matches=target_matches)

    # STEP 3: Process and save the results
    print(f"\n3️⃣ Processing and saving results...")
    training_data = process_and_save_matches(raw_matches)

    # STEP 4: Final summary
    print(f"\n🎉 DATA COLLECTION COMPLETE!")
    print(f"=" * 50)
    if training_data:
        print(f"✅ Successfully collected {len(training_data)} training matches")
        print(f"✅ All matches have complete 5v5 hero picks")
        print(f"✅ All matches have winner data")
        print(f"✅ Ready for ML training!")

        print(f"\n🚀 Next steps:")
        print(f"1. Run: python data_training.py")
        print(f"2. Your model will train on real Dota 2 match outcomes!")
    else:
        print(f"⚠️ No valid matches collected")

    print(f"\n📊 Final Summary:")
    print(f"Heroes: {len(heroes) if heroes else 0}")
    print(f"Training matches: {len(training_data) if training_data else 0}")


if __name__ == "__main__":
    main()
