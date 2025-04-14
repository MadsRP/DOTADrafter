# data_collector.py - Functions to interact with the STRATZ API
import requests
import json
import os
import time
import pandas as pd

# Configuration
API_URL = "https://api.stratz.com/graphql"
API_TOKEN = "YOUR_API_TOKEN_HERE"  # Replace with your actual API token


def run_graphql_query(query, variables=None):
    """Execute a GraphQL query to the STRATZ API"""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}" if API_TOKEN != "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                                                               ".eyJTdWJqZWN0IjoiZTg0ZDQ2OTgtNWE3OC00"
                                                               "Mjg3LTg1ZDktODQ1MWUxMTdiNThiIiwiU3Rl"
                                                               "YW1JZCI6Ijg2ODYxNTgiLCJuYmYiOjE3NDQ2Mz"
                                                               "IzOTEsImV4cCI6MTc3NjE2ODM5MSwiaWF0IjoxN"
                                                               "zQ0NjMyMzkxLCJpc3MiOiJodHRwczovL2FwaS5z"
                                                               "dHJhdHouY29tIn0.m5SbeyWgCKHXFw9WuhlLo03R"
                                                               "JbK6rm--2dRle-SwXGY" else None,
        "Content-Type": "application/json",
        "User-Agent": "STRATZ_API"  # Required as per API docs
    }

    # Remove None values from headers
    headers = {k: v for k, v in headers.items() if v is not None}

    request_data = {"query": query}
    if variables:
        request_data["variables"] = variables

    response = requests.post(API_URL, json=request_data, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def fetch_hero_data():
    """Fetch basic information about all heroes"""
    query = """
    query GetHeroes {
      heroes {
        id
        displayName
        shortName
        primaryAttribute
        roles {
          roleId
          level
        }
        stats {
          strengthBase
          strengthGain
          agilityBase
          agilityGain
          intelligenceBase
          intelligenceGain
          attackRange
          moveSpeed
        }
      }
    }
    """

    result = run_graphql_query(query)

    if result and "data" in result and "heroes" in result["data"]:
        heroes = result["data"]["heroes"]
        # Add initial win rate data (placeholder)
        for hero in heroes:
            hero['winRate'] = 50.0  # Default win rate until we gather actual data

        print(f"Fetched data for {len(heroes)} heroes")
        return heroes
    else:
        print("Failed to fetch hero data")
        return []


def fetch_match_data(limit=100, min_rank=70):
    """
    Fetch recent public matches data

    Args:
        limit: Number of matches to fetch
        min_rank: Minimum rank percentile (70 = Ancient and above)
    """
    query = """
    query GetRecentMatches($limit: Int!, $minRank: Short!) {
      matches(
        request: {
          isParsed: true,
          rankBracket: $minRank, 
          take: $limit
        }
      ) {
        id
        didRadiantWin
        durationSeconds
        endDateTime
        gameMode
        lobbyType
        pickBans {
          heroId
          isPick
          isRadiant
          order
        }
        players {
          heroId
          isRadiant
          position
          role
          lane
          kills
          deaths
          assists
          networth
        }
      }
    }
    """

    variables = {
        "limit": limit,
        "minRank": min_rank
    }

    result = run_graphql_query(query, variables)

    if result and "data" in result and "matches" in result["data"]:
        matches = result["data"]["matches"]
        print(f"Fetched {len(matches)} matches")

        # Save matches to file
        os.makedirs('data', exist_ok=True)
        with open(f'data/matches_{int(time.time())}.json', 'w') as f:
            json.dump(matches, f)

        return matches
    else:
        print("Failed to fetch match data")
        return []


def analyze_hero_winrates():
    """Analyze hero win rates from collected match data"""
    # Find all match data files
    match_files = [f for f in os.listdir('data') if f.startswith('matches_')]

    if not match_files:
        print("No match data found. Please fetch match data first.")
        return {}

    # Load hero data for reference
    if os.path.exists('data/heroes.json'):
        with open('data/heroes.json', 'r') as f:
            heroes = json.load(f)
    else:
        heroes = fetch_hero_data()

    # Create hero ID to index mapping
    hero_id_to_index = {hero['id']: i for i, hero in enumerate(heroes)}

    # Initialize counters for wins and total games
    total_games = 0
    hero_picks = {hero['id']: 0 for hero in heroes}
    hero_wins = {hero['id']: 0 for hero in heroes}

    # For hero vs hero matchup analysis
    matchups = {}  # Format: {hero_id: {vs_hero_id: [wins, total]}}

    # For hero synergy analysis
    synergies = {}  # Format: {hero_id: {with_hero_id: [wins, total]}}

    # Process all match files
    for match_file in match_files:
        with open(f'data/{match_file}', 'r') as f:
            matches = json.load(f)

        for match in matches:
            total_games += 1
            radiant_win = match['didRadiantWin']

            # Collect heroes on each team
            radiant_heroes = []
            dire_heroes = []

            for player in match['players']:
                hero_id = player['heroId']
                is_radiant = player['isRadiant']

                # Update pick count
                hero_picks[hero_id] = hero_picks.get(hero_id, 0) + 1

                # Update win count if team won
                if (is_radiant and radiant_win) or (not is_radiant and not radiant_win):
                    hero_wins[hero_id] = hero_wins.get(hero_id, 0) + 1

                # Add hero to team list
                if is_radiant:
                    radiant_heroes.append(hero_id)
                else:
                    dire_heroes.append(hero_id)

            # Process hero vs hero matchups
            for radiant_hero in radiant_heroes:
                if radiant_hero not in matchups:
                    matchups[radiant_hero] = {}

                # Record matchup against each enemy hero
                for dire_hero in dire_heroes:
                    if dire_hero not in matchups[radiant_hero]:
                        matchups[radiant_hero][dire_hero] = [0, 0]

                    matchups[radiant_hero][dire_hero][1] += 1  # Total games
                    if radiant_win:
                        matchups[radiant_hero][dire_hero][0] += 1  # Wins

            for dire_hero in dire_heroes:
                if dire_hero not in matchups:
                    matchups[dire_hero] = {}

                # Record matchup against each enemy hero
                for radiant_hero in radiant_heroes:
                    if radiant_hero not in matchups[dire_hero]:
                        matchups[dire_hero][radiant_hero] = [0, 0]

                    matchups[dire_hero][radiant_hero][1] += 1  # Total games
                    if not radiant_win:
                        matchups[dire_hero][radiant_hero][0] += 1  # Wins

            # Process hero synergies (heroes on same team)
            # Radiant team synergies
            for i, hero1 in enumerate(radiant_heroes):
                if hero1 not in synergies:
                    synergies[hero1] = {}

                for hero2 in radiant_heroes[i + 1:]:
                    if hero2 not in synergies[hero1]:
                        synergies[hero1][hero2] = [0, 0]

                    synergies[hero1][hero2][1] += 1  # Total games together
                    if radiant_win:
                        synergies[hero1][hero2][0] += 1  # Wins together

                    # Add reverse relationship
                    if hero2 not in synergies:
                        synergies[hero2] = {}
                    if hero1 not in synergies[hero2]:
                        synergies[hero2][hero1] = [0, 0]

                    synergies[hero2][hero1][1] += 1
                    if radiant_win:
                        synergies[hero2][hero1][0] += 1

            # Dire team synergies
            for i, hero1 in enumerate(dire_heroes):
                if hero1 not in synergies:
                    synergies[hero1] = {}

                for hero2 in dire_heroes[i + 1:]:
                    if hero2 not in synergies[hero1]:
                        synergies[hero1][hero2] = [0, 0]

                    synergies[hero1][hero2][1] += 1  # Total games together
                    if not radiant_win:
                        synergies[hero1][hero2][0] += 1  # Wins together

                    # Add reverse relationship
                    if hero2 not in synergies:
                        synergies[hero2] = {}
                    if hero1 not in synergies[hero2]:
                        synergies[hero2][hero1] = [0, 0]

                    synergies[hero2][hero1][1] += 1
                    if not radiant_win:
                        synergies[hero2][hero1][0] += 1

    # Calculate win rates for each hero
    win_rates = {}
    for hero_id, picks in hero_picks.items():
        if picks > 0:
            win_rate = (hero_wins.get(hero_id, 0) / picks) * 100
            win_rates[hero_id] = round(win_rate, 1)

    # Calculate matchup win rates
    matchup_rates = {}
    for hero_id, vs_heroes in matchups.items():
        matchup_rates[hero_id] = {}
        for vs_hero_id, (wins, total) in vs_heroes.items():
            if total > 0:
                win_rate = (wins / total) * 100
                matchup_rates[hero_id][vs_hero_id] = round(win_rate, 1)

    # Calculate synergy win rates
    synergy_rates = {}
    for hero_id, with_heroes in synergies.items():
        synergy_rates[hero_id] = {}
        for with_hero_id, (wins, total) in with_heroes.items():
            if total > 0:
                win_rate = (wins / total) * 100
                synergy_rates[hero_id][with_hero_id] = round(win_rate, 1)

    # Update hero data with win rates
    for hero in heroes:
        hero_id = hero['id']
        if hero_id in win_rates:
            hero['winRate'] = win_rates[hero_id]

    # Save updated hero data
    with open('data/heroes.json', 'w') as f:
        json.dump(heroes, f)

    # Save matchup data
    with open('data/matchups.json', 'w') as f:
        json.dump(matchup_rates, f)

    # Save synergy data
    with open('data/synergies.json', 'w') as f:
        json.dump(synergy_rates, f)

    return {
        'heroes': heroes,
        'matchups': matchup_rates,
        'synergies': synergy_rates
    }


if __name__ == "__main__":
    # If ran directly, fetch hero data and some matches
    print("Fetching hero data...")
    heroes = fetch_hero_data()

    print("Fetching match data...")
    matches = fetch_match_data(limit=50)

    print("Analyzing win rates...")
    analyze_hero_winrates()
