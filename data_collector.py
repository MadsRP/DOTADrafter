# data_collector.py - Functions to interact with the STRATZ API
import requests
import json
import os
import time
import pandas as pd
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
        print(f"Successfully loaded environment from {file_path}")
        return True
    except Exception as e:
        print(f"Error loading environment file: {e}")
        return False


# Your file path - adjust as needed
env_file_path = 'enviromentvariables.env'

# Load the environment variables
load_env_file(env_file_path)

# Now access the variable
API_TOKEN = os.environ.get("STRATZ_API_TOKEN")

# Configuration
API_URL = "https://api.stratz.com/graphql"

print(f"API Token value: {os.getenv(API_TOKEN)}")
print(f"API Token from environ: {os.environ.get('STRATZ_API_TOKEN')}")


def run_graphql_query(query, variables=None):
    """Execute a GraphQL query to the STRATZ API with improved error handling"""
    import requests
    import os
    import json

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "STRATZ_API"  # Required as per API docs
    }

    request_data = {"query": query}
    if variables:
        request_data["variables"] = variables

    print(f"Making API request to {API_URL}")
    print(f"Using API token: {API_TOKEN[:10]}..." if API_TOKEN else "No API token provided")

    try:
        response = requests.post(API_URL, json=request_data, headers=headers)

        # Print status code for debugging
        print(f"API response status code: {response.status_code}")

        # Save the raw response for debugging
        os.makedirs('logs', exist_ok=True)
        with open('logs/last_api_response.json', 'w') as f:
            try:
                f.write(json.dumps(response.json(), indent=2))
                print("Response saved to logs/last_api_response.json")
            except:
                f.write(response.text)
                print("Raw response text saved to logs/last_api_response.json")

        # Check for HTTP errors
        response.raise_for_status()

        # Parse the JSON response
        result = response.json()

        # Check for GraphQL errors
        if "errors" in result:
            print(f"GraphQL errors: {result['errors']}")
            return None

        return result

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response text: {response.text}")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def fetch_hero_data(save_to_file=True):
    """Fetch hero information with stats that include primaryAttribute"""
    print("\n--- Starting hero data fetch ---")

    # Check if API_TOKEN is set
    if not API_TOKEN:
        print("WARNING: No API token set. Set the STRATZ_API_TOKEN environment variable.")

    # Updated query to include stats.primaryAttribute
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

    print("Fetching hero data...")
    result = run_graphql_query(query)

    heroes = []

    if result and "data" in result and "constants" in result["data"] and "heroes" in result["data"]["constants"]:
        api_heroes = result["data"]["constants"]["heroes"]
        print(f"Successfully fetched data for {len(api_heroes)} heroes from API")

        # Process heroes to flatten the structure
        for hero in api_heroes:
            processed_hero = {
                "id": hero["id"],
                "displayName": hero["displayName"],
                "shortName": hero["shortName"]
            }

            # Extract primaryAttribute from stats if available
            if "stats" in hero and hero["stats"] and "primaryAttribute" in hero["stats"]:
                processed_hero["primaryAttribute"] = hero["stats"]["primaryAttribute"]
            else:
                # Fallback to the mapping if stats isn't available
                short_name = hero["shortName"]

            heroes.append(processed_hero)
    else:
        print("Failed to fetch hero data from API, using placeholder data")

    # Save to file if requested
    if save_to_file:
        try:
            # Ensure the data directory exists
            os.makedirs('data', exist_ok=True)

            # Write the data to the JSON file
            with open('data/heroes.json', 'w') as f:
                json.dump(heroes, f, indent=2)
            print(f"Saved {len(heroes)} heroes to data/heroes.json")
        except Exception as e:
            print(f"Error saving heroes to JSON file: {e}")

    print("--- Hero data fetch complete ---\n")
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
            image_url = f"https://cdn.dota2.com/apps/dota2/images/heroes/{valve_name}_full.png"

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
    # Implementation remains the same
    pass


if __name__ == "__main__":
    # If ran directly, fetch hero data and some matches
    print("Fetching hero data...")
    heroes = fetch_hero_data()

    print("Downloading hero portraits...")
    fetch_hero_portraits(heroes)

    print("Fetching match data...")
    matches = fetch_match_data(limit=50)

    print("Analyzing win rates...")
    # analyze_hero_winrates()
