import asyncio
import aiohttp
import pandas as pd
import re
import json
import codecs

# --- CONFIGURATION ---
SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
LEAGUE = "EPL"

# Disguise as a standard Chrome Browser on Mac
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}
# ---------------------

async def fetch_html(session, url):
    try:
        async with session.get(url, headers=HEADERS) as response:
            if response.status != 200:
                print(f"  [!] Blocked or Error: HTTP {response.status}")
                return None
            return await response.text()
    except Exception as e:
        print(f"  [!] Error fetching {url}: {e}")
        return None

def extract_json(html_content, var_name):
    # Improved Regex: Handles extra spaces (var  datesData = ...)
    # Looks for: var name = JSON.parse('...');
    pattern = re.compile(r"var\s+" + var_name + r"\s*=\s*JSON\.parse\('([^']+)'\)")
    match = pattern.search(html_content)
    
    if match:
        try:
            # Understat data is hex-encoded. We must decode it.
            hex_string = match.group(1)
            # Decode using raw_unicode_escape to fix the hex strings
            decoded_string = codecs.decode(hex_string, 'unicode_escape')
            return json.loads(decoded_string)
        except Exception as e:
            print(f"  [!] Regex matched but JSON failed: {e}")
            return None
    return None

async def process_match(session, match_id, season_label):
    url = f"https://understat.com/match/{match_id}"
    html_text = await fetch_html(session, url)
    if not html_text:
        return []

    shots_data = extract_json(html_text, "shotsData")
    if not shots_data:
        return []

    processed_shots = []
    # Shots data contains 'h' (home) and 'a' (away) lists
    try:
        for side in ['h', 'a']:
            for shot in shots_data[side]:
                processed_shots.append({
                    'season': season_label,
                    'match_id': match_id,
                    'date': shot.get('date'),
                    'player': shot['player'],
                    'player_id': shot.get('player_id'),
                    'team': shot['h_team'] if side == 'h' else shot['a_team'],
                    'minute': shot['minute'],
                    'result': shot['result'],
                    'X': float(shot['X']),
                    'Y': float(shot['Y']),
                    'xG': float(shot['xG']),
                    'shotType': shot['shotType'],
                    'situation': shot['situation'],
                    'lastAction': shot.get('lastAction')
                })
    except Exception as e:
        print(f"Error parsing match {match_id}: {e}")
        
    return processed_shots

async def main():
    print(f"Starting Direct Scraping for {len(SEASONS)} seasons...")
    all_shots = []
    
    # SSL Fix for Mac
    connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        for season in SEASONS:
            print(f"Processing {season}/{season+1}...")
            
            season_url = f"https://understat.com/league/{LEAGUE}/{season}"
            season_html = await fetch_html(session, season_url)
            
            if not season_html:
                continue

            dates_data = extract_json(season_html, "datesData")
            
            if not dates_data:
                # Save the failed HTML to check later if needed
                with open("debug_fail.html", "w", encoding="utf-8") as f:
                    f.write(season_html)
                print(f"  -> Could not find match list. (HTML saved to debug_fail.html)")
                continue

            match_ids = [match['id'] for match in dates_data if match['isResult']]
            print(f"  -> Found {len(match_ids)} matches. Scraping shots (Batching)...")
            
            # Process in small batches to be safe
            tasks = []
            for i, match_id in enumerate(match_ids):
                tasks.append(process_match(session, match_id, f"{season}/{season+1}"))
                
                # Run 20 matches at a time
                if len(tasks) >= 20:
                    results = await asyncio.gather(*tasks)
                    for res in results:
                        all_shots.extend(res)
                    tasks = []
                    print(f"     ...scraped {len(all_shots)} shots so far")
                    
            # Finish remaining
            if tasks:
                results = await asyncio.gather(*tasks)
                for res in results:
                    all_shots.extend(res)

    # Save Data
    if all_shots:
        df = pd.DataFrame(all_shots)
        filename = "Premier_League_Shots_14_24_Direct.csv"
        df.to_csv(filename, index=False)
        print(f"\nSUCCESS! Saved {len(df)} shots to {filename}")
        print(df.head())
    else:
        print("\nFailed to scrape any data.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())