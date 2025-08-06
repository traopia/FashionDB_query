import requests
import json
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
import pandas as pd
import os
import sys


def all_designers_vogue():
    # URL of the webpage to scrape
    url = 'https://www.vogue.com/fashion-shows/designers'  # Replace with the actual URL
    designer_list = []
    # Send a GET request to the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all the designer names
        designers = []
        for link in soup.find_all('a', class_='NavigationInternalLink-cWEaeo kHWqlu grouped-navigation__link link--primary navigation__link'):
            designers.append(link.get_text(strip=True))
        
        # Print the list of designer names
        for designer in designers:

            designer_list.append(designer)
    else:
        print(f'Failed to retrieve the webpage. Status code: {response.status_code}')
    return designer_list

def designer_to_shows(designer):
    # Replace spaces, puncuations, special character, etc. with - and make lowercase
    designer = designer.replace(' ','-').replace('.','-').replace('&','').replace('+','').replace('--','-').lower()
    designer = unidecode(designer)

    # Designer URL
    URL = "https://www.vogue.com/fashion-shows/designer/" + designer

    # Make request
    r = requests.get(URL)

    # Soupify
    soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib

    # Load a dict of the json file with the relevent data
    js = str(soup.find_all('script', type='text/javascript')[3])
    js = js.split(' = ')[1]
    js = js.split(';<')[0]
    data = json.loads(js)

    # Find the show data within the json
    try:
        t = data['transformed']
        d = t['runwayDesignerContent']
        designer_collections = d['designerCollections']
    except:
        print(f'could not find shows for {designer}')
        return []

    # Go through each show and add to list
    shows = []
    for show in designer_collections:
        shows.append(show['hed'])

    return shows



def designer_to_shows(designer):
    # Replace spaces, punctuations, special characters, etc., with '-' and make lowercase
    designer = designer.replace(' ', '-').replace('.', '-').replace('&', '').replace('+', '').replace('--', '-').lower()
    designer = unidecode(designer)

    # Designer URL
    URL = f"https://www.vogue.com/fashion-shows/designer/{designer}"

    # Make request with headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    r = requests.get(URL, headers=headers)

    if r.status_code != 200:
        print(f"Failed to fetch URL: {URL} (Status Code: {r.status_code})")
        return []

    # Soupify
    soup = BeautifulSoup(r.content, 'html.parser')

    # Find all show links
    show_elements = soup.find_all('a', {'data-testid': 'SummaryItemSimple'})

    if not show_elements:
        print(f"No shows found for {designer}")
        return []

    # Extract show names and links
    shows = []
    for element in show_elements:
        shows.append(element.text.strip())


    return shows

def modify_image_url(original_url):
    # Replace the width parameter in the URL for a higher-resolution image
    return original_url.replace("w_360", "w_1280")

def scrape_show_details(designer, show, all_urls=False):
    # Format designer and show names to match Vogue URL conventions
    show = unidecode(show.replace(' ', '-').lower())
    designer = unidecode(designer.replace(' ', '-').replace('.', '-').replace('&', '').replace('+', '').replace('--', '-').lower())

    # Construct the show URL
    url = f"https://www.vogue.com/fashion-shows/{show}/{designer}"
    print(f"Fetching: {url}")

    # Send request
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to retrieve {url}")
        return designer, show, None, None, None, None

    # Parse the page content
    soup = BeautifulSoup(r.content, 'html.parser')

    # Extract the description
    try:
        description_div = soup.find('div', class_='body__inner-container')
        description = description_div.get_text(strip=True) if description_div else None
    except Exception as e:
        print(f"Error extracting description for {url}: {e}")
        description = None

    # Extract the editor's name
    try:
        editor_span = soup.find('a', class_='BylineLink-gEnFiw')
        editor = editor_span.get_text(strip=True) if editor_span else None
    except Exception as e:
        print(f"Error extracting editor for {url}: {e}")
        editor = None

    # Extract the publish date
    try:
        date_span = soup.find('time', class_='ContentHeaderPublishDate-eIBicG')
        publish_date = date_span.get_text(strip=True) if date_span else None
    except Exception as e:
        print(f"Error extracting publish date for {url}: {e}")
        publish_date = None

    # Locate and parse JSON for image URLs
    try:
        # Locate the JSON data within the script tag
        script_tag = soup.find("script", string=re.compile(r'"runwayShowGalleries":'))
        script_content = script_tag.string if script_tag else None

        # Parse JSON data if found
        if script_content:
            json_data_match = re.search(r'"runwayShowGalleries":\s*({.*?})\s*;', script_content, re.DOTALL)
            if json_data_match:
                json_data_str = json_data_match.group(1).replace("\\u002F", "/")
                json_decoder = json.JSONDecoder()
                json_data, _ = json_decoder.raw_decode(json_data_str)

                # Extract image URLs
                galleries = json_data["galleries"]
                if all_urls:
                    image_urls = [modify_image_url(item["image"]["sources"]["sm"]["url"])
                                  for gallery in galleries for item in gallery["items"]]
                else:
                    image_urls = [modify_image_url(galleries[0]["items"][0]["image"]["sources"]["sm"]["url"])]
            else:
                image_urls = None
                print(f"No image JSON data found in {url}")
        else:
            image_urls = None
            print(f"No script tag found containing image data in {url}")
    except Exception as e:
        print(f"Error extracting image URLs for {url}: {e}")
        image_urls = None

    return designer, show, description, editor, publish_date, image_urls




def extract_details_fashion_shows(fashion_string):
    """Extract the location, season, year, and category from a fashion show string"""
    # Define the regex pattern to capture location, season, year, and optional category
    pattern = r'^([a-zA-Z-]+-)?(pre-)?(spring|summer|fall|winter|resort|bridal)-(\d{4})(-(menswear|ready-to-wear|couture))?$'
    
    # Match the pattern with the string
    match = re.match(pattern, fashion_string)
    
    if match:
        # Extract the necessary parts
        location = match.group(1)[:-1] if match.group(1) else ""  # Remove trailing hyphen if present
        if location == "":
            season = match.group(3) or match.group(2)
        else:
            season = (match.group(2) or "") + match.group(3)
        if location == 'pre':
            location = ''
            season = 'pre-fall'
        year = match.group(4)
        category = match.group(6) if match.group(6) else ""

        return location, season, year, category
    else:
        return None, None, None, None






def get_existing_collections_from_parquet(parquet_path):
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if "designer" in df.columns and "collection" in df.columns:
            return set(zip(df["designer"], df["collection"])), set(df["designer"])
    return set()

def main(out_path, all_urls, full_parquet_path="data/VogueRunway_full.parquet"):
    # Load already processed collections
    existing_collections,fashion_houses_of_interest  = get_existing_collections_from_parquet(full_parquet_path)

    for fashion_house in fashion_houses_of_interest:
        print(f"\n🔍 Scraping {fashion_house}")
        shows = designer_to_shows(fashion_house)

        for show in shows:
            # Check if this collection already exists
            if (fashion_house, show) in existing_collections:
                print(f"✅ Already in VogueRunway_full.parquet: {fashion_house} – {show}")
                continue

            fashion_house_scrape = fashion_house.lower().replace(' ', '-')

            try:
                # Scrape show details
                fashion_house_scrape, show, description, editor, publish_date, image_urls = scrape_show_details(
                    fashion_house_scrape, show, all_urls=all_urls)

                location, season, year, category = extract_details_fashion_shows(show)

                data = {
                    'fashion_house': fashion_house,
                    'show': show,
                    'collection': f"https://www.vogue.com/fashion-shows/{show}/{fashion_house_scrape}",
                    'description': description,
                    'editor': editor,
                    'publish_date': publish_date,
                    'image_urls': image_urls,
                    'location': location,
                    'season': season,
                    'year': year,
                    'category': category
                }

                # Append new data to the output file
                with open(out_path, 'a') as f:
                    json.dump(data, f)
                    f.write('\n')

                print(f"✅ Scraped and saved: {fashion_house} – {show}")
            except Exception as e:
                print(f"❌ Failed for {fashion_house} – {show}: {e}")

if __name__ == "__main__":

    main(out_path = "data/more_scraped.json", all_urls=True)


