import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import concurrent.futures
from tqdm import tqdm
import re

def parse_review(url):
    """
    Parses a single coffee review page
    """
    try:
        with requests.Session() as session:
            response = requests.get(url, timeout=10)
            # raise exception if bad url
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        # print(f"Could not fetch URL {url}: {e}")
        return None
    
    data = {"URL": url}
    # first scrape the first table
    # first table contains raiting, company, and coffee name
    content = soup.find("span", class_="review-template-rating")
    data["Rating"] = content.get_text().strip()
    content = soup.find("p", class_="review-roaster")
    data["Company"] = content.get_text().strip()
    content = soup.find("h1", class_="review-title")
    data["Coffee Name"] = content.get_text().strip()
    
    # scrape second table, containing coffee information
    # contains roaster location, origin, roast, agtron, price
    

    td_labels = [
        "Roaster Location:", "Coffee Origin:", "Roast Level:", "Agtron:",
        "Est. Price:", "Review Date:", "Aroma:", "Acidity/Structure:", "Acidity:",
        "Body:", "Flavor:", "Aftertaste:", "With Milk:", "Flavor in milk:"
    ]

    for td_str in td_labels:
        content = soup.find("td", string=td_str)
        if content:
            content_element = content.find_next_sibling()
            if content_element:
                # text = content_element.get_text(strip=True)
                # if td_str == "Coffee Origin:":
                #     print(content_element)
                #     print(text)
                if td_str in ["Acidity/Structure:", "Acidity:"]:
                    clean_label = "Acidity"
                elif td_str in ["With Milk:", "Flavor in milk:"]:
                    clean_label = "With Milk"
                else:
                    clean_label = td_str.replace(":", "")
                data[clean_label] = content_element.get_text(strip=True)#.replace(";", "+")
                # data[td_str] = data[td_str].str.replace("+", ";")
    
    h2_labels = ["Blind Assessment", "Notes", "Bottom Line", "The Bottom Line", "Who Should Drink It"]

    for h2_str in h2_labels:
        content = soup.find("h2", string=h2_str)
        if content:
            content_element = content.find_next_sibling()
            if content_element:
                # print(content_element)
                if h2_str in ["Bottom Line", "The Bottom Line", "Who Should Drink It"]:
                    data["Bottom Line"] = content_element.get_text(strip=True)#.replace(";", "+")
                else:
                    data[h2_str] = content_element.get_text(strip=True)#.replace(";", "+")

    # print(content.get_text())
    # content = soup.find("div", class_="entry-content")
    # if not content:
    #     print(f"Not content found at URL {url}")
    #     return None
    
    # lines = content.get_text(separator="\n").split("\n")
    # lines = [line.strip() for line in lines if line.strip()]

    # data = {"URL": url}

    # KNOWN_LABELS = {
    #     "Roaster Location:", "Coffee Origin:", "Roast Level:", "Agtron:",
    #     "Est. Price:", "Review Date:", "Aroma:", "Acidity/Structure:", "Acidity:",
    #     "Body:", "Flavor:", "Aftertaste:", "With Milk:",
    #     # "Blind Assessment", "Notes", "Bottom Line", "Who Should Drink It"
    # }

    # try:
    #     data["Rating"] = lines[0]
    #     data["Company"] = lines[1]
    #     data["Coffee Name"] = lines[2]
    # except IndexError:
    #     print(f"Failed to parse initial block for {url}")
    #     return None
    

    

    # find indices of all labels
    # label_idx = {}
    # for i, line in enumerate(lines):
    #     if line in KNOWN_LABELS or line.startswith("Bottom Line") or line.startswith("Who Should Drink It"):
    #         label_idx[line] = i

    # if not label_idx:
    #     return data
    
    # sorted_labels = sorted(label_idx.items(), key=lambda x: x[1])

    # for i, (current_label, start_index) in enumerate(sorted_labels):
    #     # determine the end of the current sections content
    #     end_index = sorted_labels[i+1][1] if i + 1 < len(sorted_labels) else len(lines)

    #     # set value to lines between start to end
    #     value_lines = lines[start_index + 1 : end_index]
    #     value = " ".join(value_lines).strip()

    #     if current_label in ["Bottom Line", "Who Should Drink It"]:
    #         clean_label = "Bottom Line"
    #     elif current_label == "Acidity/Structure:":
    #         clean_label = "Acidity"
    #     else:
    #         clean_label = current_label.replace(":", "")
        
    #     if clean_label in ["Coffee Origin", "Roaster Location"]:
    #         data[clean_label] = [place.strip() for place in value.split(";") if place.strip()]
    #     else:
    #         data[clean_label] = value

    # Find Notes
    # notes_h2 = content.find("h2", string="Notes")
    # if notes_h2:
    #     content_element = notes_h2.find_next_sibling()
    #     if content_element:
    #         data["Notes"] = content_element.get_text(strip=True)

    # # Find Bottom Line (handles multiple possible header texts)
    # bottom_line_h2 = content.find("h2", string=["Bottom Line", "Who Should Drink It", "The Bottom Line"])
    # if bottom_line_h2:
    #     content_element = bottom_line_h2.find_next_sibling()
    #     if content_element:
    #         data["Bottom Line"] = content_element.get_text(strip=True)

    # i = 3
    # while i < len(lines):
    #     line = lines[i]

    #     # check if current line is a known label
    #     if line in KNOWN_LABELS:
    #         current_label = line
    #         value_start_index = i + 1
    #         value_lines = []

    #         # capture all lines until next label is found
    #         j = value_start_index
    #         while j < len(lines) and lines[j] not in KNOWN_LABELS:
    #             value_lines.append(lines[j])
    #             j += 1
            
    #         value = " ".join(value_lines).strip()
    #         if current_label == "Coffee Origin:":
    #             origins = [origin.strip() for origin in value.split(";")]
    #             data["Coffee Origin"] = origins
    #         elif current_label == "Roaster Location:":
    #             roasters = [roaster.strip() for roaster in value.split(";")]
    #             data["Roaster Location"] = roasters
    #         elif current_label == "Bottom Line" or current_label == "Who Should Drink It":
    #             data["Bottom Line"] = value
    #         else:
    #             clean_label = current_label.replace(":", "")
    #             data[clean_label] = value
    #         # move the main index to the start of the next  block
    #         i = j
    #     else:
    #         i += 1

    return data


def get_review_links(base, params):
    page = 1
    review_links = []
    print("Scraping review links...")
    while True: # set to True for full scrape
        if page == 1:
            url = base + params
        else:
            url = f"{base}page/{page}/{params}"
        
        print(f"Scanning page for links: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Could not fetch page {page}. Stopping. Error: {e}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        titles = soup.find_all("h2", class_="review-title")
        if not titles:
            print("No more review links found. Ending scan.")
            break
    
        for h2 in titles:
            a_tag = h2.find("a", href=True)
            if a_tag:
                review_links.append(a_tag["href"])
        
        page += 1

    return review_links   

def fetch_and_parse_links_from_page(url):
    """Helper function to fetch one page and return the review links."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        titles = soup.find_all("h2", class_="review-title")
        links = [h2.find('a')['href'] for h2 in titles if h2.find('a')]
        return links
    except requests.exceptions.RequestException:
        return [] # Return empty list on failure


def get_review_links_parallel(base, params):
    """
    Finds all review links by first determining the total number of pages
    and then fetching all pages in parallel.
    """
    print("Determining total number of pages...")
    first_page_url = base + params
    
    try:
        response = requests.get(first_page_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch the first page to determine pagination. Error: {e}")
        return []

    max_pages = 148
    # Find the "Last »" link in the pagination controls
    last_link = soup.select_one('a.page-numbers[href*="/page/"]')
    if last_link:
        # Use regex to find the highest page number from any of the pagination links
        all_page_links = soup.select('a.page-numbers[href*="/page/"]')
        page_numbers = [int(re.search(r'/page/(\d+)/', link['href']).group(1)) for link in all_page_links if re.search(r'/page/(\d+)/', link['href'])]
        if page_numbers:
            max_pages = max(page_numbers)

    print(f"Found {max_pages} pages of reviews.")

    # Generate all page URLs
    page_urls = [first_page_url]
    for page_num in range(2, max_pages + 1):
        page_urls.append(f"{base}page/{page_num}/{params}")

    # Fetch all pages in parallel
    all_links = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_url = {executor.submit(fetch_and_parse_links_from_page, url): url for url in page_urls}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(page_urls), desc="Fetching review links"):
            links = future.result()
            if links:
                all_links.extend(links)
    
    return all_links

if __name__ == "__main__":
    print("---10 Page Scrape Mode---")
    base = "https://www.coffeereview.com/advanced-search/"
    params = (
        "?keyword=&search=Search+Now&locations=all&score_all"
        "=on&score_96_100=on&score_93_95=on&score_90_92=on&score_85_89=on&score_85=on"
    )
    #results
    # url = "https://www.coffeereview.com/review/motif-morning-blend/"
    # review = parse_review(url)
    # print(review)

    review_urls = get_review_links(base, params) #get_review_links_parallel(base, params) 
    print(f"Found {len(review_urls)} total reviews to parse.")
    all_reviews = []
    # for url in review_urls:
    #     print(f"Scraping review: {url}")
    #     review = parse_review(url)
    #     if review:
    #         all_reviews.append(review)
    MAX_WORKERS = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(parse_review, url): url for url in review_urls}

        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(review_urls), desc="Parsing Reviews"):
            review = future.result()
            if review:
                all_reviews.append(review)

    
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        # df["Roaster Location"] = df["Roaster Location"].str.replace("+", ";")
        # df["Coffee Origin"] = df["Coffee Origin"].str.replace("+", ";")
        # df["Blind Assessment"] = df["Blind Assessment"].str.replace("+", ";")
        # df["Notes"] = df["Notes"].str.replace("+", ";")
        # df["Bottom Line"] = df["Bottom Line"].str.replace("+", ";")
        df.to_csv("reviews_to_2020.csv", index=False)
        print("reviews_to_2020.csv")
    else:
        print("No reviews successfully parsed.")
        

        #     time.sleep(0.5)
    