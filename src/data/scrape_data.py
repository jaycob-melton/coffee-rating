import pandas as pd
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import argparse
from typing import Optional

def get_review_links(base: str, params: str, pages: Optional[int] = None) -> list:
    """
    Scrapes review links from a page of advanced search results on coffeereview.com
    """
    page = 1
    review_links = []
    print("Scraping review links...")
    while pages is not None and page <= pages:
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


def parse_review(url: str) -> Optional[dict]:
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
                data[clean_label] = content_element.get_text(strip=True)
    
    h2_labels = ["Blind Assessment", "Notes", "Bottom Line", "The Bottom Line", "Who Should Drink It"]

    for h2_str in h2_labels:
        content = soup.find("h2", string=h2_str)
        if content:
            content_element = content.find_next_sibling()
            if content_element:
                # print(content_element)
                if h2_str in ["Bottom Line", "The Bottom Line", "Who Should Drink It"]:
                    data["Bottom Line"] = content_element.get_text(strip=True)
                else:
                    data[h2_str] = content_element.get_text(strip=True)

    return data


def parse_all_reviews(review_links: list, num_workers: Optional[int] = 1) -> list:
    all_reviews = []
    MAX_WORKERS = num_workers if num_workers > 0 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(parse_review, url): url for url in review_links}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(review_links), desc="Parsing Reviews"):
            review = future.result()
            if review:
                all_reviews.append(review)
    return all_reviews

def dump_to_csv(data: dict, output_file: str, input_file: Optional[str] = None) -> None:
    """
    Dumps the scraped data to a CSV file
    """
    if data:
        df = pd.DataFrame(data)
        if input_file:
            try:
                existing_data = pd.read_csv(input_file)
                print(f"Loaded existing dataset with {len(existing_data)} reviews.")
                df = pd.concat([existing_data, df], ignore_index=True)
            except FileNotFoundError:
                print(f"Input file {input_file} not found.")
        
        # basic preprocessing
        df.columns = df.columns.str.strip().str.lower()
        df["review date"] = pd.to_datetime(df["review date"], errors='coerce')
        df = df.sort_values(by="review date", ascending=False)
        # our true unique id is a particular coffee, and we only keep the newest review 
        df["id"] = df["company"] + "_" + df["coffee name"]
        df = df.drop_duplicates(subset=["id"], keep="first")
        try:
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} reviews to {output_file}")
        except Exception as e:
            print(f"Error saving to {output_file}: {e}")
            print("Saving to output.csv instead.")
            df.to_csv("output.csv", index=False)
    else:
        print("No reviews successfully parsed.")


def main():
    # get optional existing dataset
    # get output file name
    # get scrape type: all, recent, or specific number of pages 
    parser = argparse.ArgumentParser(description="Scrape coffee reviews from CoffeeReview.com")
    parser.add_argument("output_file", type=str, help="Output file name for the dataset")
    parser.add_argument("--input_file", type=str, default=None, help="Input file with existing dataset to append to")
    parser.add_argument("--scrape_type", type=str, choices=["all", "pages"], default="all", help="Type of scrape to perform")
    parser.add_argument("--num_pages", type=int, default=5, help="Number of pages to scrape if scrape_type is 'pages'")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers for scraping")
    args = parser.parse_args()


    base = "https://www.coffeereview.com/advanced-search/"
    params = (
        "?keyword=&search=Search+Now&locations=all&score_all"
        "=on&score_96_100=on&score_93_95=on&score_90_92=on&score_85_89=on&score_85=on"
    )

    review_links = get_review_links(base, params, pages=args.num_pages if args.scrape_type == "pages" else None)
    print(f"Found {len(review_links)} total reviews to parse.")
    if not review_links:
        print("No reviews found. Exiting.")
        return
    
    all_reviews = parse_all_reviews(review_links, num_workers=args.workers)

    dump_to_csv(all_reviews, args.output_file, input_file=args.input_file)


if __name__ == "__main__":
    main()


