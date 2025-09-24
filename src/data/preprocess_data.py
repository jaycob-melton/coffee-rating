import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
import argparse
from typing import Optional
import json

known_origins = [
    'north america', 
    'central america',
    'south america',
    'asia',
    'costa rica', 
    'el salvador', 
    'puerto rico',
    'united states',
    'papua new guinea',
    'dominican republic',
    'colombia',
    'ethiopia',
    'guatemala',
    'honduras',
    'indonesia',
    'nicaragua',
    'thailand',
    'tanzania',
    'vietnam',
    'mexico',
    'brazil',
    'panama',
    'hawaii', 
    'kenya',
    'rwanda',
    'china', 
    'india',
    'peru',
    'yemen',
    'burundi',
    'taiwan',
    'st. helena',
    'ecuador',
    'zambia',
    'haiti',
    'hawaii',
    'south africa',
    'uganda',
    'malaysia',
    'latin america',
    'laos',
    'congo',
    'venezuela',
    'bolivia',
    'philippines',
    'new guinea',
    'indo-pacific',
    'africa',
    'jamaica',
    'sumatra',
    'guji',
    'gedeo',
    'apaneca'
]

def extract_origins(origin_string, known):
    """
    Extracts a list of known countries or regions from a string.

    Args:
        origin_string (str): The string from the 'coffee origin' column.
        known_list (list): A list of countries to search for.

    Returns:
        list: A list of unique countries found in the string.
    """
    if not isinstance(origin_string, str):
        return ["unknown"]
    elif origin_string == "Not disclosed":
        return ["unknown"]
    
    found_origins = set()
    
    potential_origins = origin_string.replace("’", "").replace("ʻ", "").replace("'", "").replace("‘", "").split(";")
    
    for part in potential_origins:
        for origin in known:
            if re.search(r"\b" + re.escape(origin) + r"\b", part, re.IGNORECASE):
                found_origins.add(origin)
    
    if len(found_origins) == 0:
        return ["unknown"] 
    
    return sorted(list(found_origins))


PROCESS_KEYWORDS = {
    "Honey": ["honey", "pulped natural", "semi-washed"],
    "Anaerobic": ["anaerobic", "carbonic maceration", "carbonic", "experimental"],
    "Natural": ["natural", "dry", "sun dried"],
    "Washed": ["washed", "wet washed"],
}

def extract_process(notes: str) -> str:
    """
    Extracts the processing method from the notes based on predefined keywords.

    Args:
        notes (str): The notes from the coffee review.

    Returns:
        str: The processing method found in the notes, or "Unkown" if no method is found.
    """
    if not isinstance(notes, str):
        return ["unknown"]
    
    processes = set()
    for process, keywords in PROCESS_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", notes, re.IGNORECASE):
            # if keyword in notes.lower():
                processes.add(process.lower())
    
    if len(processes) == 0:
        return ["unknown"]
    
    return list(processes)


FLAVOR_KEYWORDS = {
    # Positive Flavors
    'Fruity': [
        'strawberry', 'raspberry', 'blueberry', 'blackberry', 'marionberry', 'cranberry', # Berry
        'raisin', 'prune', 'date', 'fig', # Dried Fruit
        'grapefruit', 'orange', 'lemon', 'lime', 'tangerine', 'bergamot', 'zest', # Citrus
        'cherry', 'black cherry', 'peach', 'apricot', 'plum', 'nectarine', # Stone Fruit
        'pineapple', 'mango', 'passion fruit', 'guava', 'lychee', 'kiwi', 'papaya', 'coconut', # Tropical
        'apple', 'red apple', 'green apple', 'pear', 'grape', 'concord grape', 'white grape', 'melon', 'pomegranate', # Other Fruit
        'fruit'
    ],
    'Floral': [
        'jasmine', 'rose', 'hibiscus', 'lavender', 'chamomile', 'honeysuckle', 'orange blossom', 'elderflower', # Aromatic Flowers
        'floral'
    ],
    'Sweet': [
        'molasses', 'maple syrup', # Syrupy
        'brown sugar', 'caramel', 'butterscotch', 'toffee', 'nougat', 'marshmallow', 'cane sugar', # Sugars
        'vanilla', 'cream', 'custard', 'marzipan', # Confectionary
        'honey', 'honeydew', # Honey
        'sweet'
    ],
    'Nutty/Cocoa': [
        'almond', 'hazelnut', 'peanut', 'walnut', 'pecan', 'cashew', 'praline', # Nutty
        'chocolate', 'milk chocolate', 'dark chocolate', 'baker\'s chocolate', 'cacao nibs', # Cocoa
        'nutty', 'cocoa'
    ],
    'Spicy': [
        'cinnamon', 'nutmeg', 'clove', 'cardamom', 'allspice', 'gingerbread', # Baking Spices
        'anise', 'licorice', 'black pepper', 'coriander', 'ginger', # Pungent Spices
        'spice'
    ],
    'Roasted/Toasted': [
        'grain', 'malt', 'oatmeal', 'toast', 'sourdough bread', 'brown rice', # Cereal
        'smoke', 'ash', 'acrid', 'burnt sugar', 'char', # Burnt
        'pipe tobacco', 'cigar', 'cedar', # Tobacco
        'roasted', 'toasted'
    ],
    'Earthy/Herbal': [
        'fresh-cut grass', 'hay', 'bell pepper', 'olive', 'tomato', 'pea', # Green/Vegetative
        'black tea', 'green tea', 'oolong tea', 'mint', 'thyme', 'lemongrass', 'rooibos', # Herbal/Tea-like
        'earthy', 'herbal', 'tea-like'
    ],
    'Winey/Fermented': [
        'red wine', 'white wine', 'champagne', 'boozy', 'whiskey', 'rum', # Winey/Alcoholic
        'fermented'
    ],
    'Savory': ['umami', 'soy sauce', 'leather', 'meaty'],

    # Other Attributes
    'Mouthfeel': [
        'light-bodied', 'medium-bodied', 'full-bodied', 'heavy', 'thin', 'watery', # Body
        'creamy', 'buttery', 'silky', 'smooth', 'juicy', 'syrupy', 'velvety', 'rich', 'delicate', 'astringent', 'gritty' # Texture
    ],
    'Acidity': [
        'mild', 'soft', 'mellow', 'delicate', # Intensity
        'bright', 'crisp', 'tart', 'tangy', 'vibrant', 'lively', 'sparkling', 'effervescent', 'complex', 'structured' # Quality
    ],
    'Aftertaste': ['clean', 'lingering', 'long', 'quick', 'short', 'dry', 'sweet', 'cloying'],
    
    # Negative Flavors/Defects
    'Defect/Negative': [
        'earthy', 'damp soil', 'mushroom', 'musty', 'moldy', # Earthy/Musty
        'rubbery', 'petroleum', 'plastic', 'bitter medicine', 'iodine', 'phenolic', # Chemical
        'cardboard', 'paper', 'stale', 'woody', 'sawdust', # Woody/Papery
        'over-fermented', 'sour', 'vinegary', 'alcoholic', # Fermented Taints
        'grassy', 'beany', 'under-ripe', # Green/Unripe
        'baggy', 'hidy', 'scorched', 'tipped' # Other
    ]
}

def extract_structured_profile(text: str) -> dict:
    """
    Extracts a dictionary of flavor categories mapped to the specific
    keywords found in the text.
    """
    if not isinstance(text, str):
        return {}
    
    found_profile = {}
    lower_text = text.lower()
    
    for category, keywords in FLAVOR_KEYWORDS.items():
        found_keywords = []
        for keyword in keywords:
            # Use regex with word boundaries (\b) to avoid matching parts of words
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower_text):
                found_keywords.append(keyword)
        
        if found_keywords:
            found_profile[category] = sorted(list(set(found_keywords)))
                
    return found_profile

def format_profile_for_llm(profile_dict: dict) -> str:
    if not profile_dict:
        return ""
    parts = []
    for category, notes in profile_dict.items():
        notes_str = ", ".join(notes)
        parts.append(f"{category} ({notes_str})")
    return "; ".join(parts)


COFFEE_VARIETALS = [
    # Common Arabica
    'typica', 'bourbon', 'caturra', 'catuai', 'geisha', 'gesha', 'pacamara', 
    'pacas', 'maragogipe', 'mundo novo', 'kent', 's795', 'jember', 'villa sarchi',
    'sl28', 'sl34', 'sl14', 'batian', 'ruiru 11', 'blue mountain', 'sumatra',
    'timor hybrid', 'hibrido de timor', 'catimor', 'castillo', 'colombia',
    'sarchimor', 'ihcafe 90', 'lempira', 'parainema', 'centroamericano',
    'mokka', 'mocha', 'java', 'kona', 'yellow bourbon', 'red bourbon', 
    'pink bourbon', 'orange bourbon', 'yellow caturra', 'red caturra',
    'yellow catuai', 'red catuai', 'maracaturra', 'wush wush', 'sidra', 'sudanese rumé',
    
    # Ethiopian Heirlooms (often referred to as a group)
    'heirloom', 'ethiopian heirloom', 'kurume', 'daga', 'wolisho',

    # Common Robusta
    'robusta', 'congensis', 'canephora',

    # Other Species
    'liberica', 'excelsa', 'stenophylla', 'arabica' # Include base species
]

def extract_varietals(text: str) -> list:
    """
    Extracts a list of coffee varietals from a text string.
    """
    if not isinstance(text, str):
        return ["unkown"]
    
    found_varietals = set()
    lower_text = text.lower()
    
    for varietal in COFFEE_VARIETALS:
        # Use regex with word boundaries (\b) to avoid matching parts of words.
        # This handles cases like 'java' not matching in 'javascript'.
        # re.escape handles special characters in the varietal name.
        if re.search(r'\b' + re.escape(varietal) + r'\b', lower_text, re.IGNORECASE):
            # Standardize spellings (e.g., Gesha/Geisha)
            if varietal == 'gesha':
                found_varietals.add('geisha')
            elif varietal == 'hibrido de timor':
                found_varietals.add('timor hybrid')
            else:
                found_varietals.add(varietal)
    if len(found_varietals) == 0:
        return ["unkown"]
                
    return sorted(list(found_varietals))


def standardize_pricing(price_string: str) -> Optional[float]:
    """
    Standardizes a coffee price string to USD per ounce.

    Args:
        price_string: A string representing the coffee price,
                      e.g., '$xx.xx/y [ounces or grams]' or
                      'NT $xx.xx/y [ounces or grams]'.

    Returns:
        The price per ounce in USD as a float, or None if the
        string is not in the expected format.
    """
    if not isinstance(price_string, str):
        return None
    
    pattern = re.compile(r"""
        (?P<currency>NT\s\$|\$)? # optional currency (NT or USD)
        \s* # optional whitespace
        (?P<price>[\d,]+(?:\.\d+)?) # the price
        / # seperator
        (?P<weight>\d+) # weight
        \s* # optional whitespace
        (?P<unit>ounces|grams)                     
    """, re.VERBOSE | re.IGNORECASE)
    
    match = pattern.match(price_string)
    
    if not match:
        return None
    
    parts = match.groupdict()
    
    price = float(parts["price"].replace(",", ""))
    weight = float(parts["weight"])
    unit = parts["unit"].lower()
    currency = "USD" if parts["currency"] == "$" else "NT"
    
    if unit == 'grams':
        weight_oz = weight / 28.34952
    else:
        weight_oz = weight
        
    if weight_oz == 0:
        return None
    
    # 6/9/25
    ntd_to_usd_rate = 29.9340
    price_usd = price
    
    if currency == "NT":
        price_usd /= ntd_to_usd_rate
    
    price_per_oz = price_usd / weight_oz
    
    return price_per_oz


def calculate_yearly_tiers(price_series: pd.Series) -> pd.Series:
    """
    Calculates price tiers within a single year, into 4 quartiles
    """
    return pd.qcut(
        price_series,
        q=4,
        labels=["value", "standard", "premium", "luxury"],
        duplicates="drop"
    )


NUMERICAL_COLS_TO_NORMALIZE = [
    'rating', 'aroma', 'acidity', 'body', 'flavor', 'aftertaste', 
    'with milk', 'agtron_1', 'agtron_2', 'price_per_oz'
]

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    # load data
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.replace(".", "", regex=False)
    # extract origins
    df["coffee origin"] = df["coffee origin"].str.lower()
    df["countries_extracted"] = df["coffee origin"].apply(lambda origin: extract_origins(origin, known_origins))

    # extract test method
    is_espresso = df["blind assessment"].str.contains("espresso", case=False, regex=True).fillna(False)
    is_cold = df["blind assessment"].str.contains("cold", case=False, regex=True).fillna(False)
    # Condition 2: Check if 'with milk' is not null
    has_milk = df["with milk"].notnull()
    # espresso cases
    df["test_method"] = np.where(
        is_espresso & ~has_milk, 
        "espresso_black",
        np.where(
            is_espresso & has_milk, 
            "espresso_with_milk",
            np.where(
                is_cold & ~has_milk, 
                "cold_black",
                np.where(
                    is_cold & has_milk, 
                    "cold_with_milk",
                    np.where(
                        ~is_espresso & ~is_cold & ~has_milk, 
                        "hot_black",
                        np.where(
                            ~is_espresso & ~is_cold & has_milk, 
                            "hot_with_milk", 
                            "unknown"
                        )
                    )
                )
            )
        )
    )

    # extract process
    df["process"] = df["notes"].apply(extract_process)

    # extract flavor notes and llm string for synthetic data
    df['flavor_profile'] = df['blind assessment'].apply(extract_structured_profile)
    df['flavor_profile_str'] = df['flavor_profile'].apply(format_profile_for_llm)
    
    # extract coffee varietal
    df['varietals'] = df['notes'].apply(extract_varietals)

    # split agtron scores
    df[["agtron_1", "agtron_2"]] = df["agtron"].str.split("/", expand=True)
    
    # standardize pricing
    df["price_per_oz"] = np.round(df["est price"].apply(standardize_pricing), 2)
    # provide relative pricing scale
    df["review date"] = pd.to_datetime(df["review date"], errors='coerce')
    df["year"] = df["review date"].dt.year.astype(int)
    df["price_tier"] = df.groupby("year")["price_per_oz"].transform(calculate_yearly_tiers).reset_index(level=0, drop=True)

    # normalize numerical columns
    for col in NUMERICAL_COLS_TO_NORMALIZE:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[NUMERICAL_COLS_TO_NORMALIZE] = df[NUMERICAL_COLS_TO_NORMALIZE].fillna(0)

    scaler = MinMaxScaler()

    df_normalized = df.copy()
    df_normalized[NUMERICAL_COLS_TO_NORMALIZE] = scaler.fit_transform(df_normalized[NUMERICAL_COLS_TO_NORMALIZE])
    

    print(f"Writing preprocessed data to {output_path}")
    df_normalized.to_csv(output_path)
    return df_normalized

def dump_lists_n_dicts_json():
    """
    Dumps the known origins, flavor keywords, and varietals to JSON files.
    """
    with open("data/universal/known_origins.json", "w") as f:
        json.dump(known_origins, f, indent=4)
    
    with open("data/universal/flavor_keywords.json", "w") as f:
        json.dump(FLAVOR_KEYWORDS, f, indent=4)
    
    with open("data/universal/coffee_varietals.json", "w") as f:
        json.dump(COFFEE_VARIETALS, f, indent=4)

    with open("data/universal/process_keywords.json", "w") as f:
        json.dump(PROCESS_KEYWORDS, f, indent=4)
    

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw coffee review dataset for training")
    parser.add_argument("input_file", type=str, help="Input file with existing dataset to append to")
    parser.add_argument("output_file", type=str, help="Output file name for the dataset")
    args = parser.parse_args()
    preprocess_data(args.input_file, args.output_file)
    # dump_lists_n_dicts_json()

if __name__ == "__main__":
    main()