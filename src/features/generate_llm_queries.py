import pandas as pd
import openai
import time
import json
import re
import os
from tqdm import tqdm


try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()


def create_profile_str(row):
    profile = "[COFFEE PROFILE]\n"
    profile += f"- Origin: {row['countries_extracted']}\n"
    profile += f"- Varietal(s): {row['varietals']}\n"
    profile += f"- Processing: {row['process']}\n"
    profile += f"- Roast Level: {row['roast level']}\n"
    profile += f"- Price Tier: {row['price_tier']}\n"
    profile += f"- Test Method: {row['test_method']}\n"
    profile += f"- Scores: Rating ({row['rating']}), "
    profile = profile + f"Aroma ({row['aroma']}), " if row["aroma"] != 0. else profile
    profile = profile + f"Acidity ({row['acidity']}), " if row["acidity"] != 0. else profile
    profile = profile + f"Body ({row['body']}), " if row["body"] != 0. else profile
    profile = profile + f"Flavor ({row['flavor']}), " if row["flavor"] != 0. else profile
    profile = profile + f"Aftertaste ({row['aftertaste']}), " if row["aftertaste"] != 0. else profile
    profile = profile + f"With Milk ({row['with milk']}), " if row["with milk"] != 0. else profile
    profile += "\n"
    profile += f"- Tasting Notes: {row['flavor_profile_str']}\n"
    profile += f"- Expert Blind Assessment: {row['blind assessment']}\n"
    profile += f"- Expert Summary: {row['bottom line']}\n"
    return profile


def generate_queries(coffee_profile_str):
    messages = [
        {
            "role": "system",
            "content": """
                You are an expert assistant for a coffee recommendation system. Your task is to generate 6 distinct, realistic user search queries for the provided coffee profile. The queries should reflect different ways a user might ask for this specific coffee. Include queries that range from highly specific, using expert terminology, to very general or novice-level questions. Each query should be distinct and reflect a different user persona or level of knowledge.
                **Instructions:**
                1.  Generate exactly 6 queries.
                2.  Each query should be on a new line, starting with "Q:".
                3.  The queries should be diverse: one might focus on flavor, another on origin or varietal, another on price or occasion, another on its strengths for a particular brew method, and one can be a general summary.
                4.  Use the "Expert Summary" to guide the tone and key selling points. Use the "Expert Blind Assessment" for more nuanced flavor/notes details.
            """
        },
        {
            "role": "user",
            "content": """
                [COFFEE PROFILE]
                - Origin: Ethiopia
                - Varietal(s): Geisha
                - Processing: Washed
                - Roast Level: Light
                - Price Tier: Luxury
                - Test Method: hot_black
                - Scores: Rating (0.96), Acidity (0.9), Body (0.8)
                - Tasting Notes: Floral (jasmine, bergamot), Fruity (apricot), Sweet (honey)
                - Blind Assessment: Delicately structured, floral-toned. Jasmine, bergamot, apricot, honey, a hint of cocoa nib in aroma and cup.
                - Expert Summary: A classic, elegant, washed Ethiopia Geisha: balanced, bright, floral, and exceptionally clean.
            """
        },
        {
            "role": "assistant",
            "content": """
                [GENERATED QUERIES]
                Q: A bright, clean, and floral coffee.
                Q: What's a good light roast for pour-over?
                Q: Show me your highest-rated Ethiopia coffees.
                Q: Looking for a washed Geisha with strong jasmine and bergamot notes.
                Q: A complex, high-acidity cup.
                Q: Something elegant for a special occasion.
            """
        },
        {
            "role": "user",
            "content": """
                [COFFEE PROFILE]
                - Origin: Colombia
                - Varietal(s): Pink Bourbon
                - Processing: Anaerobic, Natural
                - Roast Level: Medium-Light
                - Price Tier: Premium
                - Test Method: hot_black
                - Scores: Rating (0.94), Acidity (0.8), Body (0.9)
                - Tasting Notes: Winey/Fermented (cherry liqueur, rum), Spicy (cinnamon), Sweet (dark chocolate)
                - Blind Assessment: Richly aromatic, fruit-forward. Cherry liqueur, dark chocolate, cinnamon, cedar, rum cake in aroma and cup.
                - Expert Summary: A deeply sweet, complex, and resonant cup that confidently expresses its anaerobic natural processing.
            """   
        },
        {
            "role": "assistant",
            "content": """
                [GENERATED QUERIES]
                Q: A wild, funky, and complex experimental coffee.
                Q: Something that tastes like cherry liqueur and rum cake.
                Q: Show me your anaerobic natural coffees from Colombia.
                Q: A sweet and spicy coffee with a heavy body.
                Q: Looking for an interesting Pink Bourbon varietal.
                Q: A coffee for someone who likes bold, winey flavors.
            """
        },
        {
            "role": "user",
            "content": """
                [COFFEE PROFILE]
                - Origin: Brazil, Guatemala
                - Varietal(s): Catuai, Bourbon
                - Processing: Natural, Washed
                - Roast Level: Medium
                - Price Tier: Value
                - Test Method: espresso_with_milk
                - Scores: Rating (0.88), Acidity (0.6), Body (0.8)
                - Tasting Notes: Nutty/Cocoa (almond, chocolate), Sweet (caramel)
                - Blind Assessment: Gently bright, balanced. Almond, caramel, a hint of baking chocolate, and a touch of orange zest.
                - Expert Summary: A balanced, straightforward, and satisfying blend. Very sweet, with classic chocolate and nut notes. Great with milk.
            """
        },
        {
            "role": "assistant",
            "content": """
                [GENERATED QUERIES]
                Q: A smooth, not bitter coffee that's good with milk.
                Q: What's a good, affordable coffee for my espresso machine?
                Q: Something with chocolate and almond notes.
                Q: A solid, everyday medium roast.
                Q: A balanced blend from Central and South America.
                Q: A good value coffee for a beginner.
            """
        },
        {
            "role": "user",
            "content": coffee_profile_str # This is the new coffee we want queries for
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.2,
            top_p=1.0,
            max_tokens=200,
            stop=["---"]
        )

        generated_text = response.choices[0].message.content

        queries = re.findall(r"Q:\s*(.*)", generated_text)
        return [q.strip() for q in queries]
    except openai.APIError as e:
        print(f"An OpenAI API error occured: {e}")
    except Exception as e:
        print(f"An unexpected error occured: {e}")

    return []


if __name__ == "__main__":
    # example_coffee_profile = """
    #     [COFFEE PROFILE]
    #     - Origin: Ethiopia
    #     - Varietal(s): Geisha
    #     - Processing: Washed
    #     - Roast Level: Light
    #     - Price Tier: Luxury
    #     - Tasting Notes: Fruity (bergamot, cherry), Floral (jasmine), Sweet (honey), Mouthfeel (tea-like)
    #     - Expert Summary: A complex, vibrant, and deeply floral cup that showcases the classic virtues of a top-tier washed Ethiopia Geisha.
    # """

    # print("--- Generating queries for a sample coffee profile ---")
    # print(example_coffee_profile)

    # start_time = time.time()
    # synthetic_queries = generate_queries(example_coffee_profile)
    # end_time = time.time()

    # print("\n--- Generated Queries ---")
    # if synthetic_queries:
    #     for i, query in enumerate(synthetic_queries, 1):
    #         print(f"{i}. {query}")
    # else:
    #     print("Failed to generate queries.")

    # print(f"\nGeneration took {end_time - start_time:.2f} seconds.")

    df = pd.read_csv("preprocessed_data.csv")
    # df = df.iloc[:100]
    # df = df.iloc[:25]

    with open("synthetic_queries_np_4_1_nano.jsonl", "w") as f:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            coffee_id = row["id"]
            coffee_profile_str = create_profile_str(row)
            
            generated_queries = generate_queries(coffee_profile_str)

            result = {
                "coffee_id": coffee_id,
                "queries": generated_queries
            }
            f.write(json.dumps(result) + "\n")

            
            # time.sleep(0.2)