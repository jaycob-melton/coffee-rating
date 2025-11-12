import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.utils import build_all_vocabs #move to data.py?
from src.config import (
    PREPROCESSED_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    VOCABS_PATH
)

if __name__ == "__main__":
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    vocabs = build_all_vocabs(train_df)
    with open(VOCABS_PATH, "w") as f:
        json.dump(vocabs, f)
    print(f"Saved vocabularies to: {VOCABS_PATH}")