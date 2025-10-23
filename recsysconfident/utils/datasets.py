import pandas as pd
from sklearn.preprocessing import LabelEncoder


def filter_positives(df: pd.DataFrame, relevance_col: str, threshold: float=0.75):

    relevance_max = df[relevance_col].max()

    df = df[df[relevance_col] >= relevance_max * threshold]
    df.loc[:,relevance_col] = 1
    return df

def map_ids(ratings_df: pd.DataFrame, items_df: pd.DataFrame, user_col: str, item_col: str):

    unique_users = ratings_df[user_col].unique()
    unique_items = ratings_df[item_col].unique()

    user_encoder = LabelEncoder().fit(unique_users)
    item_encoder = LabelEncoder().fit(unique_items)

    ratings_df.loc[:, user_col] = user_encoder.transform(ratings_df[user_col].values)
    ratings_df.loc[:, item_col] = item_encoder.transform(ratings_df[item_col].values)

    if items_df is not None:
        items_df = items_df[items_df[item_col].isin(unique_items)].copy()
        items_df.loc[:, item_col] = item_encoder.transform(items_df[item_col])

    return ratings_df, items_df
