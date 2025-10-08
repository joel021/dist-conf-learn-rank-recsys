import pandas as pd

def filter_positives(df: pd.DataFrame, relevance_col: str, threshold: float=0.75):

    relevance_max = df[relevance_col].max()

    df = df[df[relevance_col] >= relevance_max * threshold]
    df.loc[:,relevance_col] = 1
    return df
