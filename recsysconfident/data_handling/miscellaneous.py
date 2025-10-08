import pandas as pd


def check_all_users_higher_than(df: pd.DataFrame, user_col: str, rating_col: str, threshold: float):

    has_high_rating = df.groupby(user_col)[rating_col].apply(lambda x: (x > threshold).any())
    return has_high_rating.all()

def keep_users_any_r_higher_than(df, user_col: str, rating_col: str, threshold: float):
    filtered_df = df.groupby(user_col).filter(lambda x: (x[rating_col] > threshold).any())
    return filtered_df.reset_index(drop=True)

def filter_out_users_less_than_k_inter(df: pd.DataFrame, user_col: str, min_iterations: int):
    df = df.copy()
    df = df.groupby(user_col).filter(
        lambda x: len(x) >= min_iterations)  # ensure there is at least k ratings per user

    return df

def shift_relevances(df, relevance_col: str, r_pred_col: str, rmin: float):
    df.loc[:, relevance_col] = df[relevance_col] - rmin  # avoid negative scores
    df.loc[:, r_pred_col] = df[r_pred_col] - rmin  # shift to avoid negative scores
    return df


def min_max_norm(df: pd.DataFrame, col: str, default_max: float):
    df[col] = df[col].astype(float)

    col_min = df[col].min()
    col_max = df[col].max()

    if col_max == col_min:
        if col_max >= 0.9 * default_max:
            df.loc[:, col] = 1.
        else:
            df.loc[:, col] = 0.
    else:
        df.loc[:, col] = (df[col] - col_min) / (col_max - col_min)

    return df

