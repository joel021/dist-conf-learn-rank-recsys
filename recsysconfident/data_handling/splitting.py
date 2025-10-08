import pandas as pd
from sklearn.model_selection import train_test_split


def balance_items(fit_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str, item_col: str, fit_ratio: float):

    i_in_test_but_not_in_fit = list(set(test_df[item_col]) - set(fit_df[item_col].unique()))
    item_counts = fit_df.groupby(by=[item_col])[user_col].count()
    items_less_than_3 = list(item_counts[item_counts < 3].index)
    missing_items = set(i_in_test_but_not_in_fit + items_less_than_3)

    fit_extra_dfs = []
    for item in missing_items:
        extra_rows = test_df[test_df[item_col] == item].copy()
        l = int(fit_ratio * len(extra_rows))
        fit_extra_dfs.append(extra_rows.iloc[0:l])
        test_df = test_df.drop(extra_rows.index[0:l])
    fit_extra_dfs.append(fit_df)
    fit_df = pd.concat(fit_extra_dfs)

    item_counts = fit_df.groupby(by=[item_col])[user_col].count()
    missing_items = item_counts[item_counts < 2].index

    if len(missing_items) > 0:
        fit_df = fit_df[~fit_df[item_col].isin(missing_items)]
        test_df = test_df[~test_df[item_col].isin(missing_items)]

    return fit_df, test_df

def balance_users(fit_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str, item_col: str):

    user_counts = fit_df.groupby(by=[user_col])[item_col].count()
    missing_users = user_counts[user_counts < 2].index

    if len(missing_users) > 0:
        fit_df = fit_df[~fit_df[user_col].isin(missing_users)]
        test_df = test_df[~test_df[user_col].isin(missing_users)]

    return fit_df, test_df

def filter_entity_low_inter(ratings_df: pd.DataFrame, col: str, min_inter: int) -> pd.DataFrame:

    item_counts = ratings_df[col].value_counts()
    invalid_items = item_counts[item_counts < min_inter].index
    ratings_df = ratings_df[~ratings_df[col].isin(invalid_items)]
    return ratings_df

def split_ratings(ratings_df: pd.DataFrame, user_col:str, item_col:str, timestamp_col:str,
                  fit_ratio: float = 0.75, shuffle: bool = False) -> (pd.DataFrame, pd.DataFrame):

    ratings_df = filter_entity_low_inter(ratings_df, item_col, 5)
    ratings_df = filter_entity_low_inter(ratings_df, user_col, 5)
    users_grouped = ratings_df.groupby(by=[user_col])

    fit_df_list = []
    test_df_list = []

    for user, user_df in users_grouped:

        if timestamp_col and not shuffle:
            user_df = user_df.sort_values(by=[timestamp_col])

        if shuffle:
            fit_interactions, test_interactions = train_test_split(user_df,
                                                                   test_size=1 - fit_ratio,
                                                                   random_state=42)
        else:
            l = int(fit_ratio * len(user_df))
            fit_interactions = user_df.iloc[:l]
            test_interactions = user_df.iloc[l:]

        fit_df_list.append(fit_interactions)
        test_df_list.append(test_interactions)

    test_df = pd.concat(test_df_list)
    fit_df = pd.concat(fit_df_list)

    fit_df, test_df = balance_items(fit_df, test_df, user_col, item_col, fit_ratio)
    fit_df, test_df = balance_users(fit_df, test_df, user_col, item_col)
    return fit_df.reset_index(drop=True), test_df.reset_index(drop=True)
