import os

from joblib import Parallel, delayed
import pandas as pd



def filter_low_conf_items(user_df: pd.DataFrame, cutoff: float, k: int, conf_col: str= 'conf_pred'):

    user_confident_df = user_df[user_df[conf_col] > cutoff]
    if len(user_confident_df) < k:
        # Select remaining rows with lower confidence sorted by conf_pred descending
        group_low = user_df[user_df[conf_col] <= cutoff].sort_values(by=conf_col,
                                                                     ascending=False)
        additional_rows = group_low.iloc[:k - len(user_confident_df)]
        user_confident_df = pd.concat([user_confident_df, additional_rows])
    return user_confident_df

def conf_aware_rank(user_id: any, user_df: pd.DataFrame, cutoff: float, k: int, rel_col: str= 'r_true',
                    rel_pred_col: str= 'r_pred', conf_col: str= 'conf_pred'):

    user_confident_df = filter_low_conf_items(user_df, cutoff, k, conf_col)
    true_relevance = user_confident_df[rel_col].values
    pred_relevance = user_confident_df[rel_pred_col].values

    return user_id, [true_relevance, pred_relevance]

def parallel_filter_conf_aware_rank(df: pd.DataFrame, k: int, conf_threshold: float, model_conf_col: str, rating_col: str,
                                    user_col: str, rating_pred_col: str) -> dict:

    results = Parallel(n_jobs=5 * os.cpu_count())(
        delayed(conf_aware_rank)(user_id, user_df, conf_threshold, k, rating_col, rating_pred_col, model_conf_col) for user_id, user_df in df.groupby(user_col)
    )

    return dict(results)
