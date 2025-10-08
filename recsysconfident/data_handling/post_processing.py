import pandas as pd
import numpy as np

from recsysconfident.constants import NEG_FLAG_COL


class Processing:


    def __init__(self, rate_range: list[float], r_pred_col: str="r_pred", model_conf_col: str="conf_pred"):

        self.lower_rate = float(rate_range[0])
        self.upper_rate = float(rate_range[1])
        self.r_pred_col = r_pred_col
        self.model_conf_col = model_conf_col

    def parse_clip_shift(self, df: pd.DataFrame, abs_shift_conf=False) -> pd.DataFrame:
        df = df.copy()

        if NEG_FLAG_COL in df.columns:
            df = df[df[NEG_FLAG_COL] == 0]

        df.loc[:, self.model_conf_col] = df[self.model_conf_col].replace([np.inf, -np.inf, np.nan], 0)
        df.dropna(inplace=True)

        df.loc[:, self.r_pred_col] = df[self.r_pred_col].clip(lower=self.lower_rate,
                                                              upper=self.upper_rate)

        if abs_shift_conf:
            df.loc[:, self.model_conf_col] = np.abs(df[self.model_conf_col].values - df[self.model_conf_col].mean())

        return df

