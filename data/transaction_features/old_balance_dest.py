from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("transactions")) -> Any:

   df = sdf.to_pandas()

   df.loc[(df["OLDBALANCEDEST"] == 0) & (df["NEWBALANCEDEST"] == 0) & (df["AMOUNT"] != 0), 'OLDBALANCEDEST'] = - 1

   return df
