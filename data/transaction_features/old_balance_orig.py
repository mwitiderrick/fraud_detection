from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("transactions")) -> Any:

   df = sdf.to_pandas()

   df.loc[(df["OLDBALANCEORG"] == 0) & (df["NEWBALANCEORIG"] == 0) & (df["AMOUNT"] != 0), 'OLDBALANCEORG'] = -1

   return df