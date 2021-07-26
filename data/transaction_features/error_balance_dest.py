from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("transactions")) -> Any:

   df = sdf.to_pandas()
   df = df[['TRANSACTIONID', 'OLDBALANCEDEST', 'AMOUNT', 'NEWBALANCEDEST']]
   df["ERRORBALANCEDEST"] = df["OLDBALANCEDEST"] + df["AMOUNT"] - df["NEWBALANCEDEST"]
   df = df[['TRANSACTIONID','ERRORBALANCEDEST']]

   return df