from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("transactions")) -> Any:

   df = sdf.to_pandas()
   df = df[['TRANSACTIONID', 'NEWBALANCEORIG', 'AMOUNT', 'OLDBALANCEORG']]
   df["ERRORBALANCEORIG"] = df["NEWBALANCEORIG"] + df["AMOUNT"] - df["OLDBALANCEORG"]
   df = df[['TRANSACTIONID', 'ERRORBALANCEORIG']]
   return df