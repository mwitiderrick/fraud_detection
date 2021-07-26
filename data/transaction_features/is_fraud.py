from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("transactions")) -> Any:

   df = sdf.to_pandas()
   df = df[['TRANSACTIONID', 'ISFRAUD']]

   return df