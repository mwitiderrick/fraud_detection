from typing import Any
from layer import Dataset
from sklearn.preprocessing import OrdinalEncoder


def build_feature(sdf: Dataset("transactions")) -> Any:

   df = sdf.to_pandas()

   feature_data = df[["TRANSACTIONID", "TYPE"]]

   # creating instance of ordinalencoder
   ordinalencoder = OrdinalEncoder()
   feature_data = feature_data.assign(transaction_type=ordinalencoder.fit_transform(feature_data[["TYPE"]]))
   feature_data.drop(columns=["TYPE"], inplace=True)

   return feature_data