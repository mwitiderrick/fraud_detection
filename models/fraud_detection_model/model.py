"""Fraud Detection Project Example

This file demonstrates how we can develop and train our model by using the
`transaction_features` we've developed earlier. Every ML model project
should have a definition file like this one.

"""
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from layer import Featureset, Train


def train_model(train: Train, ff: Featureset("fraud_features")) -> Any:
    """Model train function

    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `transaction_features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.

    Args:
        train (layer.Train): Represents the current train of the model,
            passed by Layer when the training of the model starts.
        tf (layer.Featureset): Layer will return a Featureset object,
            an interface to access the features inside the
            `transaction_features`

    Returns:
       model: A trained model object

    """

    # We create the training and label data
    df = ff.to_pandas()
    X = df.drop(["TRANSACTIONID", "ISFRAUD"], axis=1)
    Y = df["ISFRAUD"]

    random_state = 13
    test_size = 0.2
    train.log_parameter("random_state", random_state)
    train.log_parameter("test_size", test_size)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_size,
                                                    random_state=random_state)

    # Here we register input & output of the train. Layer will use
    # this registers to extract the signature of the model and calculate
    # the drift
    train.register_input(trainX)
    train.register_output(trainY)

    # Train model
    model = LogisticRegression()
    model.fit(trainX, trainY)
    preds = model.predict(testX,testY)

    # Since the data is highly skewed, we will use the area under the
    # precision-recall curve (AUPRC) rather than the conventional area under
    # the receiver operating characteristic (AUROC). This is because the AUPRC
    # is more sensitive to differences between algorithms and their parameter
    # settings rather than the AUROC (see Davis and Goadrich,
    # 2006: http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf)
    auprc = average_precision_score(testY, preds)
    train.log_metric("auprc", auprc)

    # We return the model
    return model
