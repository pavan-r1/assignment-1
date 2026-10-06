import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features():

    # Load dataset
    df = pd.read_csv("data/churn.csv")

    # Separate input features and target
    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Numerical and categorical features
    numerical_features = [
        "tenure",
        "monthly_charges",
        "total_charges"
    ]

    categorical_features = [
        "contract_type",
        "internet_service",
        "tech_support"
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return X, y, preprocessor


if __name__ == "__main__":
    X, y, preprocessor = prepare_features()

    print("Feature engineering completed successfully.")
    print("Input features:", X.columns.tolist())
    print("Target:", "churn")
    print("Numerical features:", [
        "tenure",
        "monthly_charges",
        "total_charges"
    ])
    print("Categorical features:", [
        "contract_type",
        "internet_service",
        "tech_support"
    ])