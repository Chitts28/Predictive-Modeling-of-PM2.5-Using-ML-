import pandas as pd

def load_and_preprocess(data_path="data/pm25_data.csv"):
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.fillna(method="ffill")  # handle missing values
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    return df

if __name__ == "__main__":
    df = load_and_preprocess()
    df.to_csv("data/processed_pm25.csv", index=False)
    print("Preprocessing completed, saved to data/processed_pm25.csv")
