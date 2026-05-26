import pandas as pd


def load_price_csv(
    filename: str,
    price_col_name: str,
    base_path: str = "../data/raw/",
) -> pd.DataFrame:
    df = pd.read_csv(f"{base_path}/{filename}")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Price"]].rename(columns={"Date": "date", "Price": price_col_name})
