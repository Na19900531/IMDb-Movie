import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Drop empty storylines
    df = df.dropna(subset=["Storyline"])
    df = df[df["Storyline"].str.strip() != ""]
    return df
