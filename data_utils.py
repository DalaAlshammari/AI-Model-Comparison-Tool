import pandas as pd


def load_csv(uploaded_file):
    """Load a CSV file and return its data."""

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            return None, "The uploaded file is empty."

        if len(df.columns) < 2:
            return None, "The file must contain at least two columns."

        if df.columns.duplicated().any():
            return None, "The file contains duplicate column names."

        return df, None

    except Exception as error:
        return None, f"Could not read the CSV file: {error}"