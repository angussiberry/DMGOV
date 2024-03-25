import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def sim_results():
    df = pd.read_excel("./data/cosine_res.xlsx")

    # Define bins from 0 to 1 with 0.1 increments
    bins = np.arange(0, 1.05, 0.05)

    # Use pd.cut to bin the scores
    df["binned_score"] = pd.cut(
        df["Cosine Similarity (all-MiniLM-L6-v2)"], bins, include_lowest=True
    )

    # Convert the interval objects to strings
    df["binned_score"] = df["binned_score"].astype(str)

    # Group by the binned scores and sum the counts
    df_binned = df.groupby("binned_score")["Count"].sum().reset_index()

    print(df_binned)

    # Create a DataFrame from the bins and counts
    plot_df = pd.DataFrame(
        {
            "Cosine Similarity Score (all-MiniLM-L6-v2)": np.round(
                bins[:-1], 2
            ),  # Exclude the last bin edge
            "Number of columns with score (LLAMA2)": df_binned[
                "Count"
            ],  # Exclude the count for the first bin (which is always zero)
        }
    )

    return plot_df
