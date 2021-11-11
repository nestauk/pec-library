# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: pec_library
#     language: python
#     name: pec_library
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import re
import itertools

from datetime import datetime
from collections import Counter
from pec_library.getters.data_getters import get_library_data

# -

# ## 0. Get data

# +
# get data per mission

asf = get_library_data("climate change")
ahl = get_library_data("obesity")
afs = get_library_data("early childhood")


# -

# ## 1. EDA
# * What is the state of missing data?
# * How many subjects are there on average per book?
# * What are the most common subjects? How many different subjects?
# * How many books are published over time?
# * Holdings EDA - document types count, how many institutions on average?


def eda(records, top=10):

    # look into how much missing data there is
    records_df = pd.DataFrame.from_dict([a["bibliographic_data"] for a in records])
    print(records_df.isna().sum())

    fig, ax = plt.subplots(figsize=(7, 5))  # so graphs don't overlay each other

    if "summary" in records_df.columns:
        print(
            f"{records_df['summary'].isna().sum()/len(records_df)*100} percent of summaries are missing."
        )
        # summaries present or not over time
        records_df["date"] = (
            records_df["publication_details"].astype(str).str.findall("\d{4}")
        )
        records_df["date"] = records_df.date.apply(
            lambda x: x[0] if x != [] else np.nan
        )
        records_df = records_df[records_df["date"].notna()]

        ax = records_df.groupby("date").count()["summary"].cumsum().plot.line()
        ax.set_ylabel("# of summaries (cumulative)")
        plt.show()

    # how many subjects are there on average per book?
    # What are the most common subjects? how many different subjects?

    subjects = [
        a["bibliographic_data"]["subject"]
        for a in records
        if "subject" in a["bibliographic_data"].keys()
    ]
    print(
        f"the average no. of subjects per book is {statistics.mean([len(sub) for sub in subjects])}."
    )
    print(f"the number of subjects total is {len(subjects)}.")

    subjects_flat = list(itertools.chain(*subjects))
    subjects_flat = [
        re.sub(r"[^\w\s]", "", s).lower() for s in subjects_flat
    ]  # minimal text cleaning of subjects

    print(f"the total number of unique subjects is {len(list(set(subjects_flat)))}.")
    print(
        f"most common subjects include -------- {Counter(subjects_flat).most_common(top)}"
    )

    # books published over time?
    dates = []
    for green in records:
        if "publication_details" in green["bibliographic_data"].keys():
            date = green["bibliographic_data"]["publication_details"]
            for d in date:
                parsed_dates = re.findall("\d{4}?", d)
                if parsed_dates != []:
                    if parsed_dates[0].startswith(("17", "18", "19", "2")):
                        dates.append(datetime.strptime(parsed_dates[0], "%Y"))

    print(f"the number of unique years is {len(set(dates))}")
    print(f"the year range is between {min(dates)} and {max(dates)}.")
    ax = plt.scatter(Counter(dates).keys(), Counter(dates).values())
    plt.xlabel("date")
    plt.ylabel("# of books published")
    plt.show()

    # holdings eda - document types count, how many institutions on average hold them
    institutions = [
        name["holdings"][0]["held_at"][0]["institution"]["name"] for name in records
    ]
    material_type = [
        name["holdings"][0]["document_type"][0]
        for name in records
        if "document_type" in name["holdings"][0].keys()
    ]
    print(
        f"most common institutions include -------- {Counter(institutions).most_common()}"
    )
    print(
        f"most common material types include ------- {Counter(material_type).most_common()}"
    )


# ### 1.1 EDA per mission

eda(asf)
eda(ahl)
eda(afs)
