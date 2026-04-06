### Get Case/Control Age, Race, Ethnicity, Pregnancy Status, and Smoking Status

# Setup
import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# read in merged case/control metadata pickle
merged_df_clean = pd.read_pickle(
    "./data/merged_df_clean.pkl")
    
# read in dataset with controls to remove (controls who had survey responses indicating ovarian cancer)

# This query represents dataset "controls_yesovsurvey_personid" for domain "person" and was generated for All of Us Controlled Tier Dataset v8
dataset_65184027_person_sql = """
    SELECT
        person.person_id 
    FROM
        `""" + os.environ["WORKSPACE_CDR"] + """.person` person   
    WHERE
        person.PERSON_ID IN (SELECT
            distinct person_id  
        FROM
            `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` cb_search_person  
        WHERE
            cb_search_person.person_id IN (SELECT
                person_id 
            FROM
                `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` p 
            WHERE
                has_whole_genome_variant = 1 
            UNION
            DISTINCT SELECT
                person_id 
            FROM
                `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` p 
            WHERE
                has_lr_whole_genome_variant = 1 
            UNION
            DISTINCT SELECT
                person_id 
            FROM
                `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` p 
            WHERE
                has_array_data = 1 
            UNION
            DISTINCT SELECT
                person_id 
            FROM
                `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person` p 
            WHERE
                has_structural_variant_data = 1 ) 
            AND cb_search_person.person_id IN (SELECT
                criteria.person_id 
            FROM
                (SELECT
                    DISTINCT person_id, entry_date, concept_id 
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                WHERE
                    (concept_id IN (836778) 
                    AND is_standard = 0  
                    AND  value_source_concept_id IN (1384488) 
                    OR  concept_id IN (1384483) 
                    AND is_standard = 0  
                    AND  value_source_concept_id IN (1385372))) criteria ) 
            AND cb_search_person.person_id NOT IN (SELECT
                person_id 
            FROM
                `""" + os.environ["WORKSPACE_CDR"] + """.person` p 
            WHERE
                sex_at_birth_concept_id IN (46273637, 1585849, 0, 1177221, 903096, 45880669) ) 
            AND cb_search_person.person_id NOT IN (SELECT
                criteria.person_id 
            FROM
                (SELECT
                    DISTINCT person_id, entry_date, concept_id 
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_all_events` 
                WHERE
                    (concept_id IN(SELECT
                        DISTINCT c.concept_id 
                    FROM
                        `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c 
                    JOIN
                        (SELECT
                            CAST(cr.id as string) AS id       
                        FROM
                            `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr       
                        WHERE
                            concept_id IN (4181351, 4174891, 201801, 4178969, 4116073)       
                            AND full_text LIKE '%_rank1]%'      ) a 
                            ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                            OR c.path LIKE CONCAT('%.', a.id) 
                            OR c.path LIKE CONCAT(a.id, '.%') 
                            OR c.path = a.id) 
                    WHERE
                        is_standard = 1 
                        AND is_selectable = 1) 
                    AND is_standard = 1 )) criteria ) )"""

dataset_65184027_person_df = pd.read_gbq(
    dataset_65184027_person_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook")

dataset_65184027_person_df.head(5)

# Ensure IDs are strings
merged_df_clean["s"] = merged_df_clean["s"].astype(str)
dataset_65184027_person_df["person_id"] = dataset_65184027_person_df["person_id"].astype(str)

# pull self-reported person IDs
ovarian_self_ids = set(dataset_65184027_person_df["person_id"])

print("Controls flagged with ovarian cancer from survey:", len(ovarian_self_ids))

# identify which controls to remove appear in merged df
print("Initial dataset size:", merged_df_clean.shape)

present_ids = merged_df_clean["s"].isin(ovarian_self_ids)

print("IDs present in merged dataset:", present_ids.sum())

# Check how many are controls vs cases
print(
    merged_df_clean.loc[present_ids, "disease_status"]
    .value_counts()
)

# Remove only controls with ovarian cancer self-report
merged_df_filtered = merged_df_clean[
    ~(
        merged_df_clean["s"].isin(ovarian_self_ids)
        & (merged_df_clean["disease_status"] == "control")
    )
].copy()

print("\nDataset after filtering:", merged_df_filtered.shape)

### AGE ###

# Make sure date columns are parsed as datetime
# Ensure both date columns are datetime and tz-naive
merged_df_filtered["date_of_birth"] = pd.to_datetime(
    merged_df_filtered["date_of_birth_x"], errors="coerce"
).dt.tz_localize(None)

merged_df_filtered["visit_start_date"] = pd.to_datetime(
    merged_df_filtered["visit_start_date"], errors="coerce"
).dt.tz_localize(None)

# Compute age
merged_df_filtered["age_at_collection"] = (
    (merged_df_filtered["visit_start_date"] - merged_df_filtered["date_of_birth"])
    .dt.days / 365.25
).astype(float)

age_summary = (
    merged_df_filtered
    .groupby("disease_status")["age_at_collection"]
    .agg(
        median_age="median",
        min_age="min",
        max_age="max",
        n="count"
    )
)

age_summary # this doesn't seem biologically feasible

# apply biologically reasonable age boundaries
merged_df_filtered_ageadj = merged_df_filtered[
    (merged_df_filtered["age_at_collection"] >= 18) &
    (merged_df_filtered["age_at_collection"] <= 90)
].copy()

# recompute summaries
age_summary_filt = (
    merged_df_filtered_ageadj
    .groupby("disease_status")["age_at_collection"]
    .agg(
        median_age="median",
        min_age="min",
        max_age="max",
        n="count"
    )
)

age_summary_filt

# AGE BUCKETS

# 3-bin age buckets
merged_df_filtered_ageadj["age_3"] = pd.cut(
    merged_df_filtered_ageadj["age_at_collection"],
    bins=[18, 45, 65, np.inf],
    labels=["18-44", "45-64", ">65"],
    right=False
)

# 2-bin age buckets
merged_df_filtered_ageadj["age_2"] = pd.cut(
    merged_df_filtered_ageadj["age_at_collection"],
    bins=[18, 50, np.inf],
    labels=["<50", ">=50"],
    right=False
)

age3_summary = (
    merged_df_filtered_ageadj
    .groupby("disease_status")["age_3"]
    .value_counts()
    .unstack(fill_value=0)
)

age3_summary

age2_summary = (
    merged_df_filtered_ageadj
    .groupby("disease_status")["age_2"]
    .value_counts()
    .unstack(fill_value=0)
)

age2_summary

### RACE / ETHNICITY ###
merged_df_filtered["is_white"] = merged_df_filtered["race"].eq("White")

merged_df_filtered["is_hispanic"] = merged_df_filtered["ethnicity"].eq("Hispanic or Latino")

merged_df_filtered["white_non_hispanic"] = (
    merged_df_filtered["is_white"] & ~merged_df_filtered["is_hispanic"]
)

white_summary = (
    merged_df_filtered
    .groupby("disease_status")["is_white"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={True: "White", False: "Non-White"})
)

white_summary

hispanic_summary = (
    merged_df_filtered
    .groupby("disease_status")["is_hispanic"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={True: "Hispanic", False: "Non-Hispanic"})
)

hispanic_summary

white_nh_summary = (
    merged_df_filtered
    .groupby("disease_status")["white_non_hispanic"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={True: "White non-Hispanic", False: "All others"})
)

white_nh_summary

### Pregnancy ###
preg_col = "Pregnancy: 1 Pregnancy Status"
merged_df_filtered[preg_col].value_counts(dropna=False)

# create pregnancy variable grouped accordingly
merged_df_filtered["pregnancy_status_3"] = np.select(
    [
        merged_df_filtered[preg_col] == "1 Pregnancy Status: Yes",
        merged_df_filtered[preg_col] == "1 Pregnancy Status: No"
    ],
    [
        "Pregnant",
        "Not Pregnant"
    ],
    default="Other"
)

pregnancy_summary = (
    merged_df_filtered
    .groupby("disease_status")["pregnancy_status_3"]
    .value_counts()
    .unstack(fill_value=0)
)

# enforce nice column order
pregnancy_summary = pregnancy_summary[["Pregnant", "Not Pregnant", "Other"]]

pregnancy_summary

### Smoking
cigs100_col = "Smoking: 100 Cigs Lifetime"
freq_col = "Smoking: Smoke Frequency"

ever_smoker = merged_df_filtered[cigs100_col] == "100 Cigs Lifetime: Yes"

current_smoker = merged_df_filtered[freq_col].isin([
    "Smoke Frequency: Every Day",
    "Smoke Frequency: Some Days"
])

not_current = merged_df_filtered[freq_col] == "Smoke Frequency: Not At All"

merged_df_filtered["smoking_status"] = np.select(
    [
        ever_smoker & current_smoker,
        ever_smoker & not_current,
        merged_df_filtered[cigs100_col] == "100 Cigs Lifetime: No"
    ],
    [
        "Current smoker",
        "Former smoker",
        "Never smoker"
    ],
    default="Other / Missing"
)

smoking_summary = (
    merged_df_filtered
    .groupby("disease_status")["smoking_status"]
    .value_counts()
    .unstack(fill_value=0)
)

smoking_summary = smoking_summary[
    ["Current smoker", "Former smoker", "Never smoker", "Other / Missing"]
]

smoking_summary
