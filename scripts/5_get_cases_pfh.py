### Get Cases' Personal/Family Health History ###

# Import
import pandas
import os

# This query represents dataset "Cases_PersonalFamHealthHis" for domain "survey" and was generated for All of Us Controlled Tier Dataset v8
dataset_07289075_survey_sql = """
    SELECT
        answer.person_id,
        answer.survey_datetime,
        answer.survey,
        answer.question_concept_id,
        answer.question,
        answer.answer_concept_id,
        answer.answer,
        answer.survey_version_concept_id,
        answer.survey_version_name  
    FROM
        `""" + os.environ["WORKSPACE_CDR"] + """.ds_survey` answer   
    WHERE
        (
            question_concept_id IN (SELECT
                DISTINCT concept_id                         
            FROM
                `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` c                         
            JOIN
                (SELECT
                    CAST(cr.id as string) AS id                               
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.cb_criteria` cr                               
                WHERE
                    concept_id IN (1740639)                               
                    AND domain_id = 'SURVEY') a 
                    ON (c.path like CONCAT('%', a.id, '.%'))                         
            WHERE
                domain_id = 'SURVEY'                         
                AND type = 'PPI'                         
                AND subtype = 'QUESTION')
        )  
        AND (
            answer.PERSON_ID IN (SELECT
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
                        AND is_standard = 1 )) criteria ) 
                AND cb_search_person.person_id NOT IN (SELECT
                    person_id 
                FROM
                    `""" + os.environ["WORKSPACE_CDR"] + """.person` p 
                WHERE
                    sex_at_birth_concept_id IN (45880669, 1177221, 0, 1585849, 46273637, 903096) ) )
            )"""

dataset_07289075_survey_df = pandas.read_gbq(
    dataset_07289075_survey_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook")

unique_questions = dataset_07289075_survey_df["question"].unique()

all_people = (
    dataset_07289075_survey_df[["person_id"]]
        .drop_duplicates()
        .copy()
)

### Arthritis ###

# define personal vs family history

## select and filter to only relevant questions
arthritis_who_patterns = [
    "who in your family has had other arthritis",
    "who in your family has had osteoarthritis",
    "who in your family has had rheumatoid arthritis"
]

who_arthritis = dataset_07289075_survey_df[
    dataset_07289075_survey_df["question"]
        .str.lower()
        .apply(lambda x: any(p in x for p in arthritis_who_patterns))
].copy()

# clean up answer labels
who_arthritis["who"] = (
    who_arthritis["answer"]
        .str.split(" - ", n=1)
        .str[-1]
        .str.strip()
)

# remove non-responses
non_responses = {"PMI: Skip", "PMI: Dont Know"}

who_arthritis = who_arthritis[
    ~who_arthritis["who"].isin(non_responses)
].copy()

who_arthritis["who"].value_counts()

# aggregating across people
## define family group
family_members = {
    "Mother", "Father", "Sibling", "Daughter", "Son"
}

# now aggregate
who_person_arthritis = (
    who_arthritis
    .groupby("person_id")
    .agg(
        has_personal=("who", lambda x: (x == "Self").any()),
        has_family=("who", lambda x: x.isin(family_members).any())
    )
    .reset_index()
)
# merge arthritis table onto full cohort with any PFH info by person_id
arthritis_full = all_people.merge(
    who_person_arthritis[["person_id", "has_personal", "has_family"]],
    on="person_id",
    how="left"
)

# fill missing as False
arthritis_full[["has_personal", "has_family"]] = (
    arthritis_full[["has_personal", "has_family"]]
        .fillna(False)
)

arthritis_full["arthritis_history_type"] = "Neither"

# create 4 mutually exclusive classifications
arthritis_full.loc[
    (arthritis_full["has_personal"]) &
    (~arthritis_full["has_family"]),
    "arthritis_history_type"
] = "Personal only"

arthritis_full.loc[
    (arthritis_full["has_personal"]) &
    (arthritis_full["has_family"]),
    "arthritis_history_type"
] = "Personal + Family"

arthritis_full.loc[
    (~arthritis_full["has_personal"]) &
    (arthritis_full["has_family"]),
    "arthritis_history_type"
] = "Family only"
# final summary counts
who_arthritis_summary = (
    arthritis_full["arthritis_history_type"]
        .value_counts()
        .reset_index()
        .rename(columns={
            "index": "history_type",
            "arthritis_history_type": "n_people"
        })
)

who_arthritis_summary

### Cancer ###
# developing a loop across cancer types
# first, defining cancer list and possible family members
cancer_list = [
    "bladder cancer",
    "blood or soft tissue cancer",
    "bone cancer",
    "brain cancer",
    "breast cancer",
    "cervical cancer",
    "colon cancer/rectal cancer",
    "endocrine cancer",
    "endometrial cancer",
    "esophageal cancer",
    "eye cancer",
    "head and neck cancer",
    "kidney cancer",
    "lung cancer",
    "other cancer(s)",
    "ovarian cancer",
    "pancreatic cancer",
    "prostate cancer",
    "skin cancer",
    "stomach cancer",
    "thyroid cancer"
]

family_members = {
    "Mother", "Father", "Sibling", "Daughter", "Son"
}
# function to compute history categories for a single cancer
def summarize_cancer_history(cancer_name):

    # pattern for this specific cancer
    pattern = f"who in your family has had {cancer_name}"

    # subset to that cancer question
    cancer_df = dataset_07289075_survey_df[
        dataset_07289075_survey_df["question"]
            .str.lower()
            .str.contains(pattern, na=False)
    ].copy()

    # clean answer labels
    cancer_df["who"] = (
        cancer_df["answer"]
            .str.split(" - ", n=1)
            .str[-1]
            .str.strip()
    )

    # remove non-responses
    non_responses = {"PMI: Skip", "PMI: Dont Know"}
    cancer_df = cancer_df[~cancer_df["who"].isin(non_responses)]

    # aggregate by person
    person_summary = (
        cancer_df
        .groupby("person_id")
        .agg(
            has_personal=("who", lambda x: (x == "Self").any()),
            has_family=("who", lambda x: x.isin(family_members).any())
        )
        .reset_index()
    )

    # merge to full cohort
    full = all_people.merge(
        person_summary,
        on="person_id",
        how="left"
    )

    full[["has_personal", "has_family"]] = (
        full[["has_personal", "has_family"]]
            .fillna(False)
    )

    # classify
    full["history_type"] = "Neither"

    full.loc[
        (full["has_personal"]) & (~full["has_family"]),
        "history_type"
    ] = "Personal only"

    full.loc[
        (full["has_personal"]) & (full["has_family"]),
        "history_type"
    ] = "Personal + Family"

    full.loc[
        (~full["has_personal"]) & (full["has_family"]),
        "history_type"
    ] = "Family only"

    # count
    summary = (
        full["history_type"]
            .value_counts()
            .reindex(
                ["Personal only", "Personal + Family", "Family only", "Neither"],
                fill_value=0
            )
            .reset_index()
    )

    summary.columns = ["history_type", "n_people"]
    summary["cancer"] = cancer_name

    return summary
# run for all cancers
all_cancer_results = pandas.concat(
    [summarize_cancer_history(c) for c in cancer_list],
    ignore_index=True
)

cancer_pivot = (
    all_cancer_results
        .pivot(index="cancer", columns="history_type", values="n_people")
        .reset_index()
)

cancer_pivot

#### Non-Ovarian Cancers ####
# define personal vs family history

## select and filter to only relevant questions (here include the intro pattern as well)
non_ovarian_cancers = [
    "who in your family has had bladder cancer",
    "who in your family has had blood or soft tissue cancer",
    "who in your family has had bone cancer",
    "who in your family has had brain cancer",
    "who in your family has had breast cancer",
    "who in your family has had cervical cancer",
    "who in your family has had colon cancer/rectal cancer",
    "who in your family has had endocrine cancer",
    "who in your family has had endometrial cancer",
    "who in your family has had esophageal cancer",
    "who in your family has had eye cancer",
    "who in your family has had head and neck cancer",
    "who in your family has had kidney cancer",
    "who in your family has had lung cancer",
    "who in your family has had other cancer(s)",
    #"who in your family has had ovarian cancer",
    "who in your family has had pancreatic cancer",
    "who in your family has had prostate cancer",
    "who in your family has had skin cancer",
    "who in your family has had stomach cancer",
    "who in your family has had thyroid cancer"
]

who_novcancer = dataset_07289075_survey_df[
    dataset_07289075_survey_df["question"]
        .str.lower()
        .apply(lambda x: any(p in x for p in non_ovarian_cancers))
].copy()

# clean up answer labels
who_novcancer["who"] = (
    who_novcancer["answer"]
        .str.split(" - ", n=1)
        .str[-1]
        .str.strip()
)

# remove non-responses
non_responses = {"PMI: Skip", "PMI: Dont Know"}

who_novcancer = who_novcancer[
    ~who_novcancer["who"].isin(non_responses)
].copy()

who_novcancer["who"].value_counts()

## define family group
family_members = {
    "Mother", "Father", "Sibling", "Daughter", "Son"
}

# now aggregate
who_person_novcancer = (
    who_novcancer
    .groupby("person_id")
    .agg(
        has_personal=("who", lambda x: (x == "Self").any()),
        has_family=("who", lambda x: x.isin(family_members).any())
    )
    .reset_index()
)
# merge novcancer table onto full cohort with any PFH info by person_id
novcancer_full = all_people.merge(
    who_person_novcancer[["person_id", "has_personal", "has_family"]],
    on="person_id",
    how="left"
)

# fill missing as False
novcancer_full[["has_personal", "has_family"]] = (
    novcancer_full[["has_personal", "has_family"]]
        .fillna(False)
)

novcancer_full["novcancer_history_type"] = "Neither"

# create 4 mutually exclusive classifications
novcancer_full.loc[
    (novcancer_full["has_personal"]) &
    (~novcancer_full["has_family"]),
    "novcancer_history_type"
] = "Personal only"

novcancer_full.loc[
    (novcancer_full["has_personal"]) &
    (novcancer_full["has_family"]),
    "novcancer_history_type"
] = "Personal + Family"

novcancer_full.loc[
    (~novcancer_full["has_personal"]) &
    (novcancer_full["has_family"]),
    "novcancer_history_type"
] = "Family only"
# final summary counts
who_novcancer_summary = (
    novcancer_full["novcancer_history_type"]
        .value_counts()
        .reset_index()
        .rename(columns={
            "index": "history_type",
            "novcancer_history_type": "n_people"
        })
)

who_novcancer_summary

#### Non-Ovarian or Endometrial Cancers ####
# define personal vs family history

## select and filter to only relevant questions (here include the intro pattern as well)
non_ovendo_cancers = [
    "who in your family has had bladder cancer",
    "who in your family has had blood or soft tissue cancer",
    "who in your family has had bone cancer",
    "who in your family has had brain cancer",
    "who in your family has had breast cancer",
    "who in your family has had cervical cancer",
    "who in your family has had colon cancer/rectal cancer",
    "who in your family has had endocrine cancer",
    #"who in your family has had endometrial cancer",
    "who in your family has had esophageal cancer",
    "who in your family has had eye cancer",
    "who in your family has had head and neck cancer",
    "who in your family has had kidney cancer",
    "who in your family has had lung cancer",
    "who in your family has had other cancer(s)",
    #"who in your family has had ovarian cancer",
    "who in your family has had pancreatic cancer",
    "who in your family has had prostate cancer",
    "who in your family has had skin cancer",
    "who in your family has had stomach cancer",
    "who in your family has had thyroid cancer"
]

who_novencancer = dataset_07289075_survey_df[
    dataset_07289075_survey_df["question"]
        .str.lower()
        .apply(lambda x: any(p in x for p in non_ovendo_cancers))
].copy()

# clean up answer labels
who_novencancer["who"] = (
    who_novencancer["answer"]
        .str.split(" - ", n=1)
        .str[-1]
        .str.strip()
)

# remove non-responses
non_responses = {"PMI: Skip", "PMI: Dont Know"}

who_novencancer = who_novencancer[
    ~who_novencancer["who"].isin(non_responses)
].copy()

who_novencancer["who"].value_counts()

# aggregating across people
## define family group
family_members = {
    "Mother", "Father", "Sibling", "Daughter", "Son"
}

# now aggregate
who_person_novencancer = (
    who_novencancer
    .groupby("person_id")
    .agg(
        has_personal=("who", lambda x: (x == "Self").any()),
        has_family=("who", lambda x: x.isin(family_members).any())
    )
    .reset_index()
)
# merge novencancer table onto full cohort with any PFH info by person_id
novencancer_full = all_people.merge(
    who_person_novencancer[["person_id", "has_personal", "has_family"]],
    on="person_id",
    how="left"
)

# fill missing as False
novencancer_full[["has_personal", "has_family"]] = (
    novencancer_full[["has_personal", "has_family"]]
        .fillna(False)
)

# create 4 mutually exclusive classifications
novencancer_full["novencancer_history_type"] = "Neither"

novencancer_full.loc[
    (novencancer_full["has_personal"]) &
    (~novencancer_full["has_family"]),
    "novencancer_history_type"
] = "Personal only"

novencancer_full.loc[
    (novencancer_full["has_personal"]) &
    (novencancer_full["has_family"]),
    "novencancer_history_type"
] = "Personal + Family"

novencancer_full.loc[
    (~novencancer_full["has_personal"]) &
    (novencancer_full["has_family"]),
    "novencancer_history_type"
] = "Family only"
# final summary counts
who_novencancer_summary = (
    novencancer_full["novencancer_history_type"]
        .value_counts()
        .reset_index()
        .rename(columns={
            "index": "history_type",
            "novencancer_history_type": "n_people"
        })
)

who_novencancer_summary

### Obesity ###
pattern = "Including yourself, who in your family has had obesity? Select all that apply."

who_obesity = dataset_07289075_survey_df[
    dataset_07289075_survey_df["question"].str.contains(
        "who in your family has had obesity",
        case=False,
        na=False
    )
].copy()

# clean up answer labels
who_obesity["who"] = (
    who_obesity["answer"]
        .str.split(" - ", n=1)
        .str[-1]
        .str.strip()
)

# remove non-responses
non_responses = {"PMI: Skip", "PMI: Dont Know"}

who_obesity = who_obesity[
    ~who_obesity["who"].isin(non_responses)
].copy()

who_obesity["who"].value_counts()

# aggregating across people
## define family group
family_members = {
    "Mother", "Father", "Sibling", "Daughter", "Son"
}

# now aggregate
who_person_obesity = (
    who_obesity
    .groupby("person_id")
    .agg(
        has_personal=("who", lambda x: (x == "Self").any()),
        has_family=("who", lambda x: x.isin(family_members).any())
    )
    .reset_index()
)
# merge obesity table onto full cohort with any PFH info by person_id
obesity_full = all_people.merge(
    who_person_obesity[["person_id", "has_personal", "has_family"]],
    on="person_id",
    how="left"
)

# fill missing as False
obesity_full[["has_personal", "has_family"]] = (
    obesity_full[["has_personal", "has_family"]]
        .fillna(False)
)

# create 4 mutually exclusive classifications
obesity_full["obesity_history_type"] = "Neither"

obesity_full.loc[
    (obesity_full["has_personal"]) &
    (~obesity_full["has_family"]),
    "obesity_history_type"
] = "Personal only"

obesity_full.loc[
    (obesity_full["has_personal"]) &
    (obesity_full["has_family"]),
    "obesity_history_type"
] = "Personal + Family"

obesity_full.loc[
    (~obesity_full["has_personal"]) &
    (obesity_full["has_family"]),
    "obesity_history_type"
] = "Family only"

# final summary counts
who_obesity_summary = (
    obesity_full["obesity_history_type"]
        .value_counts()
        .reset_index()
        .rename(columns={
            "index": "history_type",
            "obesity_history_type": "n_people"
        })
)

who_obesity_summary
