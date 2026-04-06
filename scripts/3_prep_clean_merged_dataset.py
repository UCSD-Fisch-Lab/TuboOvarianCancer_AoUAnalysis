### Prep Clean Merged Dataset ###

# Setup
import pandas as pd
import numpy as np
import os
import gc
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
  roc_auc_score, roc_curve, accuracy_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load TSV files
## Note: the srWGS and Array variant statuses in the following two files weren't used for this manuscript, so that code is not provided.
## The counts used for the manuscript can still be replicated from the information here in meta_df and age_df merged into one dataframe.
srwgs_var_df = pd.read_csv("./data/srwgs_set1_variant_features.tsv", sep="\t")
array_var_df = pd.read_csv("./data/array_set1_variant_features.tsv", sep="\t")
meta_df = pd.read_csv("./data/meta_combined_all.tsv", sep="\t")
age_df = pd.read_csv("./data/meta_age_all.tsv", sep="\t")

age_df = age_df.drop_duplicates(subset="person_id", keep="first")

# Need to add case/control labels

# Case and control ID files (1 ID per line)
# Specify subsetting for case samples
# This query represents dataset "cases_person_id" for domain "person" and was generated for All of Us Controlled Tier Dataset v8
cases_person_sql = """
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
                sex_at_birth_concept_id IN (45880669, 1177221, 0, 1585849, 46273637, 903096) ) )"""

cases_person_df = pandas.read_gbq(
  cases_person_sql,
  dialect="standard",
  use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
  progress_bar_type="tqdm_notebook")

# Specify subsetting for control samples

# This query represents dataset "controls_person_id" for domain "person" and was generated for All of Us Controlled Tier Dataset v8
controls_person_sql = """
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

controls_person_df = pandas.read_gbq(
  controls_person_sql,
  dialect="standard",
  use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
  progress_bar_type="tqdm_notebook")

# Extract person_id lists for cases and controls
case_ids = cases_person_df['person_id'].astype(str).tolist()
control_ids = controls_person_df['person_id'].astype(str).tolist()

case_ids["disease_status"] = "case"
control_ids["disease_status"] = "control"

# Combine and deduplicate
labels_df = pd.concat([case_ids, control_ids], ignore_index=True).drop_duplicates(subset=["s"])

# Ensure sample IDs are strings
labels_df["s"] = labels_df["s"].astype(str)

# Merge all data sources
 first variant info, retaining all participants regardless of overlap
variant_merged = pd.merge(
    srwgs_var_df,
    array_var_df,
    on="s",                
    how="outer",
    suffixes=("_srwgs", "_array")
)

# merge age and other meta together
meta_merged = pd.merge(
    age_df,
    meta_df,
    on="person_id",
    how="right",
    suffixes=("_x", "_y")
)

# merge in metadata
merged_df = pd.merge(
    variant_merged,
    meta_merged,
    left_on="s",
    right_on="person_id",
    how="left"
)

# Ensure sample IDs are strings
merged_df["s"] = merged_df["s"].astype(str)

# Add case/control labels
merged_df = merged_df.merge(labels_df, on="s", how="left")

# Drop redundant identifiers
merged_df = merged_df.drop(columns=["person_id"])

# remove features that directly encode diagnosis
exclude_keywords = [
    "ovarian cancer", 
    "endometrial cancer",
    "cervical cancer",
    "breast cancer",
    "colon cancer",
    "rectal cancer",
    "bladder cancer",
    "other cancer",
]

filtered_cols = [
    c for c in merged_df.columns 
    if not any(kw.lower() in c.lower() for kw in exclude_keywords)
]
merged_df = merged_df[filtered_cols]# + ["disease_status"]]

# Remove manually specified features before modeling

remove_features = [
    "gender_concept_id",
    "date_of_birth_y",
    "race_concept_id",
    "ethnicity_concept_id",
    "sex_at_birth_concept_id",
    "sex_at_birth",
    "self_reported_category_concept_id",
    "AIAN: AIAN Specific",
    "Asian: Asian Specific",
    "Biological Sex At Birth: Sex At Birth",
    "Black: Black Specific",
    "Gender Identity: Sexuality Closer Description",
    "Gender: Closer Gender Description",
    "Gender: Gender Identity",
    "Have you ever been diagnosed with the following conditions? Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following bone, joint, and muscle conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following brain and nervous system conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following cancer conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following digestive conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following hearing and eye conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following heart and blood conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following hormone and endocrine conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following kidney conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following lung conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Have you or anyone in your family ever been diagnosed with the following mental health or substance use conditions? Think only of the people you are related to by blood. Select all that apply.",
    "Hispanic: Hispanic Specific",
    "Living Situation: Current Living",
    "Living Situation: How Many Living Years",
    "MENA: MENA Specific",
    "NHPI: NHPI Specific",
    "Overall Health Ovary Removal History Age",
    "Overall Health: Hysterectomy History",
    "Overall Health: Hysterectomy History Age",
    "Overall Health: Ovary Removal History",
    "Race: What Race Ethnicity",
    "Recreational Drug Use: Which Drugs Used",
    "The Basics: Birthplace",
    "White: White Specific",
    "Yes None: Menstrual Stopped Reason"
]

# Drop safely (ignore missing columns)
existing_to_drop = [col for col in remove_features if col in merged_df.columns]
missing = [col for col in remove_features if col not in merged_df.columns]

# Drop the unwanted features
merged_df_clean = merged_df.drop(columns=existing_to_drop, errors="ignore")

# Drop columns that are entirely NA 
na_cols_before = merged_df_clean.shape[1]

merged_df_clean.replace(
    ["", " ", "NA", "N/A"],
    np.nan,
    inplace=True
)

merged_df = merged_df.dropna(axis=1, how='all')

na_cols_after = merged_df_clean.shape[1]

# Now make readjustments

binary_numeric_features = ["chr1_45331840_G_A_srwgs","chr13_32333864_A_T_srwgs","chr13_32340300_GT_G_srwgs",
                           "chr16_28902649_TC_T_srwgs","chr16_68833362_A_G_srwgs","chr17_35106406_G_A_srwgs",
                           "chr17_43057062_T_TG_srwgs","chr17_43092039_ACTAGTATCTTC_A_srwgs",
                           "chr17_43094514_C_CT_srwgs","chr17_43124027_ACT_A_srwgs","chr7_5993369_T_TA_srwgs",
                           "chrX_123899119_C_CAAAAAAA_srwgs","chr1_45331840_G_A_array","chr13_32333864_A_T_array",
                           "chr13_32340300_GT_G_array","chr16_28902649_TC_T_array","chr16_68833362_A_G_array",
                           "chr17_35106406_G_A_array","chr17_43057062_T_TG_array",
                           "chr17_43092039_ACTAGTATCTTC_A_array","chr17_43094514_C_CT_array",
                           "chr17_43124027_ACT_A_array","chr7_5993369_T_TA_array",
                           "chrX_123899119_C_CAAAAAAA_array"]

categorical_features = ["gender","race","ethnicity","self_reported_category",
                        "Active Duty: Active Duty Serve Status","Alcohol: Alcohol Participant",
                        "Alcohol: Drink Frequency Past Year","Are you currently prescribed medications and/or receiving treatment for Crohn's disease?",
                        "Are you currently prescribed medications and/or receiving treatment for HIV/AIDS?",
                        "Are you currently prescribed medications and/or receiving treatment for Lou Gehrig's disease (amyotrophic lateral sclerosis)?",
                        "Are you currently prescribed medications and/or receiving treatment for Lyme disease?",
                        "Are you currently prescribed medications and/or receiving treatment for Parkinson's disease?",
                        "Are you currently prescribed medications and/or receiving treatment for West Nile virus?",
                        "Are you currently prescribed medications and/or receiving treatment for Zika virus?",
                        "Are you currently prescribed medications and/or receiving treatment for a bleeding disorder?",
                        "Are you currently prescribed medications and/or receiving treatment for a drug use disorder?",
                        "Are you currently prescribed medications and/or receiving treatment for a heart attack?",
                        "Are you currently prescribed medications and/or receiving treatment for a hernia?",
                        "Are you currently prescribed medications and/or receiving treatment for a liver condition (e.g., cirrhosis)?",
                        "Are you currently prescribed medications and/or receiving treatment for a personality disorder?",
                        "Are you currently prescribed medications and/or receiving treatment for a skin condition(s) (e.g., eczema, psoriasis)?",
                        "Are you currently prescribed medications and/or receiving treatment for a social phobia?",
                        "Are you currently prescribed medications and/or receiving treatment for a stroke?",
                        "Are you currently prescribed medications and/or receiving treatment for acid reflux?",
                        "Are you currently prescribed medications and/or receiving treatment for acne?",
                        "Are you currently prescribed medications and/or receiving treatment for acute kidney disease with no current dialysis?",
                        "Are you currently prescribed medications and/or receiving treatment for alcohol use disorder?",
                        "Are you currently prescribed medications and/or receiving treatment for allergies?",
                        "Are you currently prescribed medications and/or receiving treatment for an eating disorder?",
                        "Are you currently prescribed medications and/or receiving treatment for an enlarged prostate?",
                        "Are you currently prescribed medications and/or receiving treatment for anemia?",
                        "Are you currently prescribed medications and/or receiving treatment for anxiety reaction/panic disorder?",
                        "Are you currently prescribed medications and/or receiving treatment for aortic aneurysm?",
                        "Are you currently prescribed medications and/or receiving treatment for asthma?",
                        "Are you currently prescribed medications and/or receiving treatment for astigmatism?",
                        "Are you currently prescribed medications and/or receiving treatment for atrial fibrillation (a-fib) or atrial flutter (or a-flutter)?",
                        "Are you currently prescribed medications and/or receiving treatment for attention-deficit/hyperactivity disorder (ADHD)?",
                        "Are you currently prescribed medications and/or receiving treatment for autism spectrum disorder?",
                        "Are you currently prescribed medications and/or receiving treatment for bipolar disorder?",
                        "Are you currently prescribed medications and/or receiving treatment for blindness, all causes?",
                        "Are you currently prescribed medications and/or receiving treatment for blood or soft tissue cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for bone cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for bowel obstruction?",
                        "Are you currently prescribed medications and/or receiving treatment for brain cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for carpal tunnel syndrome?",
                        "Are you currently prescribed medications and/or receiving treatment for cataracts?",
                        "Are you currently prescribed medications and/or receiving treatment for celiac disease?",
                        "Are you currently prescribed medications and/or receiving treatment for cerebral palsy?",
                        "Are you currently prescribed medications and/or receiving treatment for chickenpox?",
                        "Are you currently prescribed medications and/or receiving treatment for chronic fatigue?",
                        "Are you currently prescribed medications and/or receiving treatment for chronic lung disease (COPD, emphysema, or bronchitis)?",
                        "Are you currently prescribed medications and/or receiving treatment for chronic sinus infections?",
                        "Are you currently prescribed medications and/or receiving treatment for colon polyps?",
                        "Are you currently prescribed medications and/or receiving treatment for concussion or loss of consciousness?",
                        "Are you currently prescribed medications and/or receiving treatment for congestive heart failure?",
                        "Are you currently prescribed medications and/or receiving treatment for coronary artery/coronary heart disease?",
                        "Are you currently prescribed medications and/or receiving treatment for dementia (includes Alzheimer's, vascular, etc.)?",
                        "Are you currently prescribed medications and/or receiving treatment for dengue fever?",
                        "Are you currently prescribed medications and/or receiving treatment for depression?",
                        "Are you currently prescribed medications and/or receiving treatment for diverticulitis/diverticulosis?",
                        "Are you currently prescribed medications and/or receiving treatment for dry eyes?",
                        "Are you currently prescribed medications and/or receiving treatment for endocrine cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for endometriosis?",
                        "Are you currently prescribed medications and/or receiving treatment for epilepsy or seizure?",
                        "Are you currently prescribed medications and/or receiving treatment for esophageal cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for eye cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for farsightedness?",
                        "Are you currently prescribed medications and/or receiving treatment for fibroids?",
                        "Are you currently prescribed medications and/or receiving treatment for fibromyalgia?",
                        "Are you currently prescribed medications and/or receiving treatment for fractured/broken bones?",
                        "Are you currently prescribed medications and/or receiving treatment for gall stones?",
                        "Are you currently prescribed medications and/or receiving treatment for glaucoma?",
                        "Are you currently prescribed medications and/or receiving treatment for gout?",
                        "Are you currently prescribed medications and/or receiving treatment for head and neck cancer? (This includes cancers of the mouth, sinuses, nose, or throat. This does not include brain cancer.)",
                        "Are you currently prescribed medications and/or receiving treatment for heart valve disease?",
                        "Are you currently prescribed medications and/or receiving treatment for hemorrhoids?",
                        "Are you currently prescribed medications and/or receiving treatment for hepatitis A?",
                        "Are you currently prescribed medications and/or receiving treatment for hepatitis B?",
                        "Are you currently prescribed medications and/or receiving treatment for hepatitis C?",
                        "Are you currently prescribed medications and/or receiving treatment for high blood pressure (hypertension)?",
                        "Are you currently prescribed medications and/or receiving treatment for high cholesterol?",
                        "Are you currently prescribed medications and/or receiving treatment for hyperthyroidism?",
                        "Are you currently prescribed medications and/or receiving treatment for hypothyroidism?",
                        "Are you currently prescribed medications and/or receiving treatment for insomnia?",
                        "Are you currently prescribed medications and/or receiving treatment for irritable bowel syndrome (IBS)?",
                        "Are you currently prescribed medications and/or receiving treatment for kidney cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for kidney disease with dialysis?",
                        "Are you currently prescribed medications and/or receiving treatment for kidney disease without dialysis?",
                        "Are you currently prescribed medications and/or receiving treatment for kidney stones?",
                        "Are you currently prescribed medications and/or receiving treatment for lung cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for macular degeneration?",
                        "Are you currently prescribed medications and/or receiving treatment for memory loss or impairment?",
                        "Are you currently prescribed medications and/or receiving treatment for migraine headaches?",
                        "Are you currently prescribed medications and/or receiving treatment for multiple sclerosis (MS)?",
                        "Are you currently prescribed medications and/or receiving treatment for muscular dystrophy (MD)?",
                        "Are you currently prescribed medications and/or receiving treatment for narcolepsy?",
                        "Are you currently prescribed medications and/or receiving treatment for nearsightedness?",
                        "Are you currently prescribed medications and/or receiving treatment for neuropathy?",
                        "Are you currently prescribed medications and/or receiving treatment for obesity?",
                        "Are you currently prescribed medications and/or receiving treatment for osteoarthritis?",
                        "Are you currently prescribed medications and/or receiving treatment for osteoporosis?",
                        "Are you currently prescribed medications and/or receiving treatment for other arthritis?",
                        "Are you currently prescribed medications and/or receiving treatment for other bone, joint, or muscle condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other brain or nervous system condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other digestive condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other hearing or eye condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other heart or blood condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other hormone/endocrine condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other infectious disease(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other kidney condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other lung condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other mental health or substance use condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for other/unknown diabetes?",
                        "Are you currently prescribed medications and/or receiving treatment for other/unknown thyroid condition(s)?",
                        "Are you currently prescribed medications and/or receiving treatment for pancreatic cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for pancreatitis?",
                        "Are you currently prescribed medications and/or receiving treatment for peptic (stomach) ulcers?",
                        "Are you currently prescribed medications and/or receiving treatment for peripheral vascular disease?",
                        "Are you currently prescribed medications and/or receiving treatment for polycystic ovarian syndrome?",
                        "Are you currently prescribed medications and/or receiving treatment for post-traumatic stress disorder (PTSD)?",
                        "Are you currently prescribed medications and/or receiving treatment for prediabetes?",
                        "Are you currently prescribed medications and/or receiving treatment for prostate cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for pseudogout (CPPD)?",
                        "Are you currently prescribed medications and/or receiving treatment for pulmonary embolism or deep vein thrombosis (DVT)?",
                        "Are you currently prescribed medications and/or receiving treatment for reactions to anesthesia (such as hyperthermia)?",
                        "Are you currently prescribed medications and/or receiving treatment for recurrent urinary tract infections (UTI)/bladder infections?",
                        "Are you currently prescribed medications and/or receiving treatment for recurrent yeast infections?",
                        "Are you currently prescribed medications and/or receiving treatment for restless leg syndrome?",
                        "Are you currently prescribed medications and/or receiving treatment for rheumatoid arthritis (RA)?",
                        "Are you currently prescribed medications and/or receiving treatment for schizophrenia?",
                        "Are you currently prescribed medications and/or receiving treatment for severe acute respiratory syndrome (SARS)?",
                        "Are you currently prescribed medications and/or receiving treatment for severe hearing loss or partial deafness in one or both ears?",
                        "Are you currently prescribed medications and/or receiving treatment for sexually transmitted infections (gonorrhea, syphilis, chlamydia)?",
                        "Are you currently prescribed medications and/or receiving treatment for shingles?",
                        "Are you currently prescribed medications and/or receiving treatment for sickle cell disease?",
                        "Are you currently prescribed medications and/or receiving treatment for skin cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for sleep apnea?",
                        "Are you currently prescribed medications and/or receiving treatment for spinal cord injury or impairment?",
                        "Are you currently prescribed medications and/or receiving treatment for spine, muscle, or bone disorders (non-cancer)?",
                        "Are you currently prescribed medications and/or receiving treatment for stomach cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for systemic lupus?",
                        "Are you currently prescribed medications and/or receiving treatment for thyroid cancer?",
                        "Are you currently prescribed medications and/or receiving treatment for tinnitus?",
                        "Are you currently prescribed medications and/or receiving treatment for transient ischemic attacks (TIAs or mini-strokes)?",
                        "Are you currently prescribed medications and/or receiving treatment for traumatic brain injury (TBI)?",
                        "Are you currently prescribed medications and/or receiving treatment for tuberculosis?",
                        "Are you currently prescribed medications and/or receiving treatment for type 1 diabetes?",
                        "Are you currently prescribed medications and/or receiving treatment for type 2 diabetes?",
                        "Are you currently prescribed medications and/or receiving treatment for ulcerative colitis?",
                        "Are you currently prescribed medications and/or receiving treatment for vitamin B deficiency?",
                        "Are you currently prescribed medications and/or receiving treatment for vitamin D deficiency?",
                        "Are you still seeing a doctor or health care provider for Crohn's disease?",
                        "Are you still seeing a doctor or health care provider for HIV/AIDS?",
                        "Are you still seeing a doctor or health care provider for Lou Gehrig's disease (amyotrophic lateral sclerosis)?",
                        "Are you still seeing a doctor or health care provider for Lyme disease?",
                        "Are you still seeing a doctor or health care provider for Parkinson's disease?",
                        "Are you still seeing a doctor or health care provider for West Nile virus?",
                        "Are you still seeing a doctor or health care provider for Zika virus?",
                        "Are you still seeing a doctor or health care provider for a bleeding disorder?",
                        "Are you still seeing a doctor or health care provider for a drug use disorder?",
                        "Are you still seeing a doctor or health care provider for a heart attack?",
                        "Are you still seeing a doctor or health care provider for a hernia?",
                        "Are you still seeing a doctor or health care provider for a liver condition (e.g., cirrhosis)?",
                        "Are you still seeing a doctor or health care provider for a personality disorder?",
                        "Are you still seeing a doctor or health care provider for a social phobia?",
                        "Are you still seeing a doctor or health care provider for a stroke?",
                        "Are you still seeing a doctor or health care provider for acid reflux?",
                        "Are you still seeing a doctor or health care provider for acne?",
                        "Are you still seeing a doctor or health care provider for acute kidney disease with no current dialysis?",
                        "Are you still seeing a doctor or health care provider for alcohol use disorder?",
                        "Are you still seeing a doctor or health care provider for allergies?",
                        "Are you still seeing a doctor or health care provider for an eating disorder?",
                        "Are you still seeing a doctor or health care provider for an enlarged prostate?",
                        "Are you still seeing a doctor or health care provider for anemia?",
                        "Are you still seeing a doctor or health care provider for anxiety reaction/panic disorder?",
                        "Are you still seeing a doctor or health care provider for aortic aneurysm?",
                        "Are you still seeing a doctor or health care provider for asthma?",
                        "Are you still seeing a doctor or health care provider for astigmatism?",
                        "Are you still seeing a doctor or health care provider for atrial fibrillation (a-fib) or atrial flutter (or a-flutter)?",
                        "Are you still seeing a doctor or health care provider for attention-deficit/hyperactivity disorder (ADHD)?",
                        "Are you still seeing a doctor or health care provider for autism spectrum disorder?",
                        "Are you still seeing a doctor or health care provider for bipolar disorder?",
                        "Are you still seeing a doctor or health care provider for blindness, all causes?",
                        "Are you still seeing a doctor or health care provider for blood or soft tissue cancer?",
                        "Are you still seeing a doctor or health care provider for bone cancer?",
                        "Are you still seeing a doctor or health care provider for bowel obstruction?",
                        "Are you still seeing a doctor or health care provider for brain cancer?",
                        "Are you still seeing a doctor or health care provider for carpal tunnel syndrome?",
                        "Are you still seeing a doctor or health care provider for cataracts?",
                        "Are you still seeing a doctor or health care provider for celiac disease?",
                        "Are you still seeing a doctor or health care provider for cerebral palsy?",
                        "Are you still seeing a doctor or health care provider for chickenpox?",
                        "Are you still seeing a doctor or health care provider for chronic fatigue?",
                        "Are you still seeing a doctor or health care provider for chronic lung disease (COPD, emphysema, or bronchitis)?",
                        "Are you still seeing a doctor or health care provider for chronic sinus infections?",
                        "Are you still seeing a doctor or health care provider for colon polyps?",
                        "Are you still seeing a doctor or health care provider for concussion or loss of consciousness?",
                        "Are you still seeing a doctor or health care provider for congestive heart failure?",
                        "Are you still seeing a doctor or health care provider for coronary artery/coronary heart disease?",
                        "Are you still seeing a doctor or health care provider for dementia (includes Alzheimer's, vascular, etc.)?",
                        "Are you still seeing a doctor or health care provider for dengue fever?",
                        "Are you still seeing a doctor or health care provider for depression?",
                        "Are you still seeing a doctor or health care provider for diverticulitis/diverticulosis?",
                        "Are you still seeing a doctor or health care provider for dry eyes?",
                        "Are you still seeing a doctor or health care provider for endocrine cancer?",
                        "Are you still seeing a doctor or health care provider for endometriosis?",
                        "Are you still seeing a doctor or health care provider for epilepsy or seizure?",
                        "Are you still seeing a doctor or health care provider for esophageal cancer?",
                        "Are you still seeing a doctor or health care provider for eye cancer?",
                        "Are you still seeing a doctor or health care provider for farsightedness?",
                        "Are you still seeing a doctor or health care provider for fibroids?",
                        "Are you still seeing a doctor or health care provider for fibromyalgia?",
                        "Are you still seeing a doctor or health care provider for fractured/broken bones?",
                        "Are you still seeing a doctor or health care provider for gall stones?",
                        "Are you still seeing a doctor or health care provider for glaucoma?",
                        "Are you still seeing a doctor or health care provider for gout?",
                        "Are you still seeing a doctor or health care provider for head and neck cancer? (This includes cancers of the mouth, sinuses, nose, or throat. This does not include brain cancer.)",
                        "Are you still seeing a doctor or health care provider for heart valve disease?",
                        "Are you still seeing a doctor or health care provider for hemorrhoids?",
                        "Are you still seeing a doctor or health care provider for hepatitis A?",
                        "Are you still seeing a doctor or health care provider for hepatitis B?",
                        "Are you still seeing a doctor or health care provider for hepatitis C?",
                        "Are you still seeing a doctor or health care provider for high blood pressure (hypertension)?",
                        "Are you still seeing a doctor or health care provider for high cholesterol?",
                        "Are you still seeing a doctor or health care provider for hyperthyroidism?",
                        "Are you still seeing a doctor or health care provider for hypothyroidism?",
                        "Are you still seeing a doctor or health care provider for insomnia?",
                        "Are you still seeing a doctor or health care provider for irritable bowel syndrome (IBS)?",
                        "Are you still seeing a doctor or health care provider for kidney cancer?",
                        "Are you still seeing a doctor or health care provider for kidney disease with dialysis?",
                        "Are you still seeing a doctor or health care provider for kidney disease without dialysis?",
                        "Are you still seeing a doctor or health care provider for kidney stones?",
                        "Are you still seeing a doctor or health care provider for lung cancer?",
                        "Are you still seeing a doctor or health care provider for macular degeneration?",
                        "Are you still seeing a doctor or health care provider for memory loss or impairment?",
                        "Are you still seeing a doctor or health care provider for migraine headaches?",
                        "Are you still seeing a doctor or health care provider for multiple sclerosis (MS)?",
                        "Are you still seeing a doctor or health care provider for muscular dystrophy (MD)?",
                        "Are you still seeing a doctor or health care provider for narcolepsy?",
                        "Are you still seeing a doctor or health care provider for nearsightedness?",
                        "Are you still seeing a doctor or health care provider for neuropathy?",
                        "Are you still seeing a doctor or health care provider for obesity?",
                        "Are you still seeing a doctor or health care provider for osteoarthritis?",
                        "Are you still seeing a doctor or health care provider for osteoporosis?",
                        "Are you still seeing a doctor or health care provider for other arthritis?",
                        "Are you still seeing a doctor or health care provider for other bone, joint, or muscle condition(s)?",
                        "Are you still seeing a doctor or health care provider for other brain or nervous system condition(s)?",
                        "Are you still seeing a doctor or health care provider for other condition(s)?",
                        "Are you still seeing a doctor or health care provider for other digestive condition(s)?",
                        "Are you still seeing a doctor or health care provider for other hearing or eye condition(s)?",
                        "Are you still seeing a doctor or health care provider for other heart or blood condition(s)?",
                        "Are you still seeing a doctor or health care provider for other hormone/endocrine condition(s)?",
                        "Are you still seeing a doctor or health care provider for other infectious disease(s)?",
                        "Are you still seeing a doctor or health care provider for other kidney condition(s)?",
                        "Are you still seeing a doctor or health care provider for other lung condition(s)?",
                        "Are you still seeing a doctor or health care provider for other mental health or substance use condition(s)?",
                        "Are you still seeing a doctor or health care provider for other/unknown diabetes?",
                        "Are you still seeing a doctor or health care provider for other/unknown thyroid condition(s)?",
                        "Are you still seeing a doctor or health care provider for pancreatic cancer?",
                        "Are you still seeing a doctor or health care provider for pancreatitis?",
                        "Are you still seeing a doctor or health care provider for peptic (stomach) ulcers?",
                        "Are you still seeing a doctor or health care provider for peripheral vascular disease?",
                        "Are you still seeing a doctor or health care provider for polycystic ovarian syndrome?",
                        "Are you still seeing a doctor or health care provider for post-traumatic stress disorder (PTSD)?",
                        "Are you still seeing a doctor or health care provider for prediabetes?",
                        "Are you still seeing a doctor or health care provider for prostate cancer?",
                        "Are you still seeing a doctor or health care provider for pseudogout (CPPD)?",
                        "Are you still seeing a doctor or health care provider for pulmonary embolism or deep vein thrombosis (DVT)?",
                        "Are you still seeing a doctor or health care provider for reactions to anesthesia (such as hyperthermia)?",
                        "Are you still seeing a doctor or health care provider for recurrent urinary tract infections (UTI)/bladder infections?",
                        "Are you still seeing a doctor or health care provider for recurrent yeast infections?",
                        "Are you still seeing a doctor or health care provider for restless leg syndrome?",
                        "Are you still seeing a doctor or health care provider for rheumatoid arthritis (RA)?",
                        "Are you still seeing a doctor or health care provider for schizophrenia?",
                        "Are you still seeing a doctor or health care provider for severe acute respiratory syndrome (SARS)?",
                        "Are you still seeing a doctor or health care provider for severe hearing loss or partial deafness in one or both ears?",
                        "Are you still seeing a doctor or health care provider for sexually transmitted infections (gonorrhea, syphilis, chlamydia)?",
                        "Are you still seeing a doctor or health care provider for shingles?",
                        "Are you still seeing a doctor or health care provider for sickle cell disease?",
                        "Are you still seeing a doctor or health care provider for skin cancer?",
                        "Are you still seeing a doctor or health care provider for skin condition(s) (e.g., eczema, psoriasis)?",
                        "Are you still seeing a doctor or health care provider for sleep apnea?",
                        "Are you still seeing a doctor or health care provider for spinal cord injury or impairment?",
                        "Are you still seeing a doctor or health care provider for spine, muscle, or bone disorders (non-cancer)?",
                        "Are you still seeing a doctor or health care provider for stomach cancer?",
                        "Are you still seeing a doctor or health care provider for systemic lupus?",
                        "Are you still seeing a doctor or health care provider for thyroid cancer?",
                        "Are you still seeing a doctor or health care provider for tinnitus?",
                        "Are you still seeing a doctor or health care provider for transient ischemic attacks (TIAs or mini-strokes)?",
                        "Are you still seeing a doctor or health care provider for traumatic brain injury (TBI)?",
                        "Are you still seeing a doctor or health care provider for tuberculosis?",
                        "Are you still seeing a doctor or health care provider for type 1 diabetes?",
                        "Are you still seeing a doctor or health care provider for type 2 diabetes?",
                        "Are you still seeing a doctor or health care provider for ulcerative colitis?",
                        "Are you still seeing a doctor or health care provider for vitamin B deficiency?",
                        "Are you still seeing a doctor or health care provider for vitamin D deficiency?",
                        "Cigar Smoking: Cigar Smoke Participant","Cigar Smoking: Current Cigar Frequency",
                        "Disability: Blind","Disability: Deaf","Disability: Difficulty Concentrating",
                        "Disability: Dressing Bathing","Disability: Errands Alone","Disability: Walking Climbing",
                        "Discrimination: What do you think is the main reason for these experiences?",
                        "Do you speak a language other than English at home?","Education Level: Highest Grade",
                        "Electronic Smoking: Electric Smoke Frequency","Electronic Smoking: Electric Smoke Participant",
                        "Employment: Employment Status","Health Insurance: Health Insurance Type",
                        "Health Insurance: Insurance Type Update","Home Own: Current Home Own",
                        "Hookah Smoking: Current Hookah Frequency","Hookah Smoking: Hookah Smoke Participant",
                        "How much do you know about illnesses or health problems for your parents, grandparents, brothers, sisters, and/or children?",
                        "How much you agree or disagree that in your neighborhood people watch out for each other?",
                        "How much you agree or disagree that people around here are willing to help their neighbor?",
                        "How much you agree or disagree that people in your neighborhood can be trusted?",
                        "How much you agree or disagree that people in your neighborhood generally get along with each other?",
                        "How much you agree or disagree that people in your neighborhood share the same values?",
                        "How much you agree or disagree that people in your neighborhood take good care of their houses and apartments?",
                        "How much you agree or disagree that there are lot of abandoned buildings in your neighborhood?",
                        "How much you agree or disagree that there are too many people hanging around on the streets near your home?",
                        "How much you agree or disagree that there is a lot of crime in your neighborhood?",
                        "How much you agree or disagree that there is a lot of graffiti in your neighborhood?",
                        "How much you agree or disagree that there is too much alcohol use in your neighborhood?",
                        "How much you agree or disagree that there is too much drug use in your neighborhood?",
                        "How much you agree or disagree that vandalism is common in your neighborhood?",
                        "How much you agree or disagree that you are always having trouble with your neighbors?",
                        "How much you agree or disagree that your neighborhood is clean?",
                        "How much you agree or disagree that your neighborhood is noisy?",
                        "How much you agree or disagree that your neighborhood is safe?",
                        "How often are you treated with less courtesy than other people when you go to a doctor's office or other health care provider?",
                        "How often are you treated with less respect than other people when you go to a doctor's office or other health care provider?",
                        "How often do you desire to be closer to or in union with God (or a higher power)?",
                        "How often do you feel God's (or a higher power's) love for you, directly or through others?",
                        "How often do you feel God's (or a higher power's) presence?","How often do you feel deep inner peace or harmony?",
                        "How often do you feel isolated from others?","How often do you feel lack companionship?",
                        "How often do you feel left out?",
                        "How often do you feel like a doctor or nurse is not listening to what you were saying. when you go to a doctor's office or other health care provider?",
                        "How often do you feel that people are around you but not with you?","How often do you feel that there is no one you can turn to?",
                        "How often do you feel that you are an outgoing person?","How often do you feel that you are spiritually touched by the beauty of creation?",
                        "How often do you feel that you are unhappy being so withdrawn?","How often do you fell that you can find companionship when you want it?",
                        "How often do you find strength and comfort in your religion?","How often do you go to religious meetings or services?",
                        "How often do you have someone to have a good time with?","How often do you have someone to help you if you were confined to bed?",
                        "How often do you have someone to help you with daily chores if you were sick?","How often do you have someone to love and make you feel wanted?",
                        "How often do you have someone to prepare your meals if you were unable to do it yourself?","How often do you have someone to take you to the doctor if you need it?",
                        "How often do you have someone to turn to for suggestions about how to deal with a personal problem?","How often do you have someone who understands your problems?",
                        "How often do you receive poorer service than others when you go to a doctor's office or other health care provider?",
                        "How often does a doctor or nurse act as if he or she is afraid of you when you go to a doctor's office or other health care provider?",
                        "How often does a doctor or nurse act as if he or she is better than you when you go to a doctor's office or other health care provider?",
                        "How often does a doctor or nurse act as if he or she thinks you are not smart when you go to a doctor's office or other health care provider?",
                        "In the last 12 months, how many times have you or your family moved from one home to another? Number of moves in past 12 months:",
                        "In the last month, how often have you been able to control irritations in your life?",
                        "In the last month, how often have you been angered because of things that were outside of your control?",
                        "In the last month, how often have you been upset because of something that happened unexpectedly?",
                        "In the last month, how often have you felt confident about your ability to handle your personal problems?",
                        "In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?",
                        "In the last month, how often have you felt that things were going your way?","In the last month, how often have you felt that you were on top of things?","In the last month, how often have you felt that you were unable to control the important things in your life?","In the last month, how often have you found that you could not cope with all the things that you had to do?","In your day-to-day life, how often are you called names or insulted?","In your day-to-day life, how often are you threatened or harassed?","In your day-to-day life, how often are you treated with less courtesy than other people?","In your day-to-day life, how often are you treated with less respect than other people?","In your day-to-day life, how often do people act as if they are afraid of you?","In your day-to-day life, how often do people act as if they think you are dishonest?","In your day-to-day life, how often do people act as if they think you are not smart?","In your day-to-day life, how often do people act as if they're better than you are?","In your day-to-day life, how often do you receive poorer service than other people at restaurants or stores?","Including yourself, who in your family has had Crohn's disease? Select all that apply.","Including yourself, who in your family has had Lou Gehrig's disease (amyotrophic lateral sclerosis)? Select all that apply.","Including yourself, who in your family has had Parkinson's disease? Select all that apply.","Including yourself, who in your family has had a bleeding disorder? Select all that apply.","Including yourself, who in your family has had a drug use disorder? Select all that apply.","Including yourself, who in your family has had a heart attack? Select all that apply.","Including yourself, who in your family has had a hernia? Select all that apply.","Including yourself, who in your family has had a liver condition (e.g., cirrhosis)? Select all that apply.","Including yourself, who in your family has had a pancreatitis? Select all that apply.","Including yourself, who in your family has had a personality disorder? Select all that apply.","Including yourself, who in your family has had a social phobia? Select all that apply.","Including yourself, who in your family has had a stroke? Select all that apply.","Including yourself, who in your family has had acid reflux? Select all that apply.","Including yourself, who in your family has had acne? Select all that apply.","Including yourself, who in your family has had acute kidney disease with no current dialysis? Select all that apply.","Including yourself, who in your family has had alcohol use disorder? Select all that apply.","Including yourself, who in your family has had allergies? Select all that apply.","Including yourself, who in your family has had an eating disorder? Select all that apply.","Including yourself, who in your family has had an enlarged prostate? Select all that apply.","Including yourself, who in your family has had anemia? Select all that apply.","Including yourself, who in your family has had anxiety reaction/panic disorder? Select all that apply.","Including yourself, who in your family has had aortic aneurysm? Select all that apply.","Including yourself, who in your family has had asthma? Select all that apply.","Including yourself, who in your family has had astigmatism? Select all that apply.","Including yourself, who in your family has had atrial fibrillation (or a-fib) or atrial flutter (or a-flutter)? Select all that apply.","Including yourself, who in your family has had attention-deficit/hyperactivity disorder (ADHD)? Select all that apply.","Including yourself, who in your family has had autism spectrum disorder? Select all that apply.","Including yourself, who in your family has had bipolar disorder? Select all that apply.","Including yourself, who in your family has had blindness, all causes? Select all that apply.","Including yourself, who in your family has had blood or soft tissue cancer? Select all that apply.","Including yourself, who in your family has had bone cancer? Select all that apply.","Including yourself, who in your family has had bowel obstruction? Select all that apply.","Including yourself, who in your family has had brain cancer? Select all that apply.","Including yourself, who in your family has had carpal tunnel syndrome? Select all that apply.","Including yourself, who in your family has had cataracts? Select all that apply.","Including yourself, who in your family has had celiac disease? Select all that apply.","Including yourself, who in your family has had cerebral palsy? Select all that apply.","Including yourself, who in your family has had chronic fatigue? Select all that apply.","Including yourself, who in your family has had chronic lung disease (COPD, emphysema, or bronchitis)? Select all that apply.","Including yourself, who in your family has had colon polyps? Select all that apply.","Including yourself, who in your family has had concussion or loss of consciousness? Select all that apply.","Including yourself, who in your family has had congestive heart failure? Select all that apply.","Including yourself, who in your family has had coronary artery/coronary heart disease? Select all that apply.","Including yourself, who in your family has had dementia (includes Alzheimer's, vascular, etc.)? Select all that apply.","Including yourself, who in your family has had depression? Select all that apply.","Including yourself, who in your family has had diverticulitis/diverticulosis? Select all that apply.","Including yourself, who in your family has had dry eyes? Select all that apply.","Including yourself, who in your family has had endocrine cancer? Select all that apply.","Including yourself, who in your family has had endometriosis? Select all that apply.","Including yourself, who in your family has had epilepsy or seizure? Select all that apply.","Including yourself, who in your family has had esophageal cancer? Select all that apply.","Including yourself, who in your family has had eye cancer? Select all that apply.","Including yourself, who in your family has had farsightedness? Select all that apply.","Including yourself, who in your family has had fibroids? Select all that apply.","Including yourself, who in your family has had fibromyalgia? Select all that apply.","Including yourself, who in your family has had fractured/broken bones in the last five years? Select all that apply.","Including yourself, who in your family has had gall stones? Select all that apply.","Including yourself, who in your family has had glaucoma? Select all that apply.","Including yourself, who in your family has had gout? Select all that apply.","Including yourself, who in your family has had head and neck cancer? (This includes cancers of the mouth, sinuses, nose, or throat. This does not include brain cancer.) Select all that apply.","Including yourself, who in your family has had heart valve disease? Select all that apply.","Including yourself, who in your family has had hemorrhoids? Select all that apply.","Including yourself, who in your family has had high blood pressure (hypertension)? Select all that apply.","Including yourself, who in your family has had high cholesterol? Select all that apply.","Including yourself, who in your family has had hyperthyroidism? Select all that apply.","Including yourself, who in your family has had hypothyroidism? Select all that apply.","Including yourself, who in your family has had insomnia? Select all that apply.","Including yourself, who in your family has had irritable bowel syndrome (IBS)? Select all that apply.","Including yourself, who in your family has had kidney cancer? Select all that apply.","Including yourself, who in your family has had kidney disease with dialysis? Select all that apply.","Including yourself, who in your family has had kidney disease without dialysis? Select all that apply.","Including yourself, who in your family has had kidney stones? Select all that apply.","Including yourself, who in your family has had lung cancer? Select all that apply.","Including yourself, who in your family has had macular degeneration? Select all that apply.","Including yourself, who in your family has had memory loss or impairment? Select all that apply.","Including yourself, who in your family has had migraine headaches? Select all that apply.","Including yourself, who in your family has had multiple sclerosis (MS)? Select all that apply.","Including yourself, who in your family has had muscular dystrophy (MD)? Select all that apply.","Including yourself, who in your family has had narcolepsy? Select all that apply.","Including yourself, who in your family has had nearsightedness? Select all that apply.","Including yourself, who in your family has had neuropathy? Select all that apply.","Including yourself, who in your family has had obesity? Select all that apply.","Including yourself, who in your family has had osteoarthritis? Select all that apply.","Including yourself, who in your family has had osteoporosis? Select all that apply.","Including yourself, who in your family has had other arthritis? Select all that apply.","Including yourself, who in your family has had other bone, joint, or muscle condition(s)? Select all that apply.","Including yourself, who in your family has had other brain or nervous system condition(s)? Select all that apply.","Including yourself, who in your family has had other condition(s)? Select all that apply.","Including yourself, who in your family has had other digestive condition? Select all that apply.","Including yourself, who in your family has had other hearing or eye condition(s)? Select all that apply.","Including yourself, who in your family has had other heart or blood condition? Select all that apply.","Including yourself, who in your family has had other hormone/endocrine condition? Select all that apply.","Including yourself, who in your family has had other kidney condition(s)? Select all that apply.","Including yourself, who in your family has had other lung condition(s)? Select all that apply.","Including yourself, who in your family has had other mental or substance use condition? Select all that apply.","Including yourself, who in your family has had other/unknown diabetes? Select all that apply.","Including yourself, who in your family has had other/unknown thyroid condition? Select all that apply.","Including yourself, who in your family has had pancreatic cancer? Select all that apply.","Including yourself, who in your family has had peptic (stomach) ulcers? Select all that apply.","Including yourself, who in your family has had peripheral vascular disease? Select all that apply.","Including yourself, who in your family has had polycystic ovarian syndrome? Select all that apply.","Including yourself, who in your family has had post-traumatic stress disorder (PTSD)? Select all that apply.","Including yourself, who in your family has had prediabetes? Select all that apply.","Including yourself, who in your family has had prostate cancer? Select all that apply.","Including yourself, who in your family has had pseudogout (CPPD)? Select all that apply.","Including yourself, who in your family has had pulmonary embolism or deep vein thrombosis (DVT)? Select all that apply.","Including yourself, who in your family has had reactions to anesthesia (such as hyperthermia)? Select all that apply.","Including yourself, who in your family has had restless leg syndrome? Select all that apply.","Including yourself, who in your family has had rheumatoid arthritis (RA)? Select all that apply.","Including yourself, who in your family has had schizophrenia? Select all that apply.","Including yourself, who in your family has had severe hearing loss or partial deafness in one or both ears? Select all that apply.","Including yourself, who in your family has had sickle cell disease? Select all that apply.","Including yourself, who in your family has had skin cancer? Select all that apply.","Including yourself, who in your family has had skin condition(s) (e.g., eczema, psoriasis)? Select all that apply.","Including yourself, who in your family has had sleep apnea? Select all that apply.","Including yourself, who in your family has had spinal cord injury or impairment? Select all that apply.","Including yourself, who in your family has had spine, muscle, or bone disorders (non-cancer)? Select all that apply.","Including yourself, who in your family has had stomach cancer? Select all that apply.","Including yourself, who in your family has had systemic lupus? Select all that apply.","Including yourself, who in your family has had thyroid cancer? Select all that apply.","Including yourself, who in your family has had tinnitus? Select all that apply.","Including yourself, who in your family has had transient ischemic attacks (TIAs or mini-strokes)? Select all that apply.","Including yourself, who in your family has had traumatic brain injury (TBI)? Select all that apply.","Including yourself, who in your family has had type 1 diabetes? Select all that apply.","Including yourself, who in your family has had type 2 diabetes? Select all that apply.","Including yourself, who in your family has had ulcerative colitis? Select all that apply.","Including yourself, who in your family has had vitamin B deficiency? Select all that apply.","Including yourself, who in your family has had vitamin D deficiency? Select all that apply.","Income: Annual Income","Insurance: Health Insurance","It is within a 10-15 minute walk to a transit stop (such as bus, train, trolley, or tram) from my home. Would you say that you...","Living Situation: Stable House Concern","Many shops, stores, markets or other places to buy things I need are within easy walking distance of my home. Would you say that you...","Marital Status: Current Marital Status","My neighborhood has several free or low-cost recreation facilities, such as parks, walking trails, bike paths, recreation centers, playgrounds, public swimming pools, etc. Would you say that you...","Organ Transplant: Organ Transplant Description","Outside Travel 6 Month: Outside Travel 6 Month How Long","Overall Health: Average Fatigue 7 Days","Overall Health: Average Pain 7 Days","Overall Health: Difficult Understand Info","Overall Health: Emotional Problem 7 Days","Overall Health: Everyday Activities","Overall Health: General Health","Overall Health: General Mental Health","Overall Health: General Physical Health","Overall Health: General Quality","Overall Health: General Social","Overall Health: Health Material Assistance","Overall Health: Medical Form Confidence","Overall Health: Menstrual Stopped","Overall Health: Organ Transplant","Overall Health: Outside Travel 6 Month","Overall Health: Social Satisfaction","Past 3 Month Use Frequency: Cocaine 3 Month Use","Past 3 Month Use Frequency: Hallucinogen 3 Month Use","Past 3 Month Use Frequency: Inhalant 3 Month Use","Past 3 Month Use Frequency: Marijuana 3 Month Use","Past 3 Month Use Frequency: Other 3 Month Use","Past 3 Month Use Frequency: Other Stimulant 3 Month Use","Past 3 Month Use Frequency: Prescription Opioid 3 Month Use","Past 3 Month Use Frequency: Prescription Stimulant 3 Month Use","Past 3 Month Use Frequency: Sedative 3 Month Use","Past 3 Month Use Frequency: Street Opioid 3 Month Use","Pregnancy: 1 Pregnancy Status","Since you speak a language other than English at home, we are interested in your own thoughts about how well you think you speak English. Would you say you speak English...","Smokeless Tobacco: Smokeless Tobacco Frequency","Smokeless Tobacco: Smokeless Tobacco Participant","Smoking: 100 Cigs Lifetime","Smoking: Smoke Frequency","The Basics: Sexual Orientation","The crime rate in my neighborhood makes it unsafe to go on walks at night. Would you say that you...","The crime rate in my neighborhood makes it unsafe to go on walks during the day. Would you say that you...","There are facilities to bicycle in or near my neighborhood, such as special lanes, separate paths or trails, or shared use paths for cycles and pedestrians. Would you say that you...",
                        "There are sidewalks on most of the streets in my neighborhood. Would you say that you...",
                        "What is the main type of housing in your neighborhood?",
                        "Within the past 12 months, were you worried whether the food you had bought just didn't last and you didn't have money to get more?",
                        "Within the past 12 months, were you worried whether your food would run out before you got money to buy more?"]

continuous_features = ["About how old were you when you were first told you had Crohn's disease?","About how old were you when you were first told you had HIV/AIDS?","About how old were you when you were first told you had Lou Gehrig's disease (amyotrophic lateral sclerosis)?","About how old were you when you were first told you had Lyme disease?","About how old were you when you were first told you had Parkinson's disease?","About how old were you when you were first told you had West Nile virus?","About how old were you when you were first told you had Zika virus?","About how old were you when you were first told you had a bleeding disorder?","About how old were you when you were first told you had a drug use disorder?","About how old were you when you were first told you had a heart attack?","About how old were you when you were first told you had a hernia?","About how old were you when you were first told you had a liver condition (e.g., cirrhosis)?","About how old were you when you were first told you had a personality disorder?","About how old were you when you were first told you had a skin condition(s) (e.g., eczema, psoriasis)?","About how old were you when you were first told you had a social phobia?","About how old were you when you were first told you had a stroke?","About how old were you when you were first told you had acid reflux?","About how old were you when you were first told you had acne?","About how old were you when you were first told you had acute kidney disease with no current dialysis?","About how old were you when you were first told you had alcohol use disorder?","About how old were you when you were first told you had allergies?","About how old were you when you were first told you had an eating disorder?","About how old were you when you were first told you had an enlarged prostate?","About how old were you when you were first told you had anemia?","About how old were you when you were first told you had anxiety reaction/panic disorder?","About how old were you when you were first told you had aortic aneurysm?","About how old were you when you were first told you had asthma?","About how old were you when you were first told you had astigmatism?","About how old were you when you were first told you had atrial fibrillation (a-fib) or atrial flutter (or a-flutter)?","About how old were you when you were first told you had attention-deficit/hyperactivity disorder (ADHD)?","About how old were you when you were first told you had autism spectrum disorder?","About how old were you when you were first told you had bipolar disorder?","About how old were you when you were first told you had blindness, all causes?","About how old were you when you were first told you had blood or soft tissue cancer?","About how old were you when you were first told you had bone cancer?","About how old were you when you were first told you had bowel obstruction?","About how old were you when you were first told you had brain cancer?","About how old were you when you were first told you had carpal tunnel syndrome?","About how old were you when you were first told you had cataracts?","About how old were you when you were first told you had celiac disease?","About how old were you when you were first told you had cerebral palsy?","About how old were you when you were first told you had chickenpox?","About how old were you when you were first told you had chronic fatigue?","About how old were you when you were first told you had chronic lung disease (COPD, emphysema, or bronchitis)?","About how old were you when you were first told you had chronic sinus infections?","About how old were you when you were first told you had colon polyps?","About how old were you when you were first told you had concussion or loss of consciousness?","About how old were you when you were first told you had congestive heart failure?","About how old were you when you were first told you had coronary artery/coronary heart disease?","About how old were you when you were first told you had dementia (includes Alzheimer's, vascular, etc.)?","About how old were you when you were first told you had dengue fever?","About how old were you when you were first told you had depression?","About how old were you when you were first told you had diverticulitis/diverticulosis?","About how old were you when you were first told you had dry eyes?","About how old were you when you were first told you had endocrine cancer?","About how old were you when you were first told you had endometriosis?","About how old were you when you were first told you had epilepsy or seizure?","About how old were you when you were first told you had esophageal cancer?","About how old were you when you were first told you had eye cancer?","About how old were you when you were first told you had farsightedness?","About how old were you when you were first told you had fibroids?","About how old were you when you were first told you had fibromyalgia?","About how old were you when you were first told you had fractured/broken bones?","About how old were you when you were first told you had gall stones?","About how old were you when you were first told you had glaucoma?","About how old were you when you were first told you had gout?","About how old were you when you were first told you had head and neck cancer? (This includes cancers of the mouth, sinuses, nose, or throat. This does not include brain cancer.)","About how old were you when you were first told you had heart valve disease?","About how old were you when you were first told you had hemorrhoids?","About how old were you when you were first told you had hepatitis A?","About how old were you when you were first told you had hepatitis B?","About how old were you when you were first told you had hepatitis C?","About how old were you when you were first told you had high blood pressure (hypertension)?","About how old were you when you were first told you had high cholesterol?","About how old were you when you were first told you had hyperthyroidism?","About how old were you when you were first told you had hypothyroidism?","About how old were you when you were first told you had insomnia?","About how old were you when you were first told you had irritable bowel syndrome (IBS)?","About how old were you when you were first told you had kidney cancer?","About how old were you when you were first told you had kidney disease with dialysis?","About how old were you when you were first told you had kidney disease without dialysis?","About how old were you when you were first told you had kidney stones?","About how old were you when you were first told you had lung cancer?","About how old were you when you were first told you had macular degeneration?","About how old were you when you were first told you had memory loss or impairment?","About how old were you when you were first told you had migraine headaches?","About how old were you when you were first told you had multiple sclerosis (MS)?","About how old were you when you were first told you had muscular dystrophy (MD)?","About how old were you when you were first told you had narcolepsy?","About how old were you when you were first told you had nearsightedness?","About how old were you when you were first told you had neuropathy?","About how old were you when you were first told you had obesity?","About how old were you when you were first told you had osteoarthritis?","About how old were you when you were first told you had osteoporosis?","About how old were you when you were first told you had other arthritis?","About how old were you when you were first told you had other bone, joint, or muscle condition(s)?","About how old were you when you were first told you had other brain or nervous system condition(s)?","About how old were you when you were first told you had other condition(s)?","About how old were you when you were first told you had other digestive condition(s)?","About how old were you when you were first told you had other hearing or eye condition(s)?","About how old were you when you were first told you had other heart or blood condition(s)?","About how old were you when you were first told you had other hormone/endocrine condition(s)?","About how old were you when you were first told you had other infectious disease(s)?","About how old were you when you were first told you had other kidney condition(s)?","About how old were you when you were first told you had other lung condition(s)?","About how old were you when you were first told you had other mental health or substance use condition(s)?","About how old were you when you were first told you had other/unknown diabetes?","About how old were you when you were first told you had other/unknown thyroid condition(s)?","About how old were you when you were first told you had pancreatic cancer?","About how old were you when you were first told you had pancreatitis?","About how old were you when you were first told you had peptic (stomach) ulcers?","About how old were you when you were first told you had peripheral vascular disease?","About how old were you when you were first told you had polycystic ovarian syndrome?","About how old were you when you were first told you had post-traumatic stress disorder (PTSD)?","About how old were you when you were first told you had prediabetes?","About how old were you when you were first told you had prostate cancer?","About how old were you when you were first told you had pseudogout (CPPD)?","About how old were you when you were first told you had pulmonary embolism or deep vein thrombosis (DVT)?","About how old were you when you were first told you had reactions to anesthesia (such as hyperthermia)?","About how old were you when you were first told you had recurrent urinary tract infections (UTI)/bladder infections?","About how old were you when you were first told you had recurrent yeast infections?","About how old were you when you were first told you had restless leg syndrome?","About how old were you when you were first told you had rheumatoid arthritis (RA)?","About how old were you when you were first told you had schizophrenia?","About how old were you when you were first told you had severe acute respiratory syndrome (SARS)?","About how old were you when you were first told you had severe hearing loss or partial deafness in one or both ears?","About how old were you when you were first told you had sexually transmitted infections (gonorrhea, syphilis, chlamydia)?","About how old were you when you were first told you had shingles?","About how old were you when you were first told you had sickle cell disease?","About how old were you when you were first told you had skin cancer?","About how old were you when you were first told you had sleep apnea?","About how old were you when you were first told you had spinal cord injury or impairment?","About how old were you when you were first told you had spine, muscle, or bone disorders (non-cancer)?","About how old were you when you were first told you had stomach cancer?","About how old were you when you were first told you had systemic lupus?","About how old were you when you were first told you had thyroid cancer?","About how old were you when you were first told you had tinnitus?","About how old were you when you were first told you had transient ischemic attacks (TIAs or mini-strokes)?","About how old were you when you were first told you had traumatic brain injury (TBI)?","About how old were you when you were first told you had tuberculosis?","About how old were you when you were first told you had type 1 diabetes?","About how old were you when you were first told you had type 2 diabetes?","About how old were you when you were first told you had ulcerative colitis?","About how old were you when you were first told you had vitamin B deficiency?","About how old were you when you were first told you had vitamin D deficiency?","Alcohol: 6 or More Drinks Occurrence","Alcohol: Average Daily Drink Count","Attempt Quit Smoking: Completely Quit Age","Living Situation: How Many People","Living Situation: People Under 18","Smoking: Average Daily Cigarette Number","Smoking: Current Daily Cigarette Number","Smoking: Daily Smoke Starting Age","Smoking: Number Of Years","Smoking: Serious Quit Attempt"]

# Make conversions

## binary
for col in binary_numeric_features:
    if col in merged_df_clean.columns:
        # Convert text-like values to numeric 0/1 if necessary
        merged_df_clean[col] = (
            merged_df_clean[col]
            .replace({
                "Yes": 1, "No": 0,
                "True": 1, "False": 0,
                True: 1, False: 0
            })
            .astype(float)
        )
        
## categorical
for col in categorical_features:
    if col in merged_df_clean.columns:
        merged_df_clean[col] = merged_df_clean[col].astype("category")

## continuous
for col in continuous_features:
    if col in merged_df_clean.columns:
        merged_df_clean[col] = pd.to_numeric(merged_df_clean[col], errors="coerce")

merged_df_clean.to_pickle("./data/merged_df_clean.pkl")
