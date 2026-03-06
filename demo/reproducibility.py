
# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Install (if needed) and load the necessary libraries.
#
# For teaching: keep installation lines commented out so students can run them
# manually if needed.

# pip install pandas numpy matplotlib statsmodels

import os
import sys
import platform

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from pandas.plotting import table

logging.info("platform module file: " + str(platform.__file__))
logging.info("python executable: " + str(sys.executable))
logging.info("python version: " + str(sys.version))
logging.info("os/platform summary: " + str(platform.platform()))


# Python dependency management:
# - Create requirements.txt after everything runs:
#     pip freeze > requirements.txt
# - On another machine:
#     pip install -r requirements.txt

# Reproducible projects should separate:
# - raw data (unchanged inputs)
# - processed data (cleaned outputs)
# - figures and tables (final outputs)
# set working direction
BASE_DIR = "/Users/a75700/Desktop/soda_501/soda_501/02_reproducibility"
# create new file folders
os.makedirs(os.path.join(BASE_DIR, "data/raw"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data/processed"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs/figures"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs/tables"), exist_ok=True)


# Logging creates an audit trail:
# - What ran
# - In what order
# - With what parameters
# - Where outputs were written
log_path = os.path.join(BASE_DIR, "analysis_log.txt")

logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")]
)

# Pipeline overview:
#   1) Load data
#   2) Save raw data (confirm location)
#   3) Clean data
#   4) Save processed data
#   5) Run three regressions (income as DV)
#   6) Create plot(s)
#   7) Save tables + session info

np.random.seed(123)  # Reproducible randomness for the full pipeline

logging.info("Starting analysis pipeline")

# Expected location for this assignment:
# - data/raw/education_income.csv
# ADD DATA PATH
logging.info("Loading education/income dataset from data_path")
os.chdir(BASE_DIR)
education_income_raw = pd.read_csv('data/raw/education_income.csv')

logging.info("Rows loaded: " + str(education_income_raw.shape[0]))
logging.info("Columns loaded: " + str(education_income_raw.shape[1]))

# In many projects, "raw" is treated as read-only and comes from outside.
# Here we re-write it to confirm the exact file used in the run.

logging.info("Saving raw data copy (unchanged)")
# education_income_raw.to_csv("data/raw/education_income.csv", index=False)

# Keep this simple and explicit:
# - Ensure education and income exist
# - Coerce to numeric (if needed)
# - Drop missing
#
# Note: No if/else. If columns are missing, the script will error (which is fine).

logging.info("Cleaning education/income data")

education_income_clean = education_income_raw.copy()
education_income_clean["education"] = pd.to_numeric(education_income_clean["education"])
education_income_clean["income"] = pd.to_numeric(education_income_clean["income"])
education_income_clean = education_income_clean.dropna(subset=["education", "income"])

logging.info("Rows after cleaning: " + str(education_income_clean.shape[0]))

# Create log-income version for Model 3
# If income has zeros or negatives, log(income) is not finite.
education_income_clean["log_income"] = np.log(education_income_clean["income"])

education_income_log = education_income_clean.copy()
education_income_log = education_income_log.replace([np.inf, -np.inf], np.nan)
education_income_log = education_income_log.dropna(subset=["log_income"])

logging.info("Rows with finite log(income): " + str(education_income_log.shape[0]))

logging.info("Saving processed data")
education_income_clean.to_csv("data/processed/cleaned_education_income.csv", index=False)

logging.info("Fitting Model 1: income ~ education")
# Fit linear regression
import statsmodels.formula.api as smf
model_1 = smf.ols('income ~ education', data =education_income_clean).fit()
model_1.summary() # see results

# model 2
logging.info("Fitting Model 2: income ~ education + education^2")
model_2 = smf.ols('income ~ education + I(education**2)', data =education_income_clean).fit()

logging.info("Plot Model 2: income ~ education + education^2")

# model 3
logging.info("Fitting Model 3: log(income) ~ education (finite log income rows only)")
model_3 = smf.ols('log_income~education', data =education_income_log).fit()

logging.info("Plot Model 3: income ~ education + education^2")

# Save model summaries (plain text) for replication checks
logging.info("Saving regression summaries to outputs/tables/")
from pathlib import Path

models = {
    "model_1": model_1,
    "model_2": model_2,
    "model_3": model_3,
}
for name, model in models.items():
    output_path = os.path.join(
        BASE_DIR,
        "outputs/tables",
        f"{name}_summary.txt"
    ) # create output path
    with open(output_path, "w") as f:
        f.write(model.summary().as_text()) # save model.text to output directions

# plot regression
import matplotlib.pyplot as plt
import seaborn as sns

# make sure output directory exist
fig_dir = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(fig_dir, exist_ok=True)

# plot x grid
x_vals = np.linspace(
    education_income_clean["education"].min(),
    education_income_clean["education"].max(),
    200
)
x_pred = pd.DataFrame({"education": x_vals})

plots = [
    {
        "name": "model_1",
        "model": model_1,
        "df": education_income_clean,
        "y_col": "income",
        "title": "Model 1: Income ~ Education (OLS)",
        "ylabel": "Income",
        "filename": "model_1_income_education.png",
    },
    {
        "name": "model_2",
        "model": model_2,
        "df": education_income_clean,
        "y_col": "income",
        "title": "Model 2: Income ~ Education + Education^2 (OLS)",
        "ylabel": "Income",
        "filename": "model_2_income_education_sq.png",
    },
    {
        "name": "model_3",
        "model": model_3,
        "df": education_income_log,
        "y_col": "log_income",
        "title": "Model 3: log(Income) ~ Education (OLS)",
        "ylabel": "log(Income)",
        "filename": "model_3_log_income_education.png",
    },
]

for p in plots:
    plt.figure(figsize=(7, 5))

    # scatter
    sns.scatterplot(
        data=p["df"],
        x="education",
        y=p["y_col"],
        alpha=0.6
    )

    # line/curve from model predictions
    y_hat = p["model"].predict(x_pred)
    plt.plot(x_vals, y_hat, color="red", label="OLS fit")

    plt.title(p["title"])
    plt.xlabel("Education")
    plt.ylabel(p["ylabel"])
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(fig_dir, p["filename"])
    plt.savefig(out_path, dpi=300)  # save it to ouputs/figures
    plt.close()

    logging.info(f"Saved plot to output.figures")

# TODO: create and write a regression_coefficients.csv table
def coef_table(model, model_name):
    # take coefficient statsmodels
    t = model.summary2().tables[1].copy()
    t.insert(0, "model", model_name) # add model name
    return t

# merge results into a table
coef_df = pd.concat(
    [
        coef_table(model_1, "model_1"),
        coef_table(model_2, "model_2"),
        coef_table(model_3, "model_3"),
    ],
    ignore_index=True
)

# save table
out_path = os.path.join(BASE_DIR, "outputs/tables", "regression_coefficients.csv")
coef_df.to_csv(out_path, index=False)

logging.info(f"Saved regression coefficient table to {out_path}")

# TODO (students):
# Write session info output to outputs/session_info.txt
session_info_path = os.path.join(BASE_DIR, "outputs", "session_info.txt")

with open(session_info_path, "w") as f:
    f.write("Platform module file: " + str(platform.__file__) + "\n")
    f.write("Python executable: " + str(sys.executable) + "\n")
    f.write("Python version: " + str(sys.version) + "\n")
    f.write("OS / platform: " + str(platform.platform()) + "\n")

logging.info("Saving session information")
# TODO: write session info to outputs/session_info.txt

# TODO (students):
# - After everything runs, snapshot dependencies.
# - Commit requirements.txt to GitHub.

logging.info("Snapshotting dependencies to requirements.txt")
# TODO: run outside Python:
#   pip freeze > requirements.txt

logging.info("Analysis pipeline completed successfully")
