# Input requirements

The sections below provide the expected columns for the input dataframe required for all three pipeline types: continuous, dichotomous, and categorical, as well as an indicator whether they are required for all uses of the bopforge package or internal IHME-specific requirements.

## Continuous pipeline

Input variables required for the continuous BoP pipeline, including variable name, data type, requirement status – all or internal (IHME-specific), and a brief description of the parameter. Variables listed here are for all models (external and internal).

| Column Name          | type          | Required | Description | Notes |
|----------------------|---------------|----------|-------------|-------|
| seq                  | `int`            | all      | Unique ID for each row of input data. For internal models, corresponds to a `seq` from the crosswalk's `bundle_version`, or a new `seq` if the row was the result of a splitting operation during crosswalking. |  |
| study_id             | `int`            | all      | ID for the associated study; all rows containing information from the same study must have the same study_id. For internal models, NID of the study | |
| risk_type            | `str`            | all      | Type of risk factor assessed; "continuous" | |
| risk_unit            | `str`            | all      | Unit of the risk factor assessed, e.g., "g/day". Will be the same for each row. | |
| ln_rr                | `float`          | all      | Natural log-transformed RR of the risk-outcome pair; values from $-\text{infinity}$ to $+\text{infinity}$. | |
| ln_rr_se             | `float`          | all      | Standard error of the natural log-transformed RR of the risk-outcome pair; values > 0. If $SE(RR)$ is reported, $SE(ln(RR)) = \frac{SE(RR)}{RR}$. | |
| ref_risk_lower       | `float`          | all      | Lower bound of the reference exposure category of the risk factor; values typically $\geq 0$. | |
| ref_risk_upper       | `float`          | all      | Upper bound of the reference exposure category of the risk factor; values typically $\geq 0$ and greater than ref_risk_lower. | |
| alt_risk_lower       | `float`          | all      | Lower bound of the alternative exposure category of the risk factor; values typically $\geq 0$. | |
| alt_risk_upper       | `float`          | all      | Upper bound of the alternative exposure category of the risk factor; values typically $\geq 0$ and greater than alt_risk_lower. | |
| cov_dummy            | `int`            | all      | Binary bias covariate that measures a source of bias in the data/study; 0 if study has gold standard and 1 otherwise. | |


## Dichotomous pipeline

Input variables required for the dichotomous BoP pipeline, including variable name, data type, requirement status – all or internal (IHME-specific), and a brief description of the parameter. Variables listed here are for all models (external and internal).

| Column Name          | type          | Required | Description | Notes |
|----------------------|---------------|----------|-------------|-------|
| seq                  | `int`            | all      | Unique ID for each row of input data. For internal models, corresponds to a `seq` from the crosswalk's `bundle_version`, or a new `seq` if the row was the result of a splitting operation during crosswalking. |  |
| study_id             | `int`            | all      | ID for the associated study; all rows containing information from the same study must have the same study_id. For internal models, NID of the study | |
| risk_type            | `str`            | all      | Type of risk factor assessed; "dichotomous" | |
| ln_rr                | `float`          | all      | Natural log-transformed RR of the risk-outcome pair; values from $-\text{infinity}$ to $+\text{infinity}$. | |
| ln_rr_se             | `float`          | all      | Standard error of the natural log-transformed RR of the risk-outcome pair; values > 0. If $SE(RR)$ is reported, $SE(ln(RR)) = \frac{SE(RR)}{RR}$. | |
| cov_dummy            | `int`            | all      | Binary bias covariate that measures a source of bias in the data/study; 0 if study has gold standard and 1 otherwise. | |


## Categorical pipeline

Input variables required for the categorical BoP pipeline, including variable name, data type, requirement status – all or internal (IHME-specific), and a brief description of the parameter. Variables listed here are for all models (external and internal).

| Column Name          | type          | Required | Description | Notes |
|----------------------|---------------|----------|-------------|-------|
| seq                  | `int`            | all      | Unique ID for each row of input data. For internal models, corresponds to a `seq` from the crosswalk's `bundle_version`, or a new `seq` if the row was the result of a splitting operation during crosswalking. |  |
| study_id             | `int`            | all      | ID for the associated study; all rows containing information from the same study must have the same study_id. For internal models, NID of the study | |
| risk_type            | `str`            | all      | Type of risk factor assessed; "categorical" | |
| ln_rr                | `float`          | all      | Natural log-transformed RR of the risk-outcome pair; values from $-\text{infinity}$ to $+\text{infinity}$. | |
| ln_rr_se             | `float`          | all      | Standard error of the natural log-transformed RR of the risk-outcome pair; values > 0. If $SE(RR)$ is reported, $SE(ln(RR)) = \frac{SE(RR)}{RR}$. | |
| ref_risk_cat         | `str`          | all      | Reference exposure category of the risk factor. Note this category does not need to be the same across all rows. | |
| alt_risk_cat         | `str`          | all      | Alternative exposure category of the risk factor. | |
| cov_dummy            | `int` or `float`            | all      | One of three types of covariates. (1) a binary bias covariate that measures a source of bias in the data/study; 0 if study has gold standard and 1 otherwise. (2) An interacted model covariate that interacts with each category independently. (3) A non-interacted model covariate that shares effect across all categories. | |


## All pipelines – INTERNAL ONLY

Input variables required for all three BoP pipelines, including variable name, data type, requirement status – all or internal (IHME-specific), and a brief description of the parameter. The variables listed here are internal-only and are shared (and required) across all three BoP pipelines for an internal model.

| Column Name          | type          | Required | Description | Notes |
|----------------------|---------------|----------|-------------|-------|
| rei_id               | `int`            | internal | Unique ID for the risk factor assessed | |
| cause_id             | `int`            | internal | Unique ID for the cause assessed | |
| bundle_id            | `int`            | internal | Dataset identifier | |
| bundle_version_id    | `int`            | internal | Dataset identifier at a given point in time | |
| crosswalk_parent_seq | `int` or NULL    | internal | Used to track which rows in a bundle version were adjusted during crosswalking | Confirm type |
| underlying_nid       | `int` or NA      | internal | If the NID is a composite NID, `underlying_nid` is the original data source that is bundled within the NID. | Confirm type |
| location_id          | `int`            | internal | ID representing the location associated with the row of data | |
| location_name        | `str`            | internal | Name representing the location associated with the row of data | |
| sex                  | `str`            | internal | Name representing the sex associated with the row of data | Confirm type |
| year_start           | `int`            | internal | The start year associated with the row of data | Confirm type is not float  |
| year_end             | `int`            | internal | The ending year associated with the row of data | Confirm type is not float  |
| age_start            | `int`            | internal | The start age associated with the row of data | Confirm type is not float  |
| age_end              | `int`            | internal | The ending age associated with the row of data | Confirm type is not float  |
| design               | `str`            | internal | Any of `randomized controlled trial`, `non-randomized controlled trial`, `non-randomized trial`, `prospective cohort`, `retrospective cohort`, `case-control`, `nested case-control`, `case-cohort`, `case-crossover`, `ecological`, `cross-sectional`. | |
| is_outlier           | `int`            | internal | A row of data outliered during the crosswalking process. These rows can be dropped entirely, if you prefer. Note that this is not the same as the output `is_outlier` column from bopforge – the latter are outliers identified during the trimming process. Unless running an internal production model, this column should not be included. | |
| rei                  | `str`            | internal | Machine-readable short name for the risk factor assessed. | Not actually needed? Listed in MSCA docs but not in `format_crosswalk` or the input data from `bop-prod` run |
| acause               | `str`            | internal | Machine-readable short name for the health outcome assessed. | Not actually needed? Listed in MSCA docs but not in `format_crosswalk` or the input data from `bop-prod` run |