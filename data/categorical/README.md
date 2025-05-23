# Data Format

The data file need to stored in name `{rei}-{acause}.csv`.
This name will be used through out the process.

**Columns of the data frame**
* `seq`: unique row id
* `study_id`: usually equivalent with `nid` but with minor exceptions
* `risk_type`: type of the risk, continuous, dichotomous, categorical, etc
* `ln_rr`: mean of log relative risk
* `ln_rr_se`: standard error of the log relative risk
* `ref_risk_cat`: reference exposure category of the risk factor
* `alt_risk_cat`: alternative exposure category of the risk factor
* `cov_{name}`: bias covariates, usually related to study design


**Settings**

`settings.yaml` file contains settings for actions in the process, main actions including
* `fit_signal_model`: fit signal model
* `select_bias_covs`: select bias covariates
* `fit_linear_model`: fit final linear model
