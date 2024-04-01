import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from causalinference import CausalModel
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from statsmodels.stats.diagnostic import het_white
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from fpdf import FPDF

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from tabulate import tabulate

df=pd.read_csv(r"C:\Users\isrea\OneDrive\Desktop\python regression\CSV's\RUSSIA_DATA_CSV.csv")
print(df)
print(df.isnull())

### Data_Wrangling 
print(df.isnull().sum())
plt.show()
null_values= sns.heatmap(df.isnull(), yticklabels=False)
print(null_values)
plt.show()
null_values_1= sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis')
print(null_values_1)
plt.show()

# Create a copy of the DataFrame for cleaning
df_cleaned = df.copy()

# Check and print the count of NaN values in each colum
print(df_cleaned.isnull().sum())

print(df_cleaned)
#column in concern 

# Define the columns to use
columns_to_analyse_df = [ 'Country',  'Year',   'Financial_sanction',    'Log_GDP',     'Log_inflation',      'Log_exchange',    'Log_FDI',       'Log_imp',     'Log_exp']

# Create the new DataFrame with selected columns
column_to_use = df_cleaned[columns_to_analyse_df].copy()

# Display the new DataFrame
print(column_to_use)

column_to_analyse =pd.DataFrame(column_to_use)

df_cleaned_df = pd.DataFrame(df_cleaned)
column_to_analyse.to_csv('column_to_analyse_df.txt', sep='\t', index=False)

print(column_to_analyse)
# Save the DataFrame to a text file
df_cleaned_df.to_csv('df_cleaned_df.txt', sep='\t', index=False)

# Drop rows with NaN values (this will modify df_cleaned in place)
df_cleaned.dropna(inplace=True)
print(df_cleaned.isnull())
# Create a heatmap to visualize NaN values
sns.heatmap(df_cleaned.isnull(), yticklabels=False, cbar=False)

plt.show()
# Create a countplot for the "Financial_sanction" column
outcome= sns.countplot(x="Financial_sanction", data=df_cleaned)
# Set labels and title
plt.xlabel("sanction or no sanction ")
plt.ylabel("Count")
plt.title("Count of Financial Sanction on Russia")

# Show the plot
plt.show()

# distribution of financial sanction
effect = sns.boxplot(x='Financial_sanction', y= 'Log_GDP', data=column_to_analyse)

plt.xlabel(" values_Sanction_sanction")
plt.ylabel("values_log_GDP")
plt.title("Distribution of financial_sanction against independent variables")
print(effect)
plt.show()
# Define the variables

# Define the variables
treatment_variable = 'Financial_sanction'
covariates = ['Log_GDP', 'Log_inflation', 'Log_exchange']

# Plot KDE for each covariate
for covariate in covariates:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df_cleaned, x=covariate, hue=treatment_variable, fill=True, common_norm=False)
    plt.title(f'Kernel Density Estimation (KDE){treatment_variable} across {covariate}')
    plt.xlabel(covariate)
    plt.ylabel('Density')
    
    # Customize legend
    plt.legend(title=f'{treatment_variable} and Covariates', 
               labels=[f'No {treatment_variable}, {covariate}', f'{treatment_variable}', f'{covariate}'])
    
    plt.show()
#treatment variable and covariates in df
# combined_data = df_cleaned[['Financial_sanction', 'Trade_sanction', 'Military_sanction', 'Multilateral_sender_sanction', 'Log_GDP', 'Log_inflation', 'Log_exchange', 'Log_FDI', 'Log_imp', 'Log_exp']]

# Defined treatment and outcome variables

combined_data = df_cleaned[['Financial_sanction', 'Trade_sanction', 'Military_sanction', 'Multilateral_sender_sanction', 'Log_GDP', 'Log_inflation', 'Log_exchange', 'Log_FDI', 'Log_imp', 'Log_exp']]

# Defined treatment and outcome variables
exposure_column = 'Financial_sanction'
outcome_column = 'Log_FDI'

# initializing CausalModel with combined DataFrame
causal = CausalModel(
    Y=combined_data[outcome_column].values, 
    D=combined_data[exposure_column].values, 
    X=combined_data.drop([outcome_column, exposure_column], axis=1).values)

# Estimate propensity scores
causal.est_propensity_s()
propensity_scores = causal.propensity['fitted']
print(propensity_scores)
#create dataframe for propensity score
propensity_scores_df=pd.DataFrame(propensity_scores)
#tabulation for propensity score
propensity_scores_df_str= tabulate(propensity_scores_df, headers='keys', tablefmt='grid')
print(propensity_scores_df_str)
# save the tabulated result to a text file
with open('save propensity_score_str.txt', 'w') as f:
    f.write(propensity_scores_df_str)
# Define coarsening schemas for each continuous covariate
coarsening_schemas = {
    'Log_GDP': {'num_bins': 3, 'labels': ['Low', 'Medium', 'High']},
    'Log_inflation': {'num_bins': 3, 'labels': ['Low', 'Medium', 'High']},
    'Log_exchange': {'num_bins': 3, 'labels': ['Low', 'Medium', 'High']},
}

# Initialize variables for matching
matched_data = combined_data.copy()

# Perform matching based on propensity scores and coarsening schemas
for covariate, schema in coarsening_schemas.items():
    bins = pd.cut(combined_data[covariate], bins=schema['num_bins'], labels=schema['labels'])
    matched_data[covariate] = bins
# Check the results
print(matched_data)
print(matched_data.columns)
# Creating a dataframe to saved the data 
matched_data_save= pd.DataFrame(matched_data)
#save the dataframe
save_matched_data_str = tabulate(matched_data_save, headers='keys', tablefmt='grid')
print(save_matched_data_str)
# save the tabulated result to a text file
with open('save_matched_data_str.txt', 'w') as f:
    f.write(save_matched_data_str)
from scipy.stats import chi2_contingency

# Create contingency table for Log_inflation, log_gdp, log_exchange
contingency_table = pd.crosstab(column_to_analyse['Financial_sanction'], matched_data['Log_inflation'])
alpha= 0.5
contingency_table = pd.crosstab(matched_data['Financial_sanction'], matched_data['Log_GDP'])
contingency_table = pd.crosstab(matched_data['Financial_sanction'], matched_data['Log_exchange'])
# Perform chi-square test for Log_inflation, log_gdp, log_exchange

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic for Log_inflation: {chi2}")
print(f"P-value for Log_inflation: {p_value}")
# Organize chi-square statistic and p-value into a dictionary
chi2_results = {'Chi-square statistic': chi2, 'P-value': p_value}

# Create a DataFrame from the dictionary
chi2_dataframe = pd.DataFrame(chi2_results, index=[0])

# Save the DataFrame to a text file
with open('chi2_results.txt', 'w') as f:
    f.write(chi2_dataframe.to_string(index=False))
if p_value <= alpha:
    print("Reject the null hypothesis")
    print("There is evidence to suggest an association.")
else:
    print("Fail to reject the null hypothesis")
    print("There isn't enough evidence to suggest an association.")
#other categorical variables (Log_GDP and Log_exchange)

chi2_2, p_value_2, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic for Log_GDP: {chi2_2}")
print(f"P-value for Log_GDP: {p_value_2}")

# Organize chi-square statistic and p-value into a dictionary
chi2_2_results = {'Chi-square statistic': chi2_2, 'P-value': p_value_2}

# Create a DataFrame from the dictionary
chi2_2_dataframe = pd.DataFrame(chi2_2_results, index=[0])

# Save the DataFrame to a text file
with open('chi2_results.txt', 'w') as f:
    f.write(chi2_2_dataframe.to_string(index=False))

if p_value_2 <= alpha:
    print("Reject the null hypothesis")
    print("There is evidence to suggest an association.")
else:
    print("Fail to reject the null hypothesis")
    print("There isn't enough evidence to suggest an association.")

chi2_3, p_value_3, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic for Log_exchange: {chi2_3}")
print(f"P-value for Log_exchange: {p_value_3}")

# Organize chi-square statistic and p-value into a dictionary
chi2_3_results = {'Chi-square statistic': chi2_3, 'P-value': p_value_3}

# Create a DataFrame from the dictionary
chi2_3_dataframe = pd.DataFrame(chi2_3_results, index=[0])

# Save the DataFrame to a text file
with open('chi2_results.txt', 'w') as f:
    f.write(chi2_3_dataframe.to_string(index=False))
    
if p_value <= alpha:
    print("Reject the null hypothesis")
    print("There is evidence to suggest an association.")
else:
    print("Fail to reject the null hypothesis")
    print("There isn't enough evidence to suggest an association.")

# Convert 'Financial_sanction' to a binary variable (0 or 1)

matched_data['Financial_sanction'] = matched_data['Financial_sanction'].astype('category').cat.codes
# Define the covariates used in CEM (the coarsened covariates)
coarsened_covariates = ['Log_GDP', 'Log_inflation', 'Log_exchange']

# Create a DataFrame with the matched data
matched_data = df_cleaned[['Financial_sanction'] + coarsened_covariates + ['Log_FDI']]
matched_data_log_imp = df_cleaned[['Financial_sanction'] + coarsened_covariates + ['Log_imp']]
matched_data_log_exp = df_cleaned[['Financial_sanction'] + coarsened_covariates + ['Log_exp']]

# Add a constant term to the regression model (intercept)
matched_data = sm.add_constant(matched_data)
matched_data_log_imp = sm.add_constant(matched_data_log_imp)
matched_data_log_exp = sm.add_constant(matched_data_log_exp)
# Perform linear regression
X = matched_data[['Financial_sanction'] + coarsened_covariates + ['const']]
y = matched_data[outcome_column]

model = sm.OLS(y, X).fit(alpha=0.05)

outcome_column_import = 'Log_imp'
X = matched_data_log_imp[['Financial_sanction'] + coarsened_covariates + ['const']]
y = matched_data_log_imp[outcome_column_import]

model_log_imp = sm.OLS(y, X).fit(alpha=0.05)
outcome_column_export = 'Log_exp'
X = matched_data_log_exp[['Financial_sanction'] + coarsened_covariates + ['const']]
y = matched_data_log_exp[outcome_column_export]

model_log_exp = sm.OLS(y, X).fit(alpha=0.05)

results = het_white(model.resid, X)
print("White test p-value for the main model:", results[1])
p_value = results[1]

# Save the p-value to a text file
with open('white_test_p_value.txt', 'w') as f:
    f.write(f"White test p-value for the main model: {p_value}")

results_log_imp = het_white(model_log_imp.resid, X)
print("White test p-value for the model with Log_imp:", results_log_imp[1])

results_log_exp = het_white(model_log_exp.resid, X)
print("White test p-value for the model with Log_exp:", results_log_exp[1])

# Get robust standard errors
# Get the summary of the regression results

regression_result_4_FDI = model.get_robustcov_results(cov_type='HC3')
regression_result_4_FDI_summary = regression_result_4_FDI.summary()

# Convert the summary to a DataFrame
regression_result_4_FDI_df = pd.DataFrame(regression_result_4_FDI_summary.tables[1].data)

# Save the DataFrame to a text file
regression_result_4_FDI_df.to_csv('regression_result_4_FDI.txt', sep='\t', index=False)

regression_result_4_export = model_log_exp.get_robustcov_results(cov_type='HC3')

# Get the summary of the regression results
regression_result_4_export_summary = regression_result_4_export.summary()

# Convert the summary to a DataFrame
regression_result_4_export_df = pd.DataFrame(regression_result_4_export_summary.tables[1].data)

# Save the DataFrame to a text file
regression_result_4_export_df.to_csv('regression_result_4_export.txt', sep='\t', index=False)

regression_result_4_import = model_log_imp.get_robustcov_results(cov_type='HC3')
regression_result_4_import_summary = regression_result_4_import.summary()

# Convert the summary to a DataFrame
regression_result_4_import_df = pd.DataFrame(regression_result_4_import_summary.tables[1].data)

# Save the DataFrame to a text file
regression_result_4_import_df.to_csv('regression_result_4_import.txt', sep='\t', index=False)

print(regression_result_4_FDI.summary())
print(regression_result_4_export.summary())
print(regression_result_4_import.summary())



