# Telco Customer Churn Prediction

This project analyzes customer data from a telecommunications company to predict customer churn and support retention strategies. The goal is to identify customers at high risk of leaving and provide insights for improving customer loyalty.

## Project Overview
- Use descriptive analytics and machine learning to understand churn behavior.
- Evaluate customer attributes, subscription details, and service usage patterns.
- Build a churn prediction model that helps prioritize retention efforts.

## Dataset
- Source: [Kaggle Telco Customer Churn IBM Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- Records: 7,043 customers
- Features: 21 columns
- Churn rate: 26.6% of customers left the service

## Dashboard
- Interactive report: [Google Data Studio](https://datastudio.google.com/reporting/61718e18-0add-4a5c-83c7-e4ebdc94ad8f)

## Data Dictionary
- **customerID** - Unique customer identifier
- **gender** - Customer gender
- **SeniorCitizen** - Whether the customer is a senior citizen
- **Partner** - Whether the customer has a partner
- **Dependents** - Whether the customer has dependents
- **tenure** - Number of months the customer has been with the company
- **PhoneService** - Whether the customer has a phone service
- **MultipleLines** - Whether the customer has multiple phone lines
- **InternetService** - Customer internet service type
- **OnlineSecurity** - Whether the customer has online security add-on
- **OnlineBackup** - Whether the customer has online backup add-on
- **DeviceProtection** - Whether the customer has device protection add-on
- **TechSupport** - Whether the customer has tech support add-on
- **StreamingTV** - Whether the customer has streaming TV service
- **StreamingMovies** - Whether the customer has streaming movies service
- **Contract** - Customer contract term
- **PaperlessBilling** - Whether the customer uses paperless billing
- **PaymentMethod** - Customer payment method
- **MonthlyCharges** - Monthly charge amount
- **TotalCharges** - Total charges to date
- **Churn** - Whether the customer has churned (Yes or No)

## Key Objectives
- Analyze churn drivers using customer profile and service data
- Perform data cleaning, feature engineering, and exploratory analysis
- Train and evaluate models to predict churn probability
- Present findings and recommendations for customer retention

## Process
1. Data loading and validation
2. Data cleaning and missing value handling
3. Feature engineering and categorical encoding
4. Exploratory data analysis and visualization
5. Model training, validation, and performance evaluation
6. Insights extraction and business recommendations

## Interesting Insights
- 26.6% of customers churned while 73.4% remained.
- Tenure: most customers stay 9–55 months, but churners are concentrated in 2–29 months.
- Monthly charges: churners are commonly in the 56.15–94.20 range.
- Total charges: churners typically fall between 134 and 2,331.
- Gender has minimal impact on churn.
- Senior citizens have a much higher churn rate (41.7%) than non-seniors (23.7%).
- Customers without a partner churn at 33.0%, compared to 19.7% for those with a partner.
- Customers with no dependents churn at 31.3%, while customers with dependents churn at only 15.5%.
- Fiber optic users churn at 41.9%, compared to 19.0% for DSL users.
- Customers without Online Security churn at 41.8%, while those with it churn at 14.6%.
- No Online Backup corresponds to 39.9% churn versus 21.6% with backup.
- No Device Protection corresponds to 39.1% churn versus 22.5% with protection.
- No Tech Support corresponds to 41.6% churn versus 15.2% with support.
- Month-to-month contract customers churn at 42.7%, while two-year contracts churn at only 2.8%.
- Paperless billing customers churn at 33.6%, compared to 16.4% for paper billing.
- Electronic Check users have the highest churn rate at 45.3%.

## Model Results
- Evaluation metric: ROC AUC
- Best model: SVC with SMOTE
- Best AUC score: 0.74
- The model demonstrates good separation of churn and non-churn cases for this dataset.

## Conclusion
The analysis shows the strongest churn drivers are contract type, service add-ons, and payment method. Customers on month-to-month plans, using fiber optic service, lacking support or security, or paying by electronic check are most likely to churn.

This project highlights how targeted retention strategies can be built from data-driven insights, helping telecommunications providers reduce churn and improve customer loyalty.
