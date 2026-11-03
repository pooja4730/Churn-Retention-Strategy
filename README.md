# Customer-Churn-Retention-Strategy

Predict which telecom customers are likely to churn, understand why, and recommend a targeted retention action for each customer segment.

| Deliverable | Link |
|---|---|
| Presentation (PPT) | [add link] |
| Working demo | [add link] |
| Project video | [add link] |
| Insights report | [reports/](reports/) |

---

## 1. Problem Statement

Customer churn directly reduces recurring revenue, and winning a new customer costs more than keeping an existing one. This project:

1. Builds a **churn prediction model** from customer account data.
2. Groups customers into **actionable segments** and explains what drives churn in each.
3. Provides a **retention playbook** (trigger, action, owner, timing, KPI) per segment.
4. Ships a **web app** that scores a customer and recommends an action.

## 2. Dataset

- **Telco Customer Churn** dataset (`data/telco.csv`): about 7,000 customers, 21 columns.
- Target: `Churn` (Yes/No). Roughly a quarter of customers churn.
- Features: demographics, contract and billing details, subscribed services, monthly and total charges.
- `data/customers_scored.csv`: every customer with their predicted churn probability, risk tier and segment.
- The dataset has no raw usage logs. Engagement is approximated with tenure, number of add-on services and whether the customer has tech support or security.

## 3. Approach

1. **Cleaning:** converted `TotalCharges` to numeric (blank for new customers) and dropped `customerID`.
2. **EDA:** churn by contract, tenure, internet service, support, payment method and charges.
3. **Feature engineering** (`src/features.py`): `num_services`, `has_support`, `avg_monthly_spend`, `charge_per_service`, `tenure_group`.
4. **Modeling:** stratified train/test split, preprocessing inside a scikit-learn Pipeline (no data leakage), class imbalance handling, and comparison of Logistic Regression, Random Forest and XGBoost.
5. **Explainability:** coefficients / SHAP show the top churn drivers behind each
