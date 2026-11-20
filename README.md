# Customer-Churn-Retention-Strategy

Predict which telecom customers are likely to churn, understand why, and recommend a targeted retention action for each customer segment.

| Deliverable | Link |
|---|---|
| Presentation (PPT) | https://1drv.ms/p/c/e8a8c1093a07042f/IQDbz1X-Jx9bR4MK3cggu1WVAdZGkvuE_2I85_Csvx0bMTI?e=P10OjV |
| Working demo | https://customer-churn-retention-strategy.streamlit.app/ |
| Project video | https://onedrive.live.com/?photosData=%2Fshare%2FE8A8C1093A07042F%21s368b554604f94c8a9af4b36a0f94ea8f%3Fithint%3Dvideo%26e%3DjjlZRu%26migratedtospo%3Dtrue&cid=E8A8C1093A07042F&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3YvYy9lOGE4YzEwOTNhMDcwNDJmL0lRQkdWWXMyLVFTS1RKcjBzMm9QbE9xUEFaYzFIckJMMDk3ZzZicXhnVllaRHlJP2U9ampsWlJ1&view=8 |
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
