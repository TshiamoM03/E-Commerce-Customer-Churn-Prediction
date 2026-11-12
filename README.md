# E-Commerce Customer Churn Prediction with Explainable AI

## Business Problem
Acquiring a new customer costs far more than keeping an existing one but retention budgets 
could be limited. A useful churn prediction model should therefore do the following: 
find the right customers (without wasting money on false alarms) and explain what is driving 
their risk so the right intervention can be chosen.

## What I Did
- Compared four models: Logistic Regression, Decision Tree, Random Forest and XGBoost
- Evaluated them on predictive performance (recall, precision, F1-score, PR-AUC)
- Further evaluated them on fairness across customer subgroups, computational cost and stability using cross-validation
- Used SHAP to explain churn drivers globally and for individual customers

**Tools:** Python, scikit-learn, XGBoost, SHAP, Fairlearn, psutil\
**Data:** synthetic [Ecommerce Customer Behavior Dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset) from Kaggle

## Key Results
|                     | Recall | Precision | F1-Score | PR-AUC |
|---------------------|--------|-----------|----------|--------|
| Logistic Regression | 0.72   | 0.48      | 0.58     | 0.60   |
| Decision Tree       | 0.80   | 0.67      | 0.72     | 0.78   |
| Random Forest       | 0.81   | 0.87      | 0.84     | 0.90   |
| XGBoost             | 0.82   | 0.86      | 0.84     | 0.89   |

- Tree based ensemble models clearly outperformed logistic regression and a single decision tree
- Random forest and XGBoost were statistically indistinguishable on most metrics, so the choice between them can rest on cost and preference. Note random forest took 142.8s to train versus 2.2s for XGBoost
- Logistic regression had the most false alarms and was the only model that fell below the 80% disparate impact guideline for fairness

## What Drives Churn
Behaviour mattered more than demographics. The most influential factors were:
- Frequent customer service calls
- High cart abandonment
- Longer periods of inactivity
- Fewer total purchases
- High lifetime value, meaning valuable customers were not necessarily safer from churn

<table>
  <tr>
    <td width="50%" valign="center">
        <img src="https://github.com/user-attachments/assets/bd8e4862-9653-467e-82ed-a892bc4a275a" width="100%">
    </td>
    <td width="50%" valign="center">
      <img src="https://github.com/user-attachments/assets/11718906-13ef-495f-9a96-64b073e03aea" width="100%"/>
    </td>
  </tr>
</table>


## Recommendations
1. Use a tree-based ensemble with SHAP in a CRM, so teams can see who is at risk and why.
2. Treat multiple service calls as an early warning sign - a spike should trigger an operational review of service quality
3. Target cart abandoners and inactive customers with re-engagement (reminders/notifications, personalised offers)
4. Prioritise by value: combine churn risk with CLV so limited budgets can protect the most valuable customers first
5. Audit fairness before deployment and monitor it afterwards since models can flag some groups more than others
6. Choose the model by use case: a single decision tree where transparency and low cost matter most, an ensemble where accuracy is worth the extra cost

## Limitations
The dataset was synthetic and only represented a snapshot of customer behaviour. There was no real retention cost or revenue data available, 
so thresholds were not tuned to business costs.\
SHAP explains model behaviour, not causal effect. The results of the research demonstrated the approach rather than a universal conclusion.

## Future Work
Deep learning models, richer data (review text, clickstream), training without class weighting, probability calibration, cost-sensitive thresholds 
and uplift modelling (with A/B testing) to measure which retention offers actually work.
