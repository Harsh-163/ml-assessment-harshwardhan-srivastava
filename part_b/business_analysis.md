# Part B: Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### (a) Machine Learning Problem Formulation

**Target Variable:** `items_sold` (sales volume / number of items sold per store per month)

**Candidate Input Features:**
- Store attributes: `store_size`, `location_type` (urban/semi-urban/rural)
- Promotion features: `promotion_type` (Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points Bonus)
- Temporal features: `month`, `is_weekend`, `is_festival`
- Contextual features: `competition_density`, `monthly_footfall`, `customer_demographics`

**Type of ML Problem:** This is a **multi-class classification problem** if the goal is to *recommend which promotion to deploy* (choosing among 5 promotion types), or a **regression problem** if the goal is to *predict items sold* given a promotion choice. The most actionable framing is **classification** — the model predicts the best promotion type for each store-month combination.

**Justification:** Classification directly answers the business question ("which promotion should we deploy?") and produces an interpretable recommendation per store per month.

---

### (b) Why Items Sold (Sales Volume) is a Better Target than Total Sales Revenue

Using **items sold (sales volume)** rather than **total sales revenue** is more appropriate for several reasons:

1. **Promotion impact is volume-driven:** Promotions like BOGO and Flat Discounts reduce unit price, which suppresses revenue even when they successfully drive customer traffic and purchase volume. Measuring revenue would make effective volume-driving promotions appear to be failures.

2. **Avoids price distortion:** Revenue confounds two variables — price and quantity. A promotion that sells 200 items at ₹50 each (₹10,000) appears worse than selling 50 items at ₹250 each (₹12,500), even though the former likely drove more customer engagement.

3. **Broader ML principle — Target Variable Alignment:** The target variable must directly reflect the business objective. Since the company wants to *maximise the number of items sold*, the target must measure exactly that. Using a proxy metric (revenue) introduces noise and misaligns model optimisation with business goals, leading to models that are technically accurate but practically misleading.

---

### (c) Alternative to a Single Global Model

A **single global model** across all 50 stores ignores the heterogeneity in store contexts. Urban stores may respond strongly to BOGO promotions due to high footfall, while rural stores may respond better to Flat Discounts due to price sensitivity.

**Proposed Alternative Strategy: Stratified / Hierarchical Modelling**

- **Option 1 — Cluster-then-Model:** Use unsupervised learning (e.g., K-Means) to cluster stores by characteristics (location type, size, competition density, demographics). Train a separate model per cluster. This balances model specificity with data sufficiency.

- **Option 2 — Mixed Effects / Hierarchical Model:** Use a model with store-level random effects (e.g., LightGBM with store embeddings or a Bayesian hierarchical model). This allows the model to learn global promotion patterns while adapting to each store's individual behaviour.

**Justification:** Stores in different locations respond differently to the same promotion due to varying customer demographics, competition, and purchasing power. A single global model would average out these differences, producing sub-optimal recommendations for every store. A stratified approach allows the model to account for these contextual differences while still benefiting from pooled data.

---

## B2. Data and EDA Strategy

### (a) Joining Tables and Defining the Modelling Dataset

**Table Structure:**
- `transactions`: store_id, month, promotion_type, items_sold
- `store_attributes`: store_id, store_size, location_type
- `promotion_details`: promotion_type, discount_depth, promotion_category
- `calendar`: date, is_weekend, is_festival

**Join Strategy:**
1. Join `transactions` with `store_attributes` on `store_id`
2. Join with `promotion_details` on `promotion_type`
3. Join with `calendar` on `date/month`

**Grain of the Final Dataset:** Each row represents **one store × one month × one promotion**. This is the finest granularity at which a promotion decision is made, and it directly corresponds to the prediction task.

**Aggregations before modelling:**
- Aggregate `items_sold` to monthly totals per store
- Compute monthly footfall averages per store
- Encode calendar features (count of weekends and festivals per month)
- Create lag features: items sold in the previous month per store (to capture momentum)

---

### (b) EDA Strategy — Four Key Analyses

1. **Promotion Type vs. Average Items Sold (Bar Chart):**
   Compare mean items sold for each of the 5 promotion types. This reveals which promotions are most effective globally and guides baseline expectations. If one promotion dominates all others, it becomes the default fallback.

2. **Location Type × Promotion Type Interaction Heatmap:**
   A heatmap of average items sold for each (location_type, promotion_type) combination. This is the most critical EDA step — it directly answers whether the same promotion performs differently across urban, semi-urban, and rural stores, justifying the stratified modelling approach.

3. **Time Series of Items Sold per Store (Line Plot):**
   Plot monthly items sold trends for a sample of stores. This reveals seasonality, trend effects (growing/declining stores), and the impact of festival months — informing which temporal features to engineer.

4. **Competition Density vs. Items Sold (Scatter + Correlation):**
   Examine whether stores with higher competition density sell fewer items, and whether this relationship changes by promotion type. This informs whether competition_density needs to be included as a moderation variable.

**How findings influence modelling:**
- If interactions exist → include interaction features or use tree-based models that capture them automatically
- If strong seasonality → add month and festival features; consider time-aware cross-validation
- If competition effects are strong → include competition_density as a key feature and explore store-level segmentation

---

### (c) Handling Class Imbalance (80% No-Promotion Transactions)

**Problem:** If 80% of transactions have no promotion, a model trained naively will learn to predict "no promotion" most of the time, ignoring the signal in the 20% of promotional transactions.

**Steps to address imbalance:**

1. **Resampling:** Use SMOTE (Synthetic Minority Oversampling) on the training set to oversample promotional transactions, or undersample no-promotion transactions. Apply only on the training set — never on validation/test sets.

2. **Class Weights:** Use `class_weight='balanced'` in sklearn models to penalise misclassification of the minority class more heavily.

3. **Evaluation Metrics:** Avoid accuracy as the primary metric — it will be misleadingly high. Use **macro F1-score**, **precision-recall curves**, and **confusion matrices** to evaluate model performance across all promotion types fairly.

4. **Stratified Splits:** Ensure train-test splits are stratified by promotion type so each split preserves the class distribution.

---

## B3. Model Evaluation and Deployment

### (a) Train-Test Split Setup and Evaluation Metrics

**Setup:**
The data spans 3 years × 50 stores. Use a **temporal split** rather than a random split:
- **Training set:** First 2 years (24 months) of data across all stores
- **Test set:** Final 1 year (12 months) — the most recent data

**Why random split is inappropriate:**
Random splitting would leak future information into training — for example, training on Month 30 while testing on Month 20. In practice, models are always trained on past data and deployed to predict future months. A temporal split accurately simulates this real-world condition and avoids data leakage.

**Evaluation Metrics:**

| Metric | Relevance to This Business Problem |
|--------|-----------------------------------|
| **RMSE (if regression)** | Measures average prediction error in units of items sold; penalises large misses more heavily |
| **MAE (if regression)** | Average absolute error; more interpretable — "on average, off by X items" |
| **Macro F1-score (if classification)** | Ensures model performs well across all 5 promotion types, not just the most common one |
| **Precision per promotion type** | Avoid recommending promotions that rarely work when recommended |
| **Recall per promotion type** | Ensure the model doesn't systematically miss the best promotion for certain store types |

**Business interpretation:** A MAE of 20 items means the model's recommendation, on average, results in 20 fewer items sold than the optimal choice — translating directly to revenue impact per store per month.

---

### (b) Investigating Different Recommendations for the Same Store in Different Months

**Observation:** The model recommends Loyalty Points Bonus for Store 12 in December but Flat Discount in March month.

**Using Feature Importance to Investigate:**

1. Extract SHAP (SHapley Additive exPlanations) values for Store 12's December and March predictions separately.
2. Compare which features drove each recommendation:
   - In December: High `is_festival` value (Christmas), high `footfall`, and `competition_density` likely pushed toward Loyalty Points Bonus — encouraging repeat purchase during high-traffic months.
   - In March: Lower footfall post-season, lower competition density, and absence of festival flags likely made Flat Discount the model's top choice — discounting to stimulate demand in a slow month.

**Communication to the Marketing Team:**

Present a side-by-side SHAP waterfall chart for December vs. March showing:
- "In December, the high footfall and festival flag are the key drivers pushing toward Loyalty Points Bonus — customers are already in-store; reward them to build long-term loyalty."
- "In March, footfall drops significantly and there are no festivals — a Flat Discount is the best tool to attract price-sensitive customers who might otherwise not visit."

This framing translates technical feature importance into actionable business reasoning.

---

### (c) End-to-End Deployment Process

**1. Saving the Model:**
- Serialise the trained pipeline (including preprocessor + model) using `joblib.dump(pipeline, 'promotion_model.pkl')`
- Store the model artifact in a versioned model registry (e.g., MLflow, AWS S3 with versioning)
- Record the training data cutoff date and model version metadata

**2. Monthly Scoring Pipeline:**
At the start of every month, run an automated batch scoring job:
- Pull store attribute data (static) and the latest footfall, competition density, and calendar data (dynamic) for the upcoming month
- Apply the same preprocessing transformations used during training (loaded from the saved pipeline)
- Run `pipeline.predict()` to generate one recommendation per store
- Output: A table of 50 rows — one per store — with the recommended promotion type and predicted items_sold

**3. Monitoring for Model Degradation:**

| Monitoring Signal | Detection Method | Threshold |
|------------------|-----------------|-----------|
| **Data drift** | Compare monthly feature distributions vs. training baseline using PSI (Population Stability Index) | Alert if PSI > 0.2 for key features |
| **Prediction drift** | Track distribution of recommended promotion types over time | Alert if one promotion type is recommended >80% of the time (may signal drift) |
| **Performance degradation** | After each month, compare actual vs. predicted items_sold using MAE | Alert if rolling 3-month MAE exceeds 1.5× training MAE |
| **Business metric drift** | Track actual promotion uplift vs. expected uplift per store | Monthly business review |

**Retraining Trigger:** Retrain the model when any monitoring alert fires, or at minimum quarterly with the latest 2 years of data as a rolling window. Use an automated CI/CD pipeline to validate the new model against a held-out recent month before promoting it to production.
