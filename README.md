# OOS_Prediction_ML_Model
# FMCG Stockout Prediction — Microsoft Fabric ML Pipeline

An end-to-end machine learning solution built on **Microsoft Fabric** that predicts stockout risk for FMCG (Fast-Moving Consumer Goods) SKUs across retail stores. Raw data is ingested into a Lakehouse, cleaned and feature-engineered into Delta tables, and then consumed by a Spark ML pipeline that writes predictions directly to Power BI via DirectLake.

---

## Overview

Stockouts cost retailers revenue and damage customer loyalty. This project predicts **days until stockout** for each SKU-store combination, enabling procurement teams to act before shelves go empty.

The pipeline runs across three stages:

```
Raw Files (Lakehouse)
      ↓
[Notebook 1] Data Cleaning + Feature Engineering  →  ml_features (Delta table)
      ↓
[Notebook 2] ML Training + Prediction             →  stockout_predictions (Delta table)
      ↓
Power BI DirectLake Dashboard
```

---

## Platform & Tech Stack

| Component | Technology |
|---|---|
| Platform | Microsoft Fabric (Lakehouse + Notebooks) |
| Compute | Apache Spark (PySpark) |
| ML Framework | Spark MLlib |
| Experiment Tracking | MLflow |
| Model Registry | Fabric Model Registry (MLflow) |
| Storage Format | Delta Lake |
| Reporting | Power BI (DirectLake mode) |

---

## Repository Structure

```
├── Notebook_1.ipynb          # Step 2 — Data Cleaning & Feature Engineering
├── model_notebook.ipynb      # Step 3 — ML Training & Prediction
└── README.md
```

> **Step 1 (data ingestion)** is done manually by uploading raw CSV/Excel files into the Microsoft Fabric Lakehouse and converting them to Delta tables via the Lakehouse UI.

---

## Data Sources

Three raw tables must be present in the Lakehouse before running the notebooks:

| Table Name | Description |
|---|---|
| `sales_data` | Daily sales transactions per SKU and store |
| `inventory_data` | Daily inventory snapshots (stock on hand, safety stock, reorder points) |
| `stockout_events` | Historical out-of-stock events with start/end dates |

### Expected Schema

**`sales_data`**
- `date`, `sku_id`, `sku_name`, `category`, `store_id`
- `units_sold`, `unit_price`, `revenue`
- `promo_flag`, `discount_pct`, `is_festival`, `is_weekend`, `shelf_life_days`

**`inventory_data`**
- `date`, `sku_id`, `store_id`
- `stock_on_hand`, `safety_stock`, `reorder_point`
- `replenishment_qty_ordered`, `lead_time_days`
- `below_safety_stock`, `days_of_supply`

**`stockout_events`**
- `sku_id`, `store_id`
- `stockout_start_date`, `stockout_end_date`, `stockout_duration_days`

---

## Pipeline Walkthrough

### Notebook 1 — Data Cleaning & Feature Engineering

Reads from the three raw Lakehouse tables, cleans each one, and engineers a rich feature set for ML.

**Data Cleaning**

- **Sales:** removes duplicate `(date, sku_id)` rows, fills missing flags with 0, filters out negative sales and zero-price records, standardises casing.
- **Inventory:** casts dates, filters impossible negative stock values, forward-fills null `stock_on_hand` per SKU, recalculates `days_of_supply` to handle zero-demand edge cases.
- **Stockout Events:** casts dates, drops corrupt records where `start_date > end_date`, uses `-1` as sentinel for unresolved stockouts.

Clean tables saved:
- `sales_clean`
- `inventory_clean`
- `stockout_events_clean`

**Feature Engineering**

After joining inventory and sales on `(date, sku_id)`, the following feature groups are built:

| Feature Group | Features |
|---|---|
| Rolling Demand | avg/sum/max sales over 7d, 14d, 30d; std dev, promo count, festival count over 7d |
| Inventory Ratios | stock-to-safety ratio, stock-vs-reorder-point, projected cover days |
| Lag Features | units sold and stock on hand lagged by 1, 3, and 7 days; SOH delta over same windows |
| Calendar | day of week, week of year, month, quarter, is_month_end flag |
| Category Encoding | ordinal encoding of 7 FMCG categories; perishable flag for Dairy & Frozen Foods |
| Target Variable | `days_until_stockout` — stock on hand divided by 7-day avg sales, capped at 90 days |

Output: `ml_features` Delta table (43 columns).

---

### Notebook 2 — ML Training & Prediction

Reads `ml_features`, trains two regression models, selects the best, registers it, and writes predictions back to the Lakehouse.

**Train / Test Split**

Time-based split to prevent data leakage:
- Train: Jan 2024 – Oct 2024
- Test: Nov 2024 – Dec 2024

**Models Trained**

| Model | Key Hyperparameters |
|---|---|
| Gradient Boosted Trees (GBT) | maxIter=100, maxDepth=5, stepSize=0.1 |
| Random Forest | numTrees=100, maxDepth=6 |

Both models use a shared Spark ML Pipeline:
```
Imputer (median) → VectorAssembler → StandardScaler → Model
```

**Evaluation Metrics:** RMSE, MAE, R² — logged to MLflow.

**Model Selection & Registry**

The model with the lower RMSE is selected automatically, registered in the Fabric Model Registry as `StockoutPredictor_FMCG`, and promoted to the `Production` stage.

**Predictions Output**

The best model scores the entire dataset. Each row in `stockout_predictions` includes:

| Column | Description |
|---|---|
| `predicted_days_until_stockout` | Model's prediction (floored at 0) |
| `risk_band` | `CRITICAL` (≤7d), `WARNING` (8–14d), `OK` (>14d) |
| `procurement_action` | `REORDER_NOW` if prediction ≤ lead time, else `MONITOR` |

This table is written in Delta format and is immediately available in **Power BI via DirectLake mode** with no import step.

---

## Setup & Execution

### Prerequisites

- Microsoft Fabric workspace with a Lakehouse attached
- Raw data tables (`sales_data`, `inventory_data`, `stockout_events`) loaded into the Lakehouse
- MLflow experiment tracking enabled (built into Fabric)

### Steps

1. **Upload raw data** into the Fabric Lakehouse and convert files to Delta tables using the Lakehouse UI.

2. **Attach the Lakehouse** to both notebooks in the Fabric notebook settings.

3. **Run Notebook 1** (`Notebook_1.ipynb`)  
   - Cleans raw tables  
   - Writes `sales_clean`, `inventory_clean`, `stockout_events_clean`  
   - Builds and writes `ml_features`

4. **Run Notebook 2** (`model_notebook.ipynb`)  
   - Trains GBT and Random Forest models  
   - Logs experiments to MLflow  
   - Registers best model to Fabric Model Registry  
   - Writes `stockout_predictions`

5. **Connect Power BI** to the Lakehouse in DirectLake mode and build dashboards on `stockout_predictions`.

---

## MLflow Experiment Tracking

All runs are logged under the experiment name `stockout_prediction_fmcg`. Each run records:

- Model type and hyperparameters
- Training cutoff date
- Number of features used
- RMSE, MAE, R² on the Nov–Dec 2024 test set

View experiments in the **Fabric ML Experiments** UI.

---

## Output Table Reference

### `stockout_predictions`

| Column | Type | Description |
|---|---|---|
| `date` | Date | Record date |
| `sku_id` | String | SKU identifier |
| `sku_name` | String | SKU display name |
| `category` | String | Product category |
| `store_id` | String | Store identifier |
| `stock_on_hand` | Integer | Current stock level |
| `safety_stock` | Integer | Minimum stock threshold |
| `reorder_point` | Integer | Stock level triggering reorder |
| `lead_time_days` | Integer | Supplier lead time |
| `actual_days_until_stockout` | Integer | Ground truth |
| `predicted_days_until_stockout` | Double | Model prediction |
| `risk_band` | String | CRITICAL / WARNING / OK |
| `procurement_action` | String | REORDER_NOW / MONITOR |

---

## Product Categories

The model handles 7 FMCG categories with perishability flags:

| Code | Category | Perishable |
|---|---|---|
| 1 | Beverages | No |
| 2 | Dairy | Yes |
| 3 | Snacks | No |
| 4 | Personal Care | No |
| 5 | Household | No |
| 6 | Staples | No |
| 7 | Frozen Foods | Yes |

---

## Customisation

- **Table names:** Update `TABLE_SALES`, `TABLE_INV`, `TABLE_OOS` in Notebook 1 and `TABLE_FEATURES`, `TABLE_PREDICTIONS` in Notebook 2 to match your Lakehouse table names.
- **Train/test cutoff:** Adjust the date filter in Notebook 2 to reflect your data range.
- **Risk thresholds:** Modify the `risk_band` cutoffs (7d / 14d) in Notebook 2 to suit your business rules.
- **Category encoding:** Extend the `category_code` mapping in Notebook 1 to include additional categories.
- **Hyperparameters:** Tune `maxIter`, `maxDepth`, `numTrees`, and `stepSize` in Notebook 2 for better accuracy on your data.

---

## License

This project is provided for internal use. Adapt as needed for your organisation.

