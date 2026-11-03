# Lab 4: Data Validation Report

## 1. Ingestion Validation
* **Expected Dimensions:** 48,842 rows, 15 columns.
* **Actual Dimensions:** 48,842 rows, 15 columns.
* **Status:** Passed.

## 2. Schema and Range Checks
* **Age (17-90):** Passed.
* **Hours-per-week (1-99):** Passed.
* **Capital-gain (0-99999):** Passed.
* **Capital-loss (0-4356):** Passed.
* **Null Policy:** Verified. Missing values are represented by `?` and were successfully detected without silently dropping rows.

## 3. Label-Quality Audit
* **Duplicate/Conflict Check:** The audit detected 5 duplicate feature rows possessing conflicting income labels (`<=50K` vs `>50K`).
* **Manual Inspection:** 50 random rows were audited against the dataset dictionary. No mathematically impossible values or schema violations were detected.
