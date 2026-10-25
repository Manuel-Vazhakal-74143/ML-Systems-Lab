# Problem Definition: UCI Adult Dataset Pipeline

**Objective:** Build a robust, leakage-free pipeline to predict whether an individual earns more than $50,000 annually based on census data.

* **Inputs:** 14 features consisting of mixed numerical data (e.g., age, capital-gain, hours-per-week) and categorical data (e.g., workclass, education, marital-status).
* **Output:** Binary classification label indicating income >50K[cite: 15].
* **Success Metrics:** Model evaluation will rely on both Accuracy and the F1 score[cite: 15].
* **Constraints:** The pipeline must enforce strict leakage prevention (train/test splitting prior to cleaning or imputing) and must execute reproducibly via a single command[cite: 15].