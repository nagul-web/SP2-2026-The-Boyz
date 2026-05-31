# Sona Power Predict - 2026

**College Name:** Sona College of Technology  
**Team Name:** The Boyz  

###  Team Members
* **Aswin M** - Year 3, Computer Science and Design (CSD)
* **Nagulan V** - Year 3, Computer Science and Design (CSD)
* **Navaneethan A R** - Year 3, Computer Science and Design (CSD)
* **Sabari P** - Year 3, Computer Science and Design (CSD)

---

###  Libraries Used in Model

The following Python libraries are utilized for data manipulation and mathematical operations:

* `pandas` : Loading CSV files, DataFrame manipulation, grouping, and building the innings-level aggregated dataset.
* `numpy` : Array operations, numerical aggregations, and calculating the custom Exponential Time-Decay weights for the machine learning engine.
* `scikit-learn` : Specifically utilizing `GradientBoostingRegressor` (an advanced tree ensemble method) for the core prediction engine.
* `re` / `os` : Used for text normalization, regular expressions, file path management, and driving the dynamic NLP text-cleaning pipeline.

---

###  Model Architecture & Methodology

* **GradientBoostingRegressor:** Core prediction engine utilizes an assembly line of 150 decision trees (`n_estimators=150`, `max_depth=5`). Tree models are exceptionally powerful at learning complex, non-linear relationships in tabular data (like pitch conditions combined with player strike rates).
  
* **Exponential Time-Decay Weighting:** Standard tree models often fail in modern IPL predictions because they average all historical data, resulting in predictions that are far too low for today's explosive game. We solved this by injecting a mathematical time-decay factor (`decay_factor = 4.0`) directly into the training weights. This forces the algorithm to give ~100% learning priority to recent matches and nearly 0% to older eras, naturally shifting the model's baseline to modern 65+ run scores without using hardcoded cheats.
  
* **Autonomous NLP Cleaning (IoU):** To ensure a crash-proof pipeline during live data ingestion, we built a custom Intersection-over-Union (IoU) text processing algorithm. It instantly detects and groups misspelled stadium or team names dynamically on the fly, preventing data fragmentation.
  
* **Bayesian Target Encoding:** Instead of basic one-hot encoding which causes dataset sparsity, categorical features (like venues) are converted into smoothed numerical run averages, protected by a prior weight against one-match outliers.

---

###  License

This project is licensed under the **MIT License**.
