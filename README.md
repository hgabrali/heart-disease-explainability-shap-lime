# Why Did My Model Predict That? — Heart Disease Explainability with SHAP & LIME

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Databricks](https://img.shields.io/badge/Platform-Databricks-red.svg)](https://databricks.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MasterSchool](https://img.shields.io/badge/MasterSchool-AI%20Enhanced%20Productivity%20Project%203-green.svg)](https://masterschool.com)

> **Project 3 of 3 — AI Enhanced Productivity Sequence**
> > Focus: Trust, Transparency & Interpretability in Healthcare ML
> >
> > ---
> >
> > ## Project Overview
> >
> > This project builds an end-to-end machine learning pipeline for **heart disease prediction** using the UCI Heart Disease Dataset. The core focus is not just prediction accuracy — it is on **understanding and explaining model decisions** using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).
> >
> > In high-stakes domains like healthcare, knowing *why* a model made a prediction is as important as the prediction itself. This project demonstrates how explainability tools bridge the gap between model outputs and clinical trust.
> >
> > ---
> >
> > ## Dataset
> >
> > | Property | Details |
> > |---|---|
> > | **Source** | UCI Heart Disease Dataset (via Kaggle) |
> > | **Records** | 303 patient records |
> > | **Target** | `num` column: 0 = no disease, 1–4 = presence of heart disease |
> > | **Features** | Age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, max HR, exercise-induced angina, ST depression, slope, vessels colored, thalassemia |
> >
> > ---
> >
> > ## Project Structure
> >
> > ```
> > heart-disease-explainability-shap-lime/
> > │
> > ├── data/
> > │   └── heart_disease_uci.csv          # UCI Heart Disease Dataset
> > │
> > ├── notebooks/
> > │   ├── 01_data_exploration_preprocessing.ipynb   # EDA & data cleaning
> > │   ├── 02_model_training_evaluation.ipynb        # Model training & metrics
> > │   ├── 03_shap_analysis.ipynb                    # SHAP explainability
> > │   ├── 04_lime_analysis.ipynb                    # LIME explainability
> > │   └── 05_business_interpretation.ipynb          # Business insights & reflection
> > │
> > ├── .gitignore
> > ├── LICENSE
> > └── README.md
> > ```
> >
> > ---
> >
> > ## Notebooks
> >
> > ### 01 — Data Exploration & Preprocessing
> > - Load and inspect the dataset
> > - - Handle missing values and outliers
> >   - - Encode categorical features (chest pain type, thalassemia, slope, etc.)
> >     - - Train/test split (80/20 stratified)
> >       - - Feature distribution visualizations
> >        
> >         - ### 02 — Model Training & Evaluation
> >         - - Train Random Forest and XGBoost classifiers
> >           - - Evaluate with: Accuracy, Precision, Recall, F1-Score, AUC-ROC
> >             - - Confusion matrix and ROC curve visualizations
> >               - - Select best model for explainability phase
> >                
> >                 - ### 03 — SHAP Analysis
> >                 - - Apply `shap.TreeExplainer` to the trained model
> >                   - - **Global explanations**: Summary plot (beeswarm), bar chart of mean |SHAP|
> >                     - - **Local explanations**: Waterfall plot for individual patient predictions
> >                       - - Markdown analysis: which clinical features are most impactful and why
> >                        
> >                         - ### 04 — LIME Analysis
> >                         - - Apply `LimeTabularExplainer` to the trained model
> >                           - - Generate local explanations for at least **two different test cases**
> >                             - - Compare LIME feature weights vs SHAP values for the same instances
> >                               - - Discuss agreement and disagreement between the two methods
> >                                
> >                                 - ### 05 — Business Interpretation
> >                                 - - Identify top clinical predictors of heart disease
> >                                   - - Reflect on how explainability affects trust in clinical AI systems
> >                                     - - Discuss SHAP vs LIME trade-offs: scope, stability, and interpretability
> >                                       - - Ethical considerations: model transparency in healthcare settings
> >                                        
> >                                         - ---
> >
> > ## Key Technical Components
> >
> > | Component | Tool/Library |
> > |---|---|
> > | Platform | Databricks (Apache Spark) |
> > | ML Models | Random Forest, XGBoost |
> > | Global Explainability | SHAP TreeExplainer |
> > | Local Explainability | LIME (LimeTabularExplainer) |
> > | Evaluation Metrics | Accuracy, Precision, Recall, AUC-ROC |
> > | Visualization | Matplotlib, Seaborn, SHAP plots |
> > | Language | Python 3.10+ |
> >
> > ---
> >
> > ## Installation & Requirements
> >
> > ```bash
> > pip install shap lime xgboost scikit-learn pandas numpy matplotlib seaborn
> > ```
> >
> > Or install from requirements file:
> >
> > ```bash
> > pip install -r requirements.txt
> > ```
> >
> > ---
> >
> > ## Results Summary
> >
> > The model achieves strong predictive performance on the UCI Heart Disease Dataset. SHAP analysis reveals that **thalassemia type**, **number of major vessels colored by fluoroscopy (ca)**, and **chest pain type** are the top three global predictors. LIME provides local explanations that align with clinical intuition for most individual cases.
> >
> > ---
> >
> > ## Reflections on Explainability
> >
> > Both SHAP and LIME serve complementary roles:
> > - **SHAP** provides theoretically grounded global and local feature attributions, stable across runs
> > - - **LIME** offers intuitive local approximations but can vary between runs due to sampling
> >  
> >   - In healthcare, SHAP is preferable for auditing and regulatory review, while LIME can support real-time clinician-facing explanations. Neither tool replaces domain expertise — they enhance accountability.
> >  
> >   - ---
> >
> > ## Part of the AI Enhanced Productivity Series
> >
> > | Project | Topic | Tools |
> > |---|---|---|
> > | Project 1 | Automated ML Workflow | PySpark, PyCaret, MLflow |
> > | Project 2 | AI-Assisted Sentiment Analysis | Gemini, AutoViz, LLMs |
> > | **Project 3** | **Model Explainability in Healthcare** | **SHAP, LIME, XGBoost** |
> >
> > ---
> >
> > ## Author
> >
> > **Hande Gabrali-Knobloch**
> > Data Science Analyst | MasterSchool AI Program
> > [LinkedIn](https://www.linkedin.com/in/hande-gabrali-knobloch-5b176615) · [GitHub](https://github.com/hgabrali)
> >
> > ---
> >
> > ## License
> >
> > This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
