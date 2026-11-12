# Multi-Agent Data Analyzer: Automated Business Intelligence System

Upload a raw Excel or CSV file, click one button, and get a cleaned dataset, a business report with strategic recommendations, 
an interactive dashboard, and a chatbot to ask questions about your data. No coding and no prompt writing required: the whole app is button-driven.

**Live app:** [multi-agent-data-analyzer.streamlit.app](https://multi-agent-data-analyzer.streamlit.app) (works on desktop and mobile)

> ### Developed during my data science internship period.

<img width="1898" height="892" alt="Data_Preview" src="https://github.com/user-attachments/assets/fb86b887-bd47-432a-b245-030b14c61d7f" />

<img width="1797" height="817" alt="Business_Report" src="https://github.com/user-attachments/assets/f74a3f02-8378-4995-b050-213e3775acb7" />

<img width="1920" height="818" alt="Smart_Dashboard" src="https://github.com/user-attachments/assets/249e2920-a315-47dc-9ebc-94d191186346" />

## Why this project

Language models write fluent analysis but should not be trusted with arithmetic. This project separates **computation** from **narration**:

- A **deterministic Python layer** (Pandas, NumPy, scikit-learn) does the cleaning, KPI calculation, trend analysis, correlation, anomaly detection, clustering and forecasting.
- A **CrewAI multi-agent team** then explains those verified numbers in plain business language, so the language model never does the math itself.

## Features

- **Automated data cleaning:** column and date standardisation, null normalisation, numeric/currency fixes, model-based missing-value imputation, and outlier capping
- **Reliability checks:** duplicate-safety gates, a row-retention check, sample-size floors and volume-anomaly detection
- **Domain detection:** adapts KPI framing and terminology to the dataset
- **Business report:** summary and strategic recommendations, downloadable as a landscape A4 PDF
- **Dashboard:** charts selected by the agents and validated by rules before display (Plotly)
- **RAG chatbot:** ask questions about the data and past reports, using ChromaDB and Sentence-Transformers
- **Cleaned CSV export:** for forecasting or further modelling
- **Secure login:** hashed passwords (bcrypt)

## How it works

1. Upload an Excel or CSV file and preview the raw data.
2. Click **Clean and Analyze**.
3. The cleaning and statistics engine produces a verified statistics package.
4. CrewAI agents turn it into a business report and pick the dashboard charts.
5. View the report and dashboard, download the PDF and cleaned CSV, or chat with the RAG chatbot.

## Sample run

A deliberately messy dataset of 1,545 rows and 17 columns (duplicate IDs, mixed date formats, text placeholders, impossible values) was processed end to end:

| Measure | Before | After |
|---|---|---|
| Rows | 1,545 | 1,500 (97.1% retained) |
| Duplicate identifiers | 45 | 0 |
| Category variants (cuisine) | 20 | 10 |
| Data-quality score | n/a | 99.4 / 100 |

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Web app | Streamlit |
| Data and ML | Pandas, NumPy, Polars, scikit-learn |
| Agents | CrewAI, LiteLLM |
| LLM | OpenAI GPT-4o-mini |
| Retrieval (RAG) | ChromaDB, Sentence-Transformers (all-MiniLM-L6-v2) |
| Visualisation | Plotly |
| Export | Markdown, xhtml2pdf |
| Security and config | bcrypt, PyYAML, python-dotenv |

## Project structure

```
Multi-Agent-Data-Analyzer/
├── app.py              # Streamlit interface and navigation
├── data_cleaner.py     # Cleaning pipeline and duplicate-safety gates
├── stats_engine.py     # KPIs, trends, anomalies, clustering, forecasting
├── crew.py             # CrewAI agents and task orchestration
├── requirements.txt
└── README.md
```

## Limitations

- Forecasting is limited to a simple linear trend.
- Authentication is basic (no password recovery or role separation).

## Author

**Krittika S**
[GitHub](https://github.com/Krittika028) | [LinkedIn](https://www.linkedin.com/in/s-krittika-2101bb292)
