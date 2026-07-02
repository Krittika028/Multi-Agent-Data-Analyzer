"""
domain_detector.py

Intelligently detects the business domain of an uploaded dataset
(Healthcare, Retail, Finance, HR, Insurance, Manufacturing, Education,
Logistics, Telecom, etc.) using an LLM agent that reasons over the
cleaned dataset's column profile, sample rows, and cleaning report —
NOT a hardcoded keyword lookup.

This is intentionally separate from DataCleaner: DataCleaner is
deterministic (pandas/regex, no LLM). DomainDetector is the genuinely
"agentic" piece — it makes a judgment call under uncertainty and
explains its reasoning, the way a human analyst opening a new dataset
for the first time would.
"""

import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()


class DomainDetector:

    KNOWN_DOMAINS = [
        "Healthcare", "Retail", "Banking & Finance", "Insurance",
        "Human Resources", "Manufacturing", "Education", "Logistics",
        "Telecom", "E-commerce", "Real Estate", "Agriculture",
        "Energy & Utilities", "Government / Public Sector", "Other",
    ]

    def __init__(self, llm=None):
        self.llm = llm or LLM(
            model=os.getenv("MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.agent = Agent(
            role="Domain Detection Specialist",
            goal=(
                "Accurately identify the real-world business domain a "
                "dataset comes from, using column names, data patterns, "
                "value ranges, and cleaning signals as evidence — never "
                "guessing from a single column name alone."
            ),
            backstory=(
                "You are a senior data consultant who has worked across "
                "healthcare, retail, banking, insurance, HR, manufacturing, "
                "education, logistics, and telecom datasets. You are skilled "
                "at quickly recognizing what kind of business data you're "
                "looking at from subtle clues: column naming conventions, "
                "the shape of categorical values, typical numeric ranges "
                "(e.g. blood pressure vs. transaction amounts vs. salaries), "
                "and which entities the rows represent (patients, customers, "
                "employees, shipments, policies). You always justify your "
                "conclusion with specific evidence from the data, and you "
                "are honest about uncertainty when the data is ambiguous."
            ),
            llm=self.llm,
            verbose=False,
        )

    # =====================================
    # BUILD EVIDENCE FOR THE AGENT
    # =====================================
    def _build_evidence(self, df, cleaning_report=None, sample_rows=8):
        column_profile = cleaning_report.get("column_profile", {}) if cleaning_report else {}
        classification = cleaning_report.get("column_classification", {}) if cleaning_report else {}

        evidence = {
            "columns": list(df.columns),
            "row_count": len(df),
            "column_count": len(df.columns),
            "column_profile": column_profile,
            "column_classification": classification,
            "sample_data": df.head(sample_rows).to_dict(orient="records"),
            "numeric_summary": {
                col: {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": round(float(df[col].mean()), 2),
                }
                for col in df.select_dtypes(include="number").columns
            },
        }
        return evidence

    # =====================================
    # BUILD THE DETECTION TASK
    # =====================================
    def _build_task(self, evidence):
        domains_list = ", ".join(self.KNOWN_DOMAINS)

        return Task(
            description=f"""
You are analyzing an uploaded dataset to determine its business domain.

DATASET EVIDENCE:
Columns: {evidence['columns']}
Row count: {evidence['row_count']}
Column count: {evidence['column_count']}

Column Profile (role classification from cleaning step):
{json.dumps(evidence['column_profile'], indent=2)}

Column Classification:
{json.dumps(evidence['column_classification'], indent=2)}

Numeric Column Ranges:
{json.dumps(evidence['numeric_summary'], indent=2)}

Sample Rows:
{json.dumps(evidence['sample_data'], indent=2, default=str)}

KNOWN DOMAIN OPTIONS (pick the closest match, or "Other" if none fit):
{domains_list}

YOUR TASK:
1. Identify the most likely business domain from the list above.
2. Identify the primary entity each row represents (e.g. "Patient", "Customer", "Transaction", "Employee", "Shipment").
3. Give a confidence level: High, Medium, or Low.
4. List 3-5 specific pieces of evidence from the data that support your conclusion (cite actual column names / value patterns / ranges).
5. If confidence is Medium or Low, briefly note what's ambiguous and what a second-best guess would be.

Respond ONLY with valid JSON in this exact structure, no markdown, no preamble:
{{
  "domain": "...",
  "primary_entity": "...",
  "confidence": "High" | "Medium" | "Low",
  "evidence": ["...", "...", "..."],
  "alternative_domain": "..." or null,
  "reasoning_summary": "1-2 sentence plain-English explanation"
}}
""",
            expected_output="A single valid JSON object matching the specified schema, with no extra text.",
            agent=self.agent,
        )

    # =====================================
    # PUBLIC ENTRY POINT
    # =====================================
    def detect(self, df, cleaning_report=None):
        """
        Run domain detection on a (cleaned) dataframe.

        Returns a dict:
            {
              "domain": str,
              "primary_entity": str,
              "confidence": "High" | "Medium" | "Low",
              "evidence": [str, ...],
              "alternative_domain": str | None,
              "reasoning_summary": str,
            }
        Falls back to a safe default structure if the LLM response
        can't be parsed, so the UI never crashes on a bad response.
        """
        evidence = self._build_evidence(df, cleaning_report)
        task = self._build_task(evidence)

        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        raw_result = crew.kickoff()

        return self._parse_result(str(raw_result))

    def _parse_result(self, raw_text):
        text = raw_text.strip()

        # Strip accidental markdown code fences if the model adds them
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            parsed = json.loads(text)
            return {
                "domain": parsed.get("domain", "Other"),
                "primary_entity": parsed.get("primary_entity", "Unknown"),
                "confidence": parsed.get("confidence", "Low"),
                "evidence": parsed.get("evidence", []),
                "alternative_domain": parsed.get("alternative_domain"),
                "reasoning_summary": parsed.get("reasoning_summary", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            return {
                "domain": "Other",
                "primary_entity": "Unknown",
                "confidence": "Low",
                "evidence": [],
                "alternative_domain": None,
                "reasoning_summary": (
                    "Could not parse a structured response from the model. "
                    "Raw output: " + text[:300]
                ),
            }