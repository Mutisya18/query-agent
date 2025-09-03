from flask import Flask, request, render_template
import pandas as pd
import json
import os
import openai


app = Flask(__name__)

# Load your data
reasons_df = pd.read_csv("data/reasons.csv")
with open("data/issues.json", "r", encoding="utf-8") as f:
    issues_data = json.load(f)

# Configure OpenAI (make sure you set your API key as env variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Helper: find matching data ---
def search_knowledge_base(query: str):
    results = []

    # Search in reasons.csv
    for _, row in reasons_df.iterrows():
        if query.lower() in str(row["reason"]).lower():
            results.append({
                "type": "reason",
                "customer_name": row.get("customer_name", "the customer"),
                "reason": row["reason"],
                "resolution": row["resolution"]
            })

    # Search in issues.json
    for category, cases in issues_data.items():
        for case in cases:
            if query.lower() in case["issue"].lower():
                results.append({
                    "type": "issue",
                    "issue": case["issue"],
                    "resolution": case["resolution"]
                })

    return results


# --- Helper: AI structured response ---
def generate_ai_response(query: str, results: list):
    if not results:
        # fallback case
        prompt = f"""
Format the response exactly like this structure:

Dear Mutisya,

The resolution for "{query}" needs further review. Kindly reach out to Mutisya.stanley@ncbagroup.com.

Regards,
Retail Digital Solutions
"""
        return prompt

    else:
        # prepare context for AI
        context_texts = []
        for r in results:
            if r["type"] == "reason":
                context_texts.append(
                    f'Customer "{r["customer_name"]}" does not have a loan limit due to "{r["reason"]}". '
                    f'Resolution: {r["resolution"]}'
                )
            elif r["type"] == "issue":
                context_texts.append(
                    f'Issue: "{r["issue"]}". Resolution: {r["resolution"]}'
                )

        context = "\n".join(context_texts)

        prompt = f"""
You are a formal assistant. Rewrite the following information into a polished response.
The response must ALWAYS follow one of these formats:

1. For customer-specific reasons:
Dear Mutisya,

The customer "customer name" does not have a loan limit due to "reason".

"resolution in formal tone (e.g., Kindly advise the customer..., The customer needs...)"

Regards,
Retail Digital Solutions

2. For general issues (no customer details):
Dear Mutisya,

To "issue", kindly "resolution".

Regards,
Retail Digital Solutions

If the data is not sufficient, use the fallback message provided.

Here is the raw info you must reframe:
{context}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # you can swap to gpt-4o if you prefer
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response["choices"][0]["message"]["content"].strip()


@app.route("/", methods=["GET", "POST"])
def index():
    bot_response = None
    if request.method == "POST":
        query = request.form["query"]
        results = search_knowledge_base(query)
        bot_response = generate_ai_response(query, results)
    return render_template("index.html", response=bot_response)


if __name__ == "__main__":
    app.run(debug=True)
