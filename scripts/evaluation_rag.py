import os
import sys
import argparse
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env if present
load_dotenv()

parser = argparse.ArgumentParser(description="Evaluate one example using Gemini (Google Generative API)")
parser.add_argument("--google-api-key", dest="google_api_key", help="Gemini/Google API key (can also be set via GOOGLE_API_KEY env var)")
parser.add_argument("--model", dest="model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"), help="Gemini model name (default: gemini-2.5-pro)")
args = parser.parse_args()

# Prefer CLI arg, then env var
api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY")
if not api_key:
	print("ERROR: No Gemini API key found. Set the GOOGLE_API_KEY environment variable or pass --google-api-key.")
	print("Example (zsh):\n  export GOOGLE_API_KEY=\"sk-...\"\n  python scripts/evaluation_rag.py")
	sys.exit(2)

# Configure genai
genai.configure(api_key=api_key)

model = genai.GenerativeModel(args.model)

# --- example data ---
question = (
	"TABLE 4.1 Laboratory Results on Admission | Parameter | Patient | Reference Range |"
	" | Haematocrit (%) | 17.6 | >30 |"
	" | Platelet count (10^9/L) | 28 | 150-450 |"
	" | Malaria RDT | Positive | Negative |"
	" Given the patient's lab results, determine the most likely diagnosis and clinical explanation."
)

context = (
	"A 4-year-old girl from Uganda presenting in a coma with low haematocrit (17.6%), "
	"thrombocytopenia (28×10⁹/L), and a positive malaria RDT. The clinical context from the record "
	"indicates cerebral malaria (retinopathy-positive severe malaria)."
)

generated_answer = (
	"Answer:\n"
	" **1. Most likely diagnosis:** Malaria\n\n"
	"**2. Key differential diagnoses:**\n"
	"- Viral haemorrhagic fever\n"
	"- Dengue fever\n"
	"- Zika fever\n\n"
	"**3. Recommended management approach:**\n"
	"- Rapid initiation of artemisinin-based therapy.\n"
	"- Intensive supportive care including fluid resuscitation, electrolyte balance, and pain management.\n"
	"- Blood transfusion in severe cases.\n"
	"- Monitoring for complications such as severe anaemia, organ failure, and death.\n"
)

ground_truth = (
	"1. Most likely diagnosis:\n"
	"Cerebral malaria (retinopathy-positive severe Plasmodium falciparum infection) — the child presents with coma, severe anaemia (Haematocrit 17.6%), thrombocytopenia (Platelet 28 × 10⁹/L), and a positive malaria RDT. Ophthalmoscopy confirms malarial retinopathy, consistent with retinopathy-positive cerebral malaria.\n\n"
	"⸻\n\n"
	"2. Key differential diagnoses:\n"
	"\t•\tAcute bacterial meningitis — may present with coma and fever, but the absence of neck stiffness and a clear, acellular CSF make this unlikely.\n"
	"\t•\tViral encephalitis — considered, but less consistent with severe anaemia and thrombocytopenia.\n"
	"\t•\tIntoxication (e.g., organophosphate poisoning) — considered but ruled out by lack of exposure history.\n"
	"\t•\tMetabolic coma — excluded by normal glucose and absence of hepatic or renal failure.\n\n"
	"⸻\n\n"
	"3. Recommended management approach:\n"
	"\t•\tAntimalarial therapy: IV artesunate 2.4 mg/kg at 0, 12, and 24 hours, then daily.\n"
	"\t•\tSupportive care: Frequent monitoring of vital signs and glucose, correction of hypoglycaemia and electrolyte imbalances.\n"
	"\t•\tOphthalmoscopic monitoring: To assess malarial retinopathy and disease severity.\n"
	"\t•\tManagement of complications: Treat seizures, anaemia (consider transfusion if Hb < 7 g/dL), and maintain airway and hydration.\n"
	"\t•\tFollow-up: Post-discharge neurological assessment due to risk of cognitive sequelae.\n\n"
	"⸻\n\n"
	"Clinical explanation:\n"
	"Cerebral malaria is defined as an otherwise unexplained coma in a patient with Plasmodium falciparum parasitaemia. This child’s positive malaria RDT, profound anaemia, thrombocytopenia, and characteristic malarial retinopathy confirm the diagnosis. The disease represents a severe form of malaria with sequestration of parasitized erythrocytes in cerebral vessels, leading to coma and potential death if untreated.\n"
)

evaluation_prompt = f"""
You are a medical reasoning evaluator.

Given the following:
---
Question: {question}

Context: {context}

System Answer: {generated_answer}

Ground Truth: {ground_truth}
---

Evaluate how accurate, complete, and clinically appropriate the system answer is compared to the ground truth and context.
Give scores between 0–1 for:
1. Factual accuracy (alignment with ground truth)
2. Relevance (does the answer address the question?)
3. Completeness (does it capture all major points?)
4. Clinical correctness (is the reasoning sound?)
Then provide a short textual feedback summary.
"""

try:
	response = model.generate_content(evaluation_prompt)
	# The genai response object generally exposes .text or structured output; print a human-readable text
	if hasattr(response, "text") and response.text:
		print(response.text)
	else:
		# Fallback: print the full object
		print(response)
except Exception as e:
	print("Gemini API call failed:", e)
	# Provide actionable hints
	print("Ensure the GOOGLE_API_KEY is correct and has access to the Gemini model. For Application Default Credentials, follow: https://ai.google.dev/gemini-api/docs/oauth")
	sys.exit(1)