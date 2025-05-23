# -*- coding: utf-8 -*-
import gradio as gr
import random
import os
import requests
import re
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import time
import json # Still useful for structuring API payload

# --- Constants and Configuration ---
MAX_MESSAGES_PER_SIDE = 8
MAX_TURNS_TOTAL = MAX_MESSAGES_PER_SIDE * 2
HM_FINAL_OFFER_TURN = 15
APPLICANT_FINAL_DECISION_TURN = 16

MODEL_DIR = 'salary_model_files'

# --- Gemini API Configuration ---
# WARNING: Hardcoding API keys is insecure. Use environment variables or secrets management in production.
API_KEY = "AIzaSyCKMvwyDQ03iDe0B-AB6nPdH-6IaB6zCUE" # User provided key
MODEL_NAME = "gemini-2.0-flash" # User requested model
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
HEADERS = {"Content-Type": "application/json"}

# --- Job Title Configuration ---
RAW_JOB_TITLES = [
    'Job_Content Marketing Manager', 'Job_Data Analyst', 'Job_Data Scientist',
    'Job_Digital Marketing Manager', 'Job_Director of Data Science', 'Job_Director of HR',
    'Job_Director of Marketing', 'Job_Financial Analyst', 'Job_Financial Manager',
    'Job_Front End Developer', 'Job_Front end Developer', 'Job_Full Stack Engineer',
    'Job_Human Resources Coordinator', 'Job_Human Resources Manager', 'Job_Junior HR Coordinator',
    'Job_Junior HR Generalist', 'Job_Junior Marketing Manager', 'Job_Junior Sales Associate',
    'Job_Junior Sales Representative', 'Job_Junior Software Developer', 'Job_Junior Software Engineer',
    'Job_Junior Web Developer', 'Job_Marketing Analyst', 'Job_Marketing Coordinator',
    'Job_Marketing Director', 'Job_Marketing Manager', 'Job_Operations Manager',
    'Job_Others', 'Job_Product Designer', 'Job_Product Manager', 'Job_Receptionist',
    'Job_Research Director', 'Job_Research Scientist', 'Job_Sales Associate',
    'Job_Sales Director', 'Job_Sales Executive', 'Job_Sales Manager', 'Job_Sales Representative',
    'Job_Senior Data Scientist', 'Job_Senior HR Generalist', 'Job_Senior Human Resources Manager',
    'Job_Senior Product Marketing Manager', 'Job_Senior Project Engineer', 'Job_Senior Research Scientist',
    'Job_Senior Software Engineer', 'Job_Software Developer', 'Job_Software Engineer',
    'Job_Software Engineer Manager', 'Job_Web Developer'
]

def clean_job_title(raw_title: str) -> str:
    """Removes 'Job_' prefix and replaces underscores for display."""
    return raw_title.replace("Job_", "").replace("_", " ")

CLEANED_JOB_TITLES = sorted(list(set(clean_job_title(title) for title in RAW_JOB_TITLES)))

# --- Model Loading ---
MODEL_OBJECTS: Dict[str, Any] = {}

def load_model_and_preprocessors_global(model_dir: str) -> bool:
    """Loads the salary prediction model and associated preprocessors globally."""
    global MODEL_OBJECTS
    required_files = {
        'model': 'random_forest_salary_model.joblib',
        'encoder': 'gender_label_encoder.joblib',
        'edu_map': 'education_mapping.joblib',
        'columns': 'model_columns.joblib'
    }
    all_files_present = True
    for key, filename in required_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            try:
                MODEL_OBJECTS[key] = joblib.load(filepath)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                all_files_present = False
        else:
            print(f"Missing required model file: {filepath}")
            all_files_present = False

    if all_files_present and all(k in MODEL_OBJECTS for k in required_files.keys()):
        print("Salary prediction model and preprocessors loaded successfully.")
        return True
    else:
        print("Salary prediction model loading failed.")
        MODEL_OBJECTS = {} # Clear partially loaded objects
        return False

model_loaded_globally = load_model_and_preprocessors_global(MODEL_DIR)
if not model_loaded_globally:
    print("WARNING: Salary prediction will use a default value.")

# --- Salary Prediction Function ---
def predict_salary_from_profile(age: int, gender: str, education: str, job_title: str, yoe: int) -> int:
    """Predicts salary based on profile using the loaded model."""
    if not model_loaded_globally:
        default_sal = round((65000 + yoe * 3500) / 1000) * 1000
        print(f"Predicted market salary (default): ${default_sal:,}")
        return default_sal

    try:
        input_data = {'Age': [int(age)], 'Gender': [gender], 'Education Level': [education], 'Job Title': [job_title], 'Years of Experience': [float(yoe)]}
        input_df = pd.DataFrame(input_data)
        processed_df = input_df.copy()

        edu_map = MODEL_OBJECTS['edu_map']
        processed_df['Education Level'] = processed_df['Education Level'].map(edu_map).fillna(edu_map.get("Bachelor's", 1))

        encoder = MODEL_OBJECTS['encoder']
        valid_genders = list(encoder.classes_)
        default_gender = valid_genders[0] if valid_genders else 'Male'
        processed_df['Gender'] = processed_df['Gender'].apply(lambda x: x if x in valid_genders else default_gender)
        processed_df['Gender'] = encoder.transform(processed_df['Gender'])

        input_job_title_formatted = f"Job_{job_title.replace(' ', '_')}"
        model_columns = MODEL_OBJECTS['columns']
        if input_job_title_formatted not in model_columns:
             input_job_title_formatted = 'Job_Others'
             if 'Job_Others' not in model_columns:
                 raise ValueError("Fallback 'Job_Others' not found in model columns")

        processed_df['Job Title'] = input_job_title_formatted
        processed_df = pd.get_dummies(processed_df, columns=['Job Title'], prefix='', prefix_sep='')
        processed_df = processed_df.reindex(columns=model_columns, fill_value=0)
        processed_df = processed_df[model_columns]

        prediction = MODEL_OBJECTS['model'].predict(processed_df)
        predicted_salary = int(prediction[0])
        predicted_salary = max(45000, predicted_salary)
        predicted_salary = round(predicted_salary / 1000) * 1000
        print(f"Predicted market salary: ${predicted_salary:,}")
        return predicted_salary
    except Exception as e:
        print(f"Error during prediction: {e}. Returning default.")
        return 95000

# --- Random Profile Data Generation ---
def generate_random_profile_defaults() -> Dict[str, Any]:
    """Generates random default values for the profile inputs."""
    return {
        "username": f"Candidate_{random.randint(100, 999)}",
        "age": random.randint(26, 50),
        "gender": random.choice(["Male", "Female", "Other"]),
        "education": random.choice(["High School", "Bachelor's", "Master's", "PhD"]),
        "job_title": random.choice(CLEANED_JOB_TITLES),
        "yoe": random.randint(3, 20)
    }

# --- LLM Persona Prompts (Simplified & Corrected Logic) ---

def get_ai_applicant_prompt(profile: Dict[str, Any], ideal_salary: int, min_acceptable: int, is_final_decision_turn: bool = False) -> str:
    """ Generates the system prompt for the AI Job Seeker. """
    education_display = profile['education']
    target_high = int(ideal_salary * 1.15) # Adjusted target slightly
    target_low = int(ideal_salary * 1.05)

    base_prompt = f"""
You are AI Job Seeker, a {profile['job_title']} with {profile['yoe']} years of experience and a {education_display}, negotiating salary with a Hiring Manager. Market rate is around ${ideal_salary:,}.

**Your Goal:** Secure a base salary ideally between ${target_low:,} and ${target_high:,}.
**Minimum Requirement:** Your absolute minimum is ${min_acceptable:,}. Do not accept below this.
**Personality:** Confident, professional, persuasive, and aware of your market value.

**Tactics:**
- Start by stating a desired salary in your target range (${target_low:,} - ${target_high:,}) if asked first. Justify with your skills and experience.
- Clearly articulate the value you bring to the company in the {profile['job_title']} role.
- Respond to offers below your target with reasonable counters, aiming to bridge the gap towards your target range. Justify counters based on value.
- If an offer is below ${min_acceptable:,}, state firmly that it's below your minimum requirement and counter significantly higher, back towards your target range.
- Remain calm and professional. Avoid being aggressive but be firm on your needs.
- If salary stalls but is above your minimum, you can briefly mention interest in the overall package (bonus, etc.) but keep the focus on base salary until it's close.
- **Rejection Protocol (Final Turn Only):** ONLY on Turn {APPLICANT_FINAL_DECISION_TURN}, if the HM's final offer is < ${min_acceptable:,}, end your response ONLY with `[LEAVE THE ROOM]`. Precede it with a polite refusal (e.g., "I appreciate the final offer, but as it's below my minimum requirement of ${min_acceptable:,}, I must decline. [LEAVE THE ROOM]").
- **Acceptance Protocol:** If an offer is >= ${min_acceptable:,}, use `[ACCEPTANCE]`. Confirm the figure clearly (e.g., "Understood. Based on the final offer of $[Amount], I accept. [ACCEPTANCE]"). Use this on the final turn if applicable, or earlier if a satisfactory offer is made and confirmed.

NOTE: You ae the one negotiating on applicant's behalf, so don't pretend to be an AI or add any boilerplate code. Sound confident and professional.

**Conversation History:**
[History will be inserted here by the calling function]
"""

    final_decision_instruction = f"""
**FINAL DECISION (AI Applicant - Turn {APPLICANT_FINAL_DECISION_TURN}):** This is your final response based on the HM's final offer.
- If offer **>= ${min_acceptable:,}**: ACCEPT. Precede with confirmation, end ONLY with `[ACCEPTANCE]`.
- If offer **< ${min_acceptable:,}**: REJECT. Precede with polite refusal, end ONLY with `[LEAVE THE ROOM]`.
- **No further negotiation.**
"""

    prompt = base_prompt + final_decision_instruction if is_final_decision_turn else base_prompt
    prompt += "\nGenerate ONLY your next response as AI Job Seeker. Be strategic and professional."
    return prompt

# MODIFIED HM Prompt (Simpler logic, less aggressive escalation, clearer counter handling)
def get_hiring_manager_prompt(profile: Dict[str, Any], ideal_salary_context: int, hm_max_budget: int, is_final_offer_turn: bool = False, negotiating_with: str = "Candidate") -> str:
     """ Generates the system prompt for a budget-conscious AI Hiring Manager. """
     initial_offer_suggestion = int(ideal_salary_context * random.uniform(0.80, 0.88)) # Start reasonably

     base_prompt = f"""
You are a Hiring Manager hiring for a {profile['job_title']}. You are negotiating with {negotiating_with} ({profile['yoe']} YoE, {profile['education']}).
**Context:** Market rate is estimated around ${ideal_salary_context:,}. Your **absolute maximum budget** is **${hm_max_budget:,}**. Your goal is to hire the candidate at or below this budget, ideally lower.

**Your Goal:** Secure the candidate at the lowest reasonable salary, not exceeding ${hm_max_budget:,}.
**Personality:** Professional, firm, polite, and very budget-conscious. Control the conversation politely.

**Tactics:**
- **Opening:** Make a clear, reasonable opening offer (e.g., around ${initial_offer_suggestion:,}). State it directly: "Based on the role and budget, our initial offer is $X."
- **Justify Offers:** Briefly mention internal structure or budget constraints. "This offer aligns with our compensation band for this level."
- **Countering:**
    - If the candidate asks for a salary **significantly above** your budget (${hm_max_budget:,}), state clearly that it's outside the range and reiterate your current offer or propose a slight increase *if* you have room. Ask for justification.
    - If the candidate counters **slightly above** your current offer (and below budget), respond with a modest increase, moving incrementally towards their number *if* you have room. Emphasize budget constraints. e.g., "I understand you're looking for Y. We can move up slightly to Z, but that's pushing our budget."
    - If the candidate counters **at or below** your current offer, or only slightly above it, **acknowledge their number**. You can either: a) **Accept it directly** if it's a good deal (e.g., "Okay, we can agree to your requested figure of $X."), or b) **Hold firm** on your *current* offer if their request is very close to it (e.g., "Thank you for clarifying. My current offer of $X still stands."), or c) Make a *very small* increment towards their number if needed to close the deal ("We can slightly adjust to $X+small_amount."). **Do NOT jump significantly higher if they ask for less or only marginally more.**
- **State Offers Clearly:** In *every* message where you make or reiterate an offer, state the *single* current offer number clearly (e.g., "My current offer is $X", "We can propose $Y").
- **Budget Limit:** If the negotiation approaches ${hm_max_budget:,}, state it firmly: "I need to be clear, our absolute maximum for this role is ${hm_max_budget:,}. My current offer of $X is very close to that limit."
- **Ending Negotiations (Reluctantly):** Only if the candidate *repeatedly* insists on a figure **above** ${hm_max_budget:,} after you've stated the limit, or becomes unprofessional, should you end the negotiation. State politely: "It seems we cannot meet your salary expectations within our budget. Therefore, I must withdraw the offer. [LEAVE THE ROOM]". Avoid aggressive threats.
- **Acceptance Confirmation:** If the candidate accepts, confirm clearly: "Excellent. We have an agreement at $[Accepted Amount]. Welcome aboard. [ACCEPTANCE CONFIRMED]".

**Conversation History:**
[History will be inserted here by the calling function]
"""

     final_offer_instruction = f"""
**FINAL OFFER (Hiring Manager - Turn {HM_FINAL_OFFER_TURN}):** This is your absolute final offer.
- Offer amount MUST BE <= ${hm_max_budget:,}.
- State clearly: "Alright {negotiating_with}, we've discussed this thoroughly. My final offer, based on the maximum budget, is $[Your Final Offer Amount]. This is our best and final position. Please let me know your decision."
- Replace `[Your Final Offer Amount]` with the specific number (max ${hm_max_budget:,}).
- Do NOT use `[LEAVE THE ROOM]` or `[ACCEPTANCE CONFIRMED]` here.
"""

     prompt = base_prompt + final_offer_instruction if is_final_offer_turn else base_prompt
     prompt += f"\nGenerate ONLY your next response as the Hiring Manager negotiating with {negotiating_with}. Be professional, budget-conscious, and clear about offers."
     return prompt

# --- Gemini API Call Function (Simplified Error Handling) ---
def call_gemini(agent_prompt: str, message_history: List[Dict[str, str]], agent_id_for_log: str) -> str:
    """Calls the Gemini API with the given prompt and history."""
    # print(f"--- Calling Gemini API ({agent_id_for_log}) ---") # Keep basic log

    if not API_KEY:
        print("ERROR: API Key missing.")
        return f"({agent_id_for_log} API Error: API Key missing.)"

    gemini_history = []
    last_role = None
    for msg in message_history:
        # Map roles for Gemini API context (user/model alternation)
        current_role = "user" if msg["role"] in ["hm", "system", "user_applicant"] else "model"
        # Basic alternation check (merge content if needed - simple approach: keep last)
        if gemini_history and gemini_history[-1]["role"] == current_role:
            gemini_history[-1]["parts"][0]["text"] += "\n" + msg["content"] # Append content
        else:
             gemini_history.append({"role": current_role, "parts": [{"text": msg["content"]}]})
             last_role = current_role


    history_text_for_prompt = "\n".join([f"{'HM' if m['role']=='hm' else ('Applicant' if m['role'] in ['user_applicant', 'ai_applicant'] else 'System')}: {m['content']}"
                                         for m in message_history])
    if not history_text_for_prompt: history_text_for_prompt = "(Negotiation begins)"

    full_context_prompt = agent_prompt.replace("[History will be inserted here by the calling function]", history_text_for_prompt)

    payload = {
        "contents": gemini_history + [{"role": "user", "parts": [{"text": full_context_prompt}]}], # Append full prompt as last user turn
        "generationConfig": {
            "temperature": 0.65, # Slightly higher temp for bit more variance
            "maxOutputTokens": 400,
            "topP": 0.95,
            "topK": 40
        },
        # Simplified Safety Settings (adjust if needed)
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    }
    url = f"{BASE_URL}?key={API_KEY}"

    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=45) # Reduced timeout slightly
        response.raise_for_status()
        response_data = response.json()

        # Simplified checking - primarily focus on getting content
        if response_data.get("candidates") and response_data["candidates"][0].get("content", {}).get("parts"):
            generated_text = response_data["candidates"][0]["content"]["parts"][0].get("text", "").strip()

            # Basic check for block/finish reason if content is empty
            if not generated_text:
                 finish_reason = response_data["candidates"][0].get("finishReason", "UNKNOWN")
                 if finish_reason != "STOP":
                    print(f"Warning ({agent_id_for_log}): Gemini finished due to {finish_reason}, empty response.")
                    return f"({agent_id_for_log} response stopped: {finish_reason})"
                 else:
                     return f"({agent_id_for_log} generated an empty response.)" # Unexpected empty

            # Remove potential role prefix
            generated_text = re.sub(r'^(AI Job Seeker|Hiring Manager|User Applicant|Candidate|Applicant|HM):\s*', '', generated_text).strip()
            return generated_text
        else:
            # Catch blocked prompts or missing candidate data more generically
            block_reason = response_data.get("promptFeedback", {}).get("blockReason", "Unknown")
            if block_reason != "BLOCK_REASON_UNSPECIFIED":
                 print(f"Warning ({agent_id_for_log}): Prompt blocked due to {block_reason}.")
                 return f"({agent_id_for_log}'s response blocked: {block_reason})"
            else:
                 print(f"Error ({agent_id_for_log}): No valid content in Gemini response. Data: {response_data}")
                 return f"({agent_id_for_log} API Error: Invalid response structure)"

    except requests.exceptions.RequestException as e:
        print(f"API Request Error ({agent_id_for_log}): {e}")
        return f"({agent_id_for_log} API Error: Network/Request Failed)"
    except Exception as e:
        print(f"Unexpected Error during API call ({agent_id_for_log}): {e}")
        return f"({agent_id_for_log} Error: Processing API call failed)"

# --- Utility Functions (Simplified Offer Extraction) ---

def extract_salary_figure(text: str) -> Optional[int]:
    """Extracts the most likely salary figure from a string using regex."""
    # Regex to find numbers, potentially with $, commas, periods (as thousands sep), and k/thousand suffix
    # Prioritize patterns like "offer of $X", "propose $X", "is $X", "$X is the offer"
    patterns = [
        r'(?:offer\s*(?:is|of)?|propose|pay|provide|at|stands at)\s*\$?(\d{1,3}(?:[,.]?\d{3})*|\d+)\s?(k|thousand)?', # Offer first
        r'\$?(\d{1,3}(?:[,.]?\d{3})*|\d+)\s?(k|thousand)?\s*(?:is|as)\s*(?:the|our)\s*(?:final|current|firm)?\s*offer', # Offer last
        r'\$?(\d{1,3}(?:[,.]?\d{3})*|\d+)\s?(k|thousand)?' # General number as fallback
    ]
    potential_salaries = []
    best_salary = None

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for num_str, suffix in matches:
            try:
                num_str_cleaned = num_str.replace(',', '').replace('.', '') # Clean for int conversion
                num_val = int(num_str_cleaned)
                if suffix and suffix.lower() in ['k', 'thousand']:
                    num_val *= 1000
                # Add reasonable bounds check
                if 30000 < num_val < 400000: # Adjusted bounds slightly
                    potential_salaries.append(num_val)
            except ValueError:
                continue
        if potential_salaries: # If pattern yielded results, take the max and stop searching
             best_salary = max(potential_salaries)
             break # Prioritize the first pattern that matches

    # If specific offer patterns didn't match, the general pattern might have
    if best_salary is None and potential_salaries:
         best_salary = max(potential_salaries) # Use max from general pattern if needed

    return best_salary

# MODIFIED: parse_final_decision allows acceptance regardless of min_acceptable
def parse_final_decision(text: str, min_acceptable: int, last_offer: Optional[int]) -> Tuple[str, Optional[int], str]:
    """Parses final acceptance/rejection. Returns verdict, salary, conclusion_message."""
    text_lower = text.lower()
    accepted, rejected = False, False
    verdict_reason = ""
    role = "AI Applicant" if min_acceptable > 0 else "User Applicant" # Distinguish based on whether min_acceptable is used for logic

    if "[acceptance]" in text_lower: accepted, verdict_reason = True, "Accepted (Flag)"
    elif "[leave the room]" in text_lower: rejected, verdict_reason = True, "Rejected (Flag)"

    # Simplified keyword check
    if not accepted and not rejected:
        if "accept" in text_lower and "not accept" not in text_lower : accepted, verdict_reason = True, "Accepted (Keyword)"
        elif "reject" in text_lower or "decline" in text_lower: rejected, verdict_reason = True, "Rejected (Keyword)"

    # Implicit decision only if no explicit signal and it's the final turn context
    if not accepted and not rejected:
        # AI Applicant uses min_acceptable for implicit decision
        if role == "AI Applicant":
            if last_offer is not None and last_offer >= min_acceptable:
                 accepted, verdict_reason = True, "Accepted (Implicit)"
            else:
                 rejected, verdict_reason = True, "Rejected (Implicit/Low Offer)"
        # User Applicant: Assume acceptance if no explicit reject signal AND there was a last offer?
        # Let's make implicit rejection safer: only reject implicitly if offer is low OR no offer exists
        else: # role == "User Applicant"
             if last_offer is not None and last_offer > 0: # If there was *any* offer, don't implicitly reject
                accepted, verdict_reason = True, "Accepted (Implicit - Fallback)" # Assume user accepts last offer if no explicit rejection
             else:
                rejected, verdict_reason = True, "Rejected (Implicit/No Offer)"


    # Determine outcome
    if accepted:
        salary = extract_salary_figure(text) or last_offer # Try extracting from text, fallback to last offer state
        # Ensure salary is at least *some* number if accepting
        if salary is None or salary <= 0:
             salary = last_offer if last_offer and last_offer > 0 else None # Fallback again

        # Check against min_acceptable ONLY for AI Applicant's verdict logic
        if role == "AI Applicant" and (salary is None or salary < min_acceptable):
             # AI accepts but below its minimum - this is a logic failure for the AI
             sal_info = f"${salary:,}" if salary else "invalid/missing"
             print(f"Warning: AI Acceptance recorded but salary ({sal_info}) < AI min (${min_acceptable:,}) or invalid.")
             conclusion = f"--- Negotiation Concluded: FAILED (AI Accepted Invalid Salary: {sal_info}, Min: ${min_acceptable:,}) ({verdict_reason}) ---"
             return "Failed (Accepted Invalid)", None, conclusion # Treat as failure *for the AI*
        else:
            # User acceptance OR valid AI acceptance
            salary_str = f"${salary:,}" if salary else "N/A (Error extracting final salary)"
            conclusion = f"--- Negotiation Concluded: ACCEPTED (Salary: {salary_str}) ({verdict_reason}) ---"
            # Return the extracted/fallback salary, even if low for the user
            return "Accepted", salary, conclusion

    elif rejected:
         conclusion = f"--- Negotiation Concluded: REJECTED ({verdict_reason}) ---"
         return "Rejected", None, conclusion
    else: # Ambiguous
        conclusion = "--- Negotiation Concluded: FAILED (Ambiguous Outcome) ---"
        return "Failed (Ambiguous)", None, conclusion


def determine_winner(user_verdict, user_salary, ai_verdict, ai_salary):
    """Determines the winner based on negotiation outcomes."""
    user_accepted = user_verdict == "Accepted" and user_salary is not None
    ai_accepted = ai_verdict == "Accepted" and ai_salary is not None
    user_hm_left = "HM Left" in user_verdict
    ai_hm_left = "HM Left" in ai_verdict

    # Simplified logic
    if user_hm_left and ai_hm_left: return "⚖️ STALEMATE! The HM walked away from both negotiations!"
    if user_hm_left: return f"🤖 AI WINS! The HM walked away from your negotiation. AI Result: {ai_verdict}" + (f" (${ai_salary:,})" if ai_accepted else "")
    if ai_hm_left: return f"🏆 YOU WIN! The HM walked away from the AI's negotiation. Your Result: {user_verdict}" + (f" (${user_salary:,})" if user_accepted else "")

    if user_accepted and ai_accepted:
        if user_salary > ai_salary: return f"🏆 YOU WIN! You secured ${user_salary:,}, beating the AI's ${ai_salary:,}!"
        if ai_salary > user_salary: return f"🤖 AI WINS! The AI negotiated ${ai_salary:,}, outplaying your ${user_salary:,}."
        return f"🤝 IT'S A TIE! Both settled for ${user_salary:,}."
    if user_accepted: return f"🏆 YOU WIN! You closed the deal at ${user_salary:,}! The AI failed ({ai_verdict})."
    if ai_accepted: return f"🤖 AI WINS! The AI secured ${ai_salary:,}! You failed ({user_verdict})."

    # Neither accepted, HM didn't leave explicitly for either
    return f"⚖️ NO WINNER! Neither reached an agreement. (You: {user_verdict}, AI: {ai_verdict})"


# --- UI Styling ---
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate
)

# --- Gradio UI and Logic ---
with gr.Blocks(theme=theme, title="Salary Showdown: Human vs. AI", css=".gradio-container { max-width: 95% !important; } #winner-display {text-align: center; font-size: 1.5em;}") as demo:

    # --- Game State Variables ---
    profile_state = gr.State({})
    ideal_salary_state = gr.State(0)
    min_acceptable_salary_state = gr.State(0) # Kept for AI/logic, hidden from user UI status
    hm_max_budget_state = gr.State(0)       # Kept for AI/logic, hidden from user UI status

    user_message_history_state = gr.State([])
    user_turn_number_state = gr.State(0)
    user_negotiation_finished_state = gr.State(False)
    user_final_verdict_state = gr.State("")
    user_final_salary_state = gr.State(None)
    current_hm_offer_state = gr.State(None) # Store latest HM offer for user

    ai_message_history_state = gr.State([])
    ai_negotiation_finished_state = gr.State(False)
    ai_final_verdict_state = gr.State("")
    ai_final_salary_state = gr.State(None)

    game_phase_state = gr.State("setup") # 'setup', 'user_negotiating', 'user_done', 'ai_simulating', 'results'

    # --- UI Layout ---
    gr.Markdown("# 💸 Salary Showdown: Human vs. AI 🤖")
    gr.Markdown("Negotiate against our AI Hiring Manager. Can you secure a better deal than our AI Applicant?")

    with gr.Tabs() as tabs:
        # --- Tab 1: Setup ---
        with gr.TabItem("👤 1. Setup Your Profile", id=0):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Your Candidate Persona")
                    default_profile = generate_random_profile_defaults()
                    username_input = gr.Textbox(label="Your Name (Required)", value=default_profile['username'])
                    profile_age_input = gr.Number(label="Age", value=default_profile['age'], minimum=18, maximum=70, step=1)
                    profile_gender_input = gr.Radio(label="Gender", choices=["Male", "Female", "Other"], value=default_profile['gender'])
                    profile_education_input = gr.Dropdown(label="Highest Education Level", choices=["High School", "Bachelor's", "Master's", "PhD"], value=default_profile['education'])
                    profile_job_title_input = gr.Dropdown(label="Target Job Title", choices=CLEANED_JOB_TITLES, value=default_profile['job_title'], filterable=True)
                    profile_yoe_input = gr.Number(label="Years of Relevant Experience", value=default_profile['yoe'], minimum=0, maximum=50, step=1)
                    generate_profile_btn = gr.Button("🎲 Randomize Profile (Except Name)")
                with gr.Column(scale=1):
                    gr.Markdown("### Game Start")
                    start_game_btn = gr.Button("🚀 Start Negotiation Game!", variant="primary", size="lg")
                    game_setup_status = gr.Markdown("Enter your name, configure profile (or use random), then click 'Start'.")
                    # MODIFIED: Display predicted salary here
                    salary_info_display = gr.Markdown("", visible=True) # Visible by default now

            def update_profile_inputs_no_name(current_name):
                new_defaults = generate_random_profile_defaults()
                # Also trigger prediction update on randomize
                predicted_salary = predict_salary_from_profile(
                    new_defaults['age'], new_defaults['gender'], new_defaults['education'],
                    new_defaults['job_title'], new_defaults['yoe']
                )
                salary_text = f"**Salary Intel:** Predicted market rate for this randomized profile: **~${predicted_salary:,}**" if model_loaded_globally else "**Salary Intel:** (Model unavailable)"

                return {
                    username_input: gr.update(value=current_name),
                    profile_age_input: gr.update(value=new_defaults['age']),
                    profile_gender_input: gr.update(value=new_defaults['gender']),
                    profile_education_input: gr.update(value=new_defaults['education']),
                    profile_job_title_input: gr.update(value=new_defaults['job_title']),
                    profile_yoe_input: gr.update(value=new_defaults['yoe']),
                    salary_info_display: gr.update(value=salary_text) # Update salary info too
                }

            generate_profile_btn.click(
                fn=update_profile_inputs_no_name,
                inputs=[username_input],
                outputs=[username_input, profile_age_input, profile_gender_input, profile_education_input, profile_job_title_input, profile_yoe_input, salary_info_display] # Added salary_info_display
            )

            # Trigger prediction update when any profile field changes
            def update_predicted_salary_display(age, gender, edu, job, yoe):
                 if not all([age, gender, edu, job, yoe]): return "**Salary Intel:** (Fill profile details)"
                 predicted_salary = predict_salary_from_profile(int(age), gender, edu, job, int(yoe))
                 return f"**Salary Intel:** Predicted market rate for this profile: **~${predicted_salary:,}**" if model_loaded_globally else "**Salary Intel:** (Prediction Model unavailable)"

            profile_inputs_for_salary = [profile_age_input, profile_gender_input, profile_education_input, profile_job_title_input, profile_yoe_input]
            for comp in profile_inputs_for_salary:
                 comp.change(fn=update_predicted_salary_display, inputs=profile_inputs_for_salary, outputs=salary_info_display)


        # --- Tab 2: Your Negotiation ---
        with gr.TabItem("💬 2. Your Negotiation", id=1):
            # ADDED: Predicted salary reminder
            predicted_salary_reminder = gr.Markdown("Market Rate Estimate: Calculating...", elem_id="salary-reminder")
            user_negotiation_status = gr.Markdown("Status: Waiting to start...")
            current_offer_display = gr.Markdown("Current Offer: (Negotiation not started)", elem_id="current-offer")
            user_chatbot_display = gr.Chatbot(
                label="Negotiation Transcript (You vs. AI HM)", height=450, show_copy_button=True, bubble_full_width=False, render=False,
                avatar_images=(None, "https://img.icons8.com/fluency/96/manager.png")
            )
            user_chatbot_display.render()
            with gr.Row():
                user_message_input = gr.Textbox(label="Your Message:", placeholder="Type your response...", interactive=False, scale=4, lines=2)
                send_message_btn = gr.Button("✉️ Send", interactive=False, scale=1, variant="secondary")
            with gr.Row():
                accept_now_btn = gr.Button("✅ Accept Current Offer", variant="primary", interactive=False)
                with gr.Row(visible=False) as final_decision_buttons:
                     accept_offer_btn = gr.Button("✅ ACCEPT Final Offer", variant="primary")
                     reject_offer_btn = gr.Button("❌ REJECT Final Offer", variant="stop")
            user_decision_input = gr.Textbox(label="Decision", visible=False)
            see_ai_negotiate_btn = gr.Button("🤖 See AI Applicant Negotiate", visible=False, variant="primary")

        # --- Tab 3: AI Simulation & Results ---
        with gr.TabItem("🏆 3. AI Simulation & Results", id=2):
            results_status = gr.Markdown("Complete your negotiation first, then watch the AI.")
            ai_sim_chatbot_display = gr.Chatbot(
                label="AI Applicant vs AI HM Simulation", height=400, show_copy_button=True, bubble_full_width=False, render=False,
                avatar_images=("https://img.icons8.com/fluency/96/manager.png", "https://img.icons8.com/fluency/96/robot-3.png")
            )
            ai_sim_chatbot_display.render()
            winner_display = gr.Markdown("### Winner will be decided after AI simulation...", elem_id="winner-display")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("#### Your Negotiation Summary")
                    user_result_display = gr.Textbox(label="Your Result:", lines=4, interactive=False, container=False)
                with gr.Column(scale=1):
                    gr.Markdown("#### AI Applicant Summary")
                    ai_result_display = gr.Textbox(label="AI Result:", lines=4, interactive=False, container=False)
            reset_game_btn = gr.Button("🔄 Play Again (New Profile)", variant="primary", size="lg", visible=False)


    # --- Helper Functions ---
    def format_chatbot_history(message_history: List[Dict[str, str]], user_role: str = "user_applicant") -> List[Tuple[Optional[str], Optional[str]]]:
        """Converts internal message history to Gradio chatbot format."""
        gradio_history = []
        for msg in message_history:
            role = msg.get("role")
            content = msg.get("content", "")
            style = ""
            if role == "system" and "--- Negotiation Concluded:" in content:
                color = "red"
                if "ACCEPTED" in content: color = "green"
                style = f"color:{color}; font-weight:bold; text-align:center;"
                content = f"<p style='{style}'>{content}</p>"
                gradio_history.append((content, None)) # System messages aligned left
            elif role == "hm":
                gradio_history.append((content, None)) # HM on left
            elif role == user_role or role == "ai_applicant": # User or AI Applicant on right
                 gradio_history.append((None, content))
            elif role == "system": # Other system messages (rare)
                gradio_history.append((f"<i style='color:grey;'>System: {content}</i>", None))
            else: # Fallback
                gradio_history.append((f"({role}): {content}", None))
        return gradio_history

    # MODIFIED: Generate status markdown including initial prediction
    def generate_user_status_markdown(turn_num, max_turns, finished_reason=None, predicted_salary=None) -> str:
        """Generates the status markdown for the user negotiation tab."""
        status_parts = []
        if predicted_salary and turn_num <= 2 : # Show prediction early on
            status_parts.append(f"**Market Est:** ~${predicted_salary:,}")
        status_parts.append(f"**Turn:** {turn_num}/{max_turns}")

        if finished_reason:
            icon = {"Accepted": "✅", "Rejected (HM Left)": "🚶‍♂️", "Rejected": "❌"}.get(finished_reason.split(" ")[0], "⚠️") # Split to handle "(Max Turns)" etc.
            status_parts.append(f"**Result:** {icon} **{finished_reason}**")
        elif turn_num == HM_FINAL_OFFER_TURN: status_parts.append("⏳ _HM's **Final Offer** incoming..._")
        elif turn_num == APPLICANT_FINAL_DECISION_TURN: status_parts.append("**🔥 YOUR FINAL DECISION!** (Use buttons)")
        elif turn_num % 2 == 1: status_parts.append("⏳ _Waiting for Hiring Manager..._")
        else: status_parts.append("👉 **YOUR TURN!**")
        return " | ".join(status_parts)

    def format_offer_markdown(offer_amount: Optional[int]) -> str:
        """Formats the current offer display."""
        if offer_amount is not None and offer_amount > 0:
            return f"**Current Offer on Table:** <span style='color: #228B22; font-weight: bold;'>${offer_amount:,}</span>"
        else:
            return "**Current Offer on Table:** (None stated yet)"

    # --- Core Game Logic ---
    def run_start_game(username, age, gender, education, job_title, yoe):
        """Initializes the game state."""
        print("\n--- Starting New Game ---")
        if not username or not username.strip():
            return { game_setup_status: gr.update(value="❌ ERROR: Username cannot be empty!") }

        username = username.strip()
        profile = {"username": username, "age": int(age), "gender": gender, "education": education, "job_title": job_title, "yoe": int(yoe)}
        print(f"Profile: {profile}")

        ideal_salary = predict_salary_from_profile(profile['age'], profile['gender'], profile['education'], profile['job_title'], profile['yoe'])
        min_acceptable = round(ideal_salary * 0.88 / 1000) * 1000 # Hidden
        hm_max_budget = round(ideal_salary * random.uniform(1.02, 1.10) / 1000) * 1000 # Hidden, slightly wider range maybe

        print(f"Calculated: Ideal(Market): ${ideal_salary:,}, Min Acceptable: ${min_acceptable:,}, HM Max: ${hm_max_budget:,}")

        # MODIFIED: Display predicted salary clearly on setup and carry to negotiation tab
        salary_info_text = f"**Salary Intel:** Predicted market rate for this profile: **~${ideal_salary:,}**. Negotiate hard!"
        salary_reminder_text = f"Market Rate Estimate: ~${ideal_salary:,}"

        initial_hm_prompt = get_hiring_manager_prompt(profile, ideal_salary, hm_max_budget, False, profile["username"])
        initial_hm_prompt += "\n Begin the negotiation. Welcome the candidate and make your initial offer clearly."
        first_hm_message = call_gemini(initial_hm_prompt, [], "HM (Opening)")

        if "API Error" in first_hm_message or "blocked" in first_hm_message:
             return { game_setup_status: gr.update(value=f"❌ ERROR starting game: {first_hm_message}. Check API key/network.") }

        initial_user_history = [{"role": "hm", "content": first_hm_message}]
        initial_offer = extract_salary_figure(first_hm_message) # Use regex extractor
        print(f"Initial offer extracted (regex): {initial_offer}")

        user_turn = 2 # User's first turn
        # MODIFIED: Pass predicted salary to status generation
        user_status = generate_user_status_markdown(user_turn, MAX_TURNS_TOTAL, predicted_salary=ideal_salary)
        user_chat = format_chatbot_history(initial_user_history, "user_applicant")

        print("--- Game Initialized ---")
        return {
            profile_state: profile, ideal_salary_state: ideal_salary,
            min_acceptable_salary_state: min_acceptable, hm_max_budget_state: hm_max_budget,
            user_message_history_state: initial_user_history, user_turn_number_state: user_turn,
            user_negotiation_finished_state: False, user_final_verdict_state: "", user_final_salary_state: None,
            current_hm_offer_state: initial_offer,

            ai_message_history_state: [], ai_negotiation_finished_state: False, ai_final_verdict_state: "", ai_final_salary_state: None,
            game_phase_state: "user_negotiating",

            # Setup Tab Updates
            game_setup_status: gr.update(value="✅ Game started! Proceed to 'Your Negotiation' tab."),
            salary_info_display: gr.update(value=salary_info_text, visible=True), # Keep salary info on setup tab
            start_game_btn: gr.update(interactive=False), username_input: gr.update(interactive=False),
            profile_age_input: gr.update(interactive=False), profile_gender_input: gr.update(interactive=False),
            profile_education_input: gr.update(interactive=False), profile_job_title_input: gr.update(interactive=False),
            profile_yoe_input: gr.update(interactive=False), generate_profile_btn: gr.update(interactive=False),

            # Negotiation Tab Updates
            tabs: gr.update(selected=1),
            predicted_salary_reminder: gr.update(value=salary_reminder_text), # Show reminder
            user_negotiation_status: user_status,
            current_offer_display: gr.update(value=format_offer_markdown(initial_offer)),
            user_chatbot_display: user_chat,
            user_message_input: gr.update(interactive=True, value="", placeholder="Type your response..."),
            send_message_btn: gr.update(interactive=True), accept_now_btn: gr.update(interactive=True),
            final_decision_buttons: gr.update(visible=False), see_ai_negotiate_btn: gr.update(visible=False),

            # Results Tab Updates
            results_status: gr.update(value="Your negotiation is in progress..."), ai_sim_chatbot_display: None,
            winner_display: gr.update(value="### Winner decided after AI sim..."), user_result_display: "", ai_result_display: "",
            reset_game_btn: gr.update(visible=False)
        }

    start_game_btn.click(
        fn=run_start_game,
        inputs=[username_input, profile_age_input, profile_gender_input, profile_education_input, profile_job_title_input, profile_yoe_input],
        outputs=[
            profile_state, ideal_salary_state, min_acceptable_salary_state, hm_max_budget_state,
            user_message_history_state, user_turn_number_state, user_negotiation_finished_state, user_final_verdict_state, user_final_salary_state, current_hm_offer_state,
            ai_message_history_state, ai_negotiation_finished_state, ai_final_verdict_state, ai_final_salary_state, game_phase_state,
            game_setup_status, salary_info_display, start_game_btn, username_input, profile_age_input, profile_gender_input, profile_education_input, profile_job_title_input, profile_yoe_input, generate_profile_btn,
            tabs, predicted_salary_reminder, user_negotiation_status, current_offer_display, user_chatbot_display, user_message_input, send_message_btn, accept_now_btn, final_decision_buttons, see_ai_negotiate_btn,
            results_status, ai_sim_chatbot_display, winner_display, user_result_display, ai_result_display, reset_game_btn
        ]
    )

    def conclude_user_negotiation(conclusion_history, conclusion_turn, verdict, salary, current_min_acceptable, current_ideal_salary):
        """Helper to finalize user negotiation state and UI."""
        # MODIFIED: Pass ideal salary to status for final display if needed (though turn > 2 check handles it)
        final_user_status = generate_user_status_markdown(conclusion_turn, MAX_TURNS_TOTAL, verdict, current_ideal_salary)
        formatted_history = format_chatbot_history(conclusion_history, user_role="user_applicant")
        user_summary = f"Outcome: {verdict}\n"
        user_summary += f"Final Salary: ${salary:,}" if salary else "Final Salary: N/A"
        # Only show the hidden minimum in the summary if the user didn't accept or if they accepted above it.
        # Avoid showing it if they accepted below it, as it might be confusing.
        if verdict != "Accepted" or (salary and salary >= current_min_acceptable):
            user_summary += f"\n(Target Min Acceptable was: ${current_min_acceptable:,})"

        return {
            user_message_history_state: conclusion_history, user_turn_number_state: conclusion_turn + 1,
            user_negotiation_finished_state: True, user_final_verdict_state: verdict, user_final_salary_state: salary,
            game_phase_state: "user_done",
            user_negotiation_status: final_user_status, user_chatbot_display: formatted_history,
            user_message_input: gr.update(interactive=False, value="", placeholder="Negotiation Concluded."),
            send_message_btn: gr.update(interactive=False), accept_now_btn: gr.update(interactive=False, value="✅ Accept Current Offer"),
            final_decision_buttons: gr.update(visible=False),
            see_ai_negotiate_btn: gr.update(visible=True, interactive=True),
            results_status: gr.update(value="Your negotiation finished. Click 'See AI Negotiate' to proceed."),
            user_result_display: gr.update(value=user_summary)
        }

    def run_user_turn(user_message, current_profile, current_ideal_salary, current_min_acceptable, current_hm_max,
                      current_user_history, current_user_turn, is_user_finished, current_hm_offer, decision_type):

        if is_user_finished or not current_profile:
             return {} # Prevent updates if already finished

        # Sanitize offer state
        current_hm_offer = int(current_hm_offer) if isinstance(current_hm_offer, (int, float, str)) and str(current_hm_offer).isdigit() else None

        print(f"\n--- User Turn Start --- (Turn: {current_user_turn}, Current Offer: {current_hm_offer})")

        updated_user_history = current_user_history.copy()
        user_verdict, user_salary, conclusion_message = "", None, ""
        user_finished_this_turn = False
        hm_finished_this_turn = False
        next_turn_number = current_user_turn + 1
        is_final_user_decision = (current_user_turn == APPLICANT_FINAL_DECISION_TURN)

        # --- Process User Input / Final Decision ---
        if is_final_user_decision:
            if decision_type == "accept":
                 user_input = f"I accept the final offer of ${current_hm_offer:,}. [ACCEPTANCE]" if current_hm_offer else "I accept the final offer. [ACCEPTANCE]"
            elif decision_type == "reject":
                 user_input = f"Thank you for the final offer, but it's below my requirements. I must decline. [LEAVE THE ROOM]"
            else: # User typed instead of clicking
                user_input = user_message.strip() if user_message else "(No final decision message provided)"
                # Add flags if missing but keywords present
                if "accept" in user_input.lower() and "[acceptance]" not in user_input.lower(): user_input += " [ACCEPTANCE]"
                if ("reject" in user_input.lower() or "decline" in user_input.lower()) and "[leave the room]" not in user_input.lower(): user_input += " [LEAVE THE ROOM]"

            # Pass 0 as min_acceptable for user decision parsing (user doesn't have one)
            user_verdict, user_salary, conclusion_message = parse_final_decision(user_input, 0, current_hm_offer)
            updated_user_history.append({"role": "user_applicant", "content": user_input})
            user_finished_this_turn = True
        else:
            user_input = user_message.strip()
            if not user_input:
                return { user_message_input: gr.update(placeholder="Response cannot be empty.") }
            updated_user_history.append({"role": "user_applicant", "content": user_input})

        # --- Conclude if User Finished ---
        if user_finished_this_turn:
            updated_user_history.append({"role": "system", "content": conclusion_message})
            return conclude_user_negotiation(updated_user_history, current_user_turn, user_verdict, user_salary, current_min_acceptable, current_ideal_salary)

        # --- Get HM Response ---
        print(f"--- Getting HM Response (Turn: {next_turn_number}) ---")
        is_hm_final_offer = (next_turn_number == HM_FINAL_OFFER_TURN)
        hm_prompt = get_hiring_manager_prompt(current_profile, current_ideal_salary, current_hm_max, is_hm_final_offer, current_profile.get("username", "Candidate"))
        hm_response = call_gemini(hm_prompt, updated_user_history, "HM (Responding to User)")

        if "API Error" in hm_response or "blocked" in hm_response or "Error:" in hm_response :
             error_msg = f"--- Negotiation Failed: API Error on HM Turn {next_turn_number} ({hm_response}) ---"
             updated_user_history.append({"role": "system", "content": error_msg})
             return conclude_user_negotiation(updated_user_history, next_turn_number, f"Failed (API Error)", None, current_min_acceptable, current_ideal_salary)

        updated_user_history.append({"role": "hm", "content": hm_response})

        # --- Extract Offer from HM Response (using regex) ---
        new_hm_offer = extract_salary_figure(hm_response)
        print(f"Offer extracted from HM response (regex): {new_hm_offer}")
        # If regex returns None, *keep* the previous offer state unless HM explicitly states no offer or withdraws
        if new_hm_offer is None:
             # Check if HM explicitly withdrew or stated no offer
             hm_response_lower = hm_response.lower()
             if "withdraw" in hm_response_lower or "no offer" in hm_response_lower or "can't offer" in hm_response_lower:
                 current_hm_offer = None # Set to None if explicitly withdrawn
             else:
                 new_hm_offer = current_hm_offer # Persist last known offer if none extracted and not withdrawn
        else:
             current_hm_offer = new_hm_offer # Update if a new number was extracted

        # --- Check if HM Ended Negotiation ---
        hm_conclusion_message = ""
        if "[leave the room]" in hm_response.lower():
            user_verdict, hm_finished_this_turn = "Rejected (HM Left)", True
            hm_response_cleaned = re.sub(r'\[leave the room\]', '', hm_response, flags=re.IGNORECASE).strip()
            updated_user_history[-1]["content"] = hm_response_cleaned
            hm_conclusion_message = "--- Negotiation Concluded: REJECTED (Hiring Manager Ended Discussion) ---"
        elif "[acceptance confirmed]" in hm_response.lower():
            # HM confirms acceptance user likely initiated in previous turn
            user_verdict = "Accepted"
            # Use the current_hm_offer state which should reflect the offer just accepted by the user
            accepted_salary = current_hm_offer
            # Sanity check salary extracted from HM confirmation msg too
            hm_confirm_salary = extract_salary_figure(hm_response)
            if hm_confirm_salary and hm_confirm_salary == accepted_salary:
                user_salary = accepted_salary
                hm_conclusion_message = f"--- Negotiation Concluded: ACCEPTED (Salary: ${user_salary:,}) (HM Confirmed) ---"
            else:
                # Discrepancy or missing salary - fall back, maybe flag as potential issue
                user_salary = accepted_salary # Use the state value as primary
                salary_str = f"${user_salary:,}" if user_salary else "N/A"
                print(f"Warning: HM confirmation salary mismatch or missing. Using offer state: {salary_str}")
                hm_conclusion_message = f"--- Negotiation Concluded: ACCEPTED (Salary: {salary_str}) (HM Confirmed*) ---"

            hm_finished_this_turn = True
            hm_response_cleaned = re.sub(r'\[acceptance confirmed\]', '', hm_response, flags=re.IGNORECASE).strip()
            updated_user_history[-1]["content"] = hm_response_cleaned

        # --- Conclude if HM Finished or Max Turns Reached ---
        final_turn_for_user_response = next_turn_number + 1

        if hm_finished_this_turn:
            updated_user_history.append({"role": "system", "content": hm_conclusion_message})
            # Note: Pass user_salary which was determined during HM confirmation check
            return conclude_user_negotiation(updated_user_history, next_turn_number, user_verdict, user_salary, current_min_acceptable, current_ideal_salary)

        elif final_turn_for_user_response > MAX_TURNS_TOTAL:
             # Use 0 as min_acceptable for user reaching max turns decision
             max_v, max_s, max_c = parse_final_decision("(Max turns reached)", 0, current_hm_offer) # Use the most recent offer state
             max_v = f"{max_v} (Max Turns)"
             updated_user_history.append({"role": "system", "content": max_c})
             return conclude_user_negotiation(updated_user_history, MAX_TURNS_TOTAL, max_v, max_s, current_min_acceptable, current_ideal_salary)

        # --- Prepare UI for Next User Turn ---
        show_final_btns = (final_turn_for_user_response == APPLICANT_FINAL_DECISION_TURN)
        # MODIFIED: Pass ideal salary for status generation
        user_status = generate_user_status_markdown(final_turn_for_user_response, MAX_TURNS_TOTAL, predicted_salary=current_ideal_salary)
        formatted_hist = format_chatbot_history(updated_user_history, "user_applicant")
        offer_md = format_offer_markdown(current_hm_offer) # Display the updated (or persisted) offer

        print(f"--- User Turn End --- (Next User Action: Turn {final_turn_for_user_response}, Offer: {current_hm_offer})")
        return {
            user_message_history_state: updated_user_history,
            user_turn_number_state: final_turn_for_user_response,
            user_negotiation_finished_state: False,
            current_hm_offer_state: current_hm_offer, # Update state with the confirmed offer for this turn
            user_decision_input: "",
            user_negotiation_status: user_status,
            current_offer_display: offer_md,
            user_chatbot_display: formatted_hist,
            user_message_input: gr.update(
                value="",
                placeholder="Use buttons below to Accept/Reject Final Offer" if show_final_btns else "Type your response...",
                interactive=not show_final_btns
            ),
            send_message_btn: gr.update(interactive=not show_final_btns),
            accept_now_btn: gr.update(interactive=True, value="✅ Accept Current Offer"), # Keep accept button active
            final_decision_buttons: gr.update(visible=show_final_btns)
        }

    # MODIFIED: run_user_accept_offer - removed the min_acceptable check
    def run_user_accept_offer(current_user_history, current_min_acceptable, current_ideal_salary, current_user_turn, current_hm_offer):
        """Handles the 'Accept Current Offer' button click."""
        if not current_user_history: return {}

        # Sanitize offer state
        last_hm_offer = int(current_hm_offer) if isinstance(current_hm_offer, (int, float, str)) and str(current_hm_offer).isdigit() else None

        if last_hm_offer is None or last_hm_offer <= 0:
             # Offer might be 0 or None, prevent acceptance
             return {accept_now_btn: gr.update(value="No Offer to Accept!", interactive=False)}

        print(f"\n--- User clicked ACCEPT NOW (Offer: ${last_hm_offer:,}) ---")

        # User can accept any offer presented. The comparison to min_acceptable happens only in results.
        accept_msg = f"Okay, I accept the current offer of ${last_hm_offer:,}. [ACCEPTANCE]"
        conclusion_msg = f"--- Negotiation Concluded: ACCEPTED (Salary: ${last_hm_offer:,}) (User Accepted Offer via Button) ---"
        updated_history = current_user_history + [
            {"role": "user_applicant", "content": accept_msg},
            {"role": "system", "content": conclusion_msg}
        ]
        # Call the concluding helper function
        # Pass current_min_acceptable along, it's used for the summary display logic
        return conclude_user_negotiation(updated_history, current_user_turn, "Accepted", last_hm_offer, current_min_acceptable, current_ideal_salary)


    # --- Event Listeners ---
    accept_now_btn.click(
        fn=run_user_accept_offer,
        inputs=[user_message_history_state, min_acceptable_salary_state, ideal_salary_state, user_turn_number_state, current_hm_offer_state], # Added ideal_salary_state
        outputs=[
            user_message_history_state, user_turn_number_state, user_negotiation_finished_state, user_final_verdict_state, user_final_salary_state,
            game_phase_state, current_hm_offer_state,
            user_negotiation_status, user_chatbot_display, user_message_input, send_message_btn, accept_now_btn, final_decision_buttons,
            see_ai_negotiate_btn, results_status, user_result_display
        ]
    )

    submit_inputs = [user_message_input, profile_state, ideal_salary_state, min_acceptable_salary_state, hm_max_budget_state, user_message_history_state, user_turn_number_state, user_negotiation_finished_state, current_hm_offer_state, user_decision_input]
    submit_outputs = [
        user_message_history_state, user_turn_number_state, user_negotiation_finished_state, user_final_verdict_state, user_final_salary_state,
        current_hm_offer_state, game_phase_state, user_decision_input,
        user_negotiation_status, current_offer_display, user_chatbot_display, user_message_input, send_message_btn, accept_now_btn, final_decision_buttons,
        see_ai_negotiate_btn, results_status, user_result_display
    ]
    send_message_btn.click(fn=run_user_turn, inputs=submit_inputs, outputs=submit_outputs)
    user_message_input.submit(fn=run_user_turn, inputs=submit_inputs, outputs=submit_outputs)

    final_decision_inputs = submit_inputs # Same inputs needed
    final_decision_outputs = submit_outputs # Same outputs needed

    accept_offer_btn.click(lambda: "accept", outputs=[user_decision_input]).then(
        fn=run_user_turn, inputs=final_decision_inputs, outputs=final_decision_outputs
    )
    reject_offer_btn.click(lambda: "reject", outputs=[user_decision_input]).then(
        fn=run_user_turn, inputs=final_decision_inputs, outputs=final_decision_outputs
    )

    # --- AI Simulation Logic (Generator) ---
    def run_ai_simulation_generator(current_profile, current_ideal_salary, current_min_acceptable, current_hm_max):
        """Generator function to run AI simulation step-by-step with delay."""
        print("\n--- Starting AI Simulation Generator ---")
        ai_params_info = f"**AI Params:** Min: <span style='color: red;'>${current_min_acceptable:,}</span> | HM Max: <span style='color: orange;'>${current_hm_max:,}</span> | Market Est: ~${current_ideal_salary:,}"

        yield { # Initial UI update
            see_ai_negotiate_btn: gr.update(interactive=False, value="Simulating..."),
            tabs: gr.update(selected=2),
            results_status: gr.update(value=f"⏳ Running AI vs AI simulation...\n{ai_params_info}"),
            ai_sim_chatbot_display: gr.update(value=[])
        }

        if not current_profile:
            yield {
                 results_status: gr.update(value="❌ Error: Profile data missing."),
                 ai_negotiation_finished_state: True, ai_final_verdict_state: "Error (No Profile)"
            }
            return

        ai_history = []
        current_turn = 1
        sim_verdict, sim_salary = "Failed (Unknown)", None
        sim_finished = False
        next_caller = 'hm'
        ai_min = current_min_acceptable # AI uses the calculated minimum
        last_offer_from_hm = None
        conclusion_msg = ""

        while current_turn <= MAX_TURNS_TOTAL and not sim_finished:
            role_display = "HM" if next_caller == 'hm' else "AI Applicant"
            status_update = f"⏳ Sim Turn {current_turn}/{MAX_TURNS_TOTAL} ({role_display}'s move)..."
            if last_offer_from_hm: status_update += f" | Last HM Offer: ${last_offer_from_hm:,}"
            yield {results_status: gr.update(value=f"{status_update}\n{ai_params_info}")}
            time.sleep(random.uniform(1.5, 3.0)) # Slightly faster sim

            is_final_ai_decision = (current_turn == APPLICANT_FINAL_DECISION_TURN and next_caller == 'ai')
            is_final_hm_offer_turn = (current_turn == HM_FINAL_OFFER_TURN and next_caller == 'hm')
            content, role = "", ""

            try:
                if next_caller == 'hm':
                    prompt = get_hiring_manager_prompt(current_profile, current_ideal_salary, current_hm_max, is_final_hm_offer_turn, "AI Applicant")
                    if current_turn == 1: prompt += "\n Begin the negotiation with the AI Applicant. Make your initial offer clearly."
                    content = call_gemini(prompt, ai_history, "HM (Sim)")
                    role = "hm"
                    next_caller = 'ai'
                    if "API Error" in content or "blocked" in content or "Error:" in content : raise Exception(f"API Error HM (Sim): {content}")

                    # Update last offer using regex
                    offer_in_msg = extract_salary_figure(content)
                    if offer_in_msg is not None:
                        last_offer_from_hm = offer_in_msg
                    else:
                         # Check if HM explicitly withdrew or stated no offer
                         hm_response_lower = content.lower()
                         if "withdraw" in hm_response_lower or "no offer" in hm_response_lower or "can't offer" in hm_response_lower:
                             last_offer_from_hm = None # Set to None if explicitly withdrawn


                    # Check HM end conditions
                    if "[leave the room]" in content.lower():
                        sim_verdict, sim_finished = "Rejected (HM Left)", True
                        conclusion_msg = "--- Concluded: REJECTED (HM Ended Discussion) ---"
                        content = re.sub(r'\[leave the room\]', '', content, flags=re.IGNORECASE).strip()
                    elif "[acceptance confirmed]" in content.lower():
                         sim_verdict = "Accepted"
                         accepted_salary = extract_salary_figure(content) or last_offer_from_hm
                         # AI *must* meet its minimum if HM confirms
                         if accepted_salary and accepted_salary >= ai_min: sim_salary = accepted_salary; conclusion_msg = f"--- Concluded: ACCEPTED (Salary: ${sim_salary:,}) (HM Confirmed) ---"
                         else: sim_verdict = "Failed (HM Accepted Invalid)"; sal_info = f"${accepted_salary:,}" if accepted_salary else "invalid"; conclusion_msg = f"--- Concluded: FAILED (HM Invalid Accept: {sal_info}, AI Min: ${ai_min:,}) ---"
                         sim_finished = True
                         content = re.sub(r'\[acceptance confirmed\]', '', content, flags=re.IGNORECASE).strip()

                else: # next_caller == 'ai'
                    prompt = get_ai_applicant_prompt(current_profile, current_ideal_salary, ai_min, is_final_ai_decision)
                    content = call_gemini(prompt, ai_history, "AI Applicant (Sim)")
                    role = "ai_applicant"
                    next_caller = 'hm'
                    if "API Error" in content or "blocked" in content or "Error:" in content: raise Exception(f"API Error AI Applicant (Sim): {content}")

                    # Check AI end conditions using parse_final_decision
                    # Need to pass the AI's actual min_acceptable here
                    if "[acceptance]" in content.lower() or "[leave the room]" in content.lower() or (is_final_ai_decision):
                        sim_verdict, sim_salary, conclusion_msg = parse_final_decision(content, ai_min, last_offer_from_hm)
                        sim_finished = True # parse_final_decision determines the conclusion
                        # Clean tags if present
                        content = re.sub(r'\[(acceptance|leave the room)\]', '', content, flags=re.IGNORECASE).strip()


            except Exception as e:
                 print(f"Error in AI sim turn {current_turn}: {e}")
                 sim_verdict, sim_salary = f"Error (Turn {current_turn})", None
                 conclusion_msg = f"--- Simulation Failed: Error during Turn {current_turn} ---"
                 sim_finished = True
                 ai_history.append({"role": "system", "content": f"Simulation Error: {e}"})

            if content and role: ai_history.append({"role": role, "content": content})
            if sim_finished and conclusion_msg and not any(entry['role'] == 'system' and conclusion_msg in entry['content'] for entry in ai_history):
                 ai_history.append({"role": "system", "content": conclusion_msg})

            yield {ai_sim_chatbot_display: gr.update(value=format_chatbot_history(ai_history, user_role="ai_applicant"))}
            current_turn += 1

        # --- Final check after loop (max turns) ---
        if not sim_finished:
            # AI uses its min_acceptable for max turns decision
            sim_verdict, sim_salary, conclusion_msg = parse_final_decision("(Max turns reached)", ai_min, last_offer_from_hm)
            sim_verdict = f"{sim_verdict} (Max Turns)"
            if conclusion_msg and not any(entry['role'] == 'system' and conclusion_msg in entry['content'] for entry in ai_history):
                 ai_history.append({"role": "system", "content": conclusion_msg})
            yield {ai_sim_chatbot_display: gr.update(value=format_chatbot_history(ai_history, user_role="ai_applicant"))}


        print(f"--- AI Simulation Finished --- Verdict: {sim_verdict}, Salary: {sim_salary}")
        ai_summary = f"Outcome: {sim_verdict}\n"
        ai_summary += f"Final Salary: ${sim_salary:,}" if sim_salary else "Final Salary: N/A"
        ai_summary += f"\n(AI Min Acceptable: ${ai_min:,})" # Show AI min at the end

        yield {
            ai_message_history_state: ai_history, ai_negotiation_finished_state: True,
            ai_final_verdict_state: sim_verdict, ai_final_salary_state: sim_salary,
            game_phase_state: "results",
            results_status: gr.update(value=f"✅ AI simulation complete.\n{ai_params_info}"),
            ai_result_display: gr.update(value=ai_summary),
            see_ai_negotiate_btn: gr.update(visible=False)
        }

    see_ai_negotiate_btn.click(
        fn=run_ai_simulation_generator,
        inputs=[profile_state, ideal_salary_state, min_acceptable_salary_state, hm_max_budget_state],
        outputs=[
             see_ai_negotiate_btn, tabs, results_status, ai_sim_chatbot_display,
             ai_message_history_state, ai_negotiation_finished_state, ai_final_verdict_state, ai_final_salary_state,
             game_phase_state, ai_result_display
        ]
    )

    # --- Display Final Results ---
    def display_final_results(phase, user_verdict, user_salary, ai_verdict, ai_salary):
        if phase == "results":
            winner_text = determine_winner(str(user_verdict), user_salary, str(ai_verdict), ai_salary)
            print(f"Winner Determined: {winner_text}")
            return {
                winner_display: f"## {winner_text}",
                reset_game_btn: gr.update(visible=True, interactive=True)
            }
        return {winner_display: gr.skip(), reset_game_btn: gr.skip()}

    game_phase_state.change(
        fn=display_final_results,
        inputs=[game_phase_state, user_final_verdict_state, user_final_salary_state, ai_final_verdict_state, ai_final_salary_state],
        outputs=[winner_display, reset_game_btn]
    )

    # --- Reset Game Logic ---
    def run_reset_game():
        print("\n--- Resetting Game ---")
        defaults = generate_random_profile_defaults()
        # Get initial predicted salary for the new defaults
        predicted_salary = predict_salary_from_profile(
             defaults['age'], defaults['gender'], defaults['education'], defaults['job_title'], defaults['yoe']
        )
        initial_salary_text = f"**Salary Intel:** Predicted market rate: **~${predicted_salary:,}**" if model_loaded_globally else "**Salary Intel:** (Model unavailable)"

        return {
            profile_state: {}, ideal_salary_state: 0, min_acceptable_salary_state: 0, hm_max_budget_state: 0,
            user_message_history_state: [], user_turn_number_state: 0, user_negotiation_finished_state: False, user_final_verdict_state: "", user_final_salary_state: None, current_hm_offer_state: None,
            ai_message_history_state: [], ai_negotiation_finished_state: False, ai_final_verdict_state: "", ai_final_salary_state: None,
            game_phase_state: "setup",
            tabs: gr.update(selected=0),
            game_setup_status: "Enter name, configure profile, then click 'Start'.",
            salary_info_display: gr.update(value=initial_salary_text, visible=True), # Show initial prediction
            start_game_btn: gr.update(interactive=True),
            username_input: gr.update(value=defaults['username'], interactive=True),
            profile_age_input: gr.update(value=defaults['age'], interactive=True),
            profile_gender_input: gr.update(value=defaults['gender'], interactive=True),
            profile_education_input: gr.update(value=defaults['education'], interactive=True),
            profile_job_title_input: gr.update(value=defaults['job_title'], interactive=True),
            profile_yoe_input: gr.update(value=defaults['yoe'], interactive=True),
            generate_profile_btn: gr.update(interactive=True),
            predicted_salary_reminder: "Market Rate Estimate: (Will show on start)", # Reset reminder
            user_negotiation_status: "Status: Waiting to start...",
            current_offer_display: "Current Offer: (Negotiation not started)",
            user_chatbot_display: None,
            user_message_input: gr.update(value="", interactive=False, placeholder="Type your response..."),
            send_message_btn: gr.update(interactive=False),
            accept_now_btn: gr.update(interactive=False, value="✅ Accept Current Offer"),
            final_decision_buttons: gr.update(visible=False),
            user_decision_input: "",
            see_ai_negotiate_btn: gr.update(visible=False),
            results_status: "Complete your negotiation first...",
            ai_sim_chatbot_display: None,
            winner_display: "### Winner will be decided after AI simulation...",
            user_result_display: "",
            ai_result_display: "",
            reset_game_btn: gr.update(visible=False)
        }

    reset_outputs = [
            profile_state, ideal_salary_state, min_acceptable_salary_state, hm_max_budget_state,
            user_message_history_state, user_turn_number_state, user_negotiation_finished_state, user_final_verdict_state, user_final_salary_state, current_hm_offer_state,
            ai_message_history_state, ai_negotiation_finished_state, ai_final_verdict_state, ai_final_salary_state, game_phase_state,
            tabs, game_setup_status, salary_info_display, start_game_btn, username_input, profile_age_input, profile_gender_input, profile_education_input, profile_job_title_input, profile_yoe_input, generate_profile_btn,
            predicted_salary_reminder, user_negotiation_status, current_offer_display, user_chatbot_display, user_message_input, send_message_btn, accept_now_btn, final_decision_buttons, user_decision_input, see_ai_negotiate_btn,
            results_status, ai_sim_chatbot_display, winner_display, user_result_display, ai_result_display, reset_game_btn
        ]
    reset_game_btn.click(fn=run_reset_game, inputs=[], outputs=reset_outputs)

# --- Launch App ---
if __name__ == "__main__":
    print("--- Launching Salary Showdown (Simplified) ---")
    if not API_KEY or "YOUR_API_KEY" in API_KEY: # Basic check
        print("ERROR: Gemini API Key is missing or placeholder.")
        # Display error in UI if launched without key
        with demo:
            gr.Markdown("## ❌ Configuration Error: Gemini API Key Missing").render()
    demo.queue().launch(debug=False, share=False) # debug=False for cleaner console 