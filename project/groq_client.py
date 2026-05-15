import os
import requests
from typing import List, Dict, Any
import json
import time


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key

    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("groq_api_key", "")
    return ""


def generate_advisory(
    prediction: str,
    risk_factors: List[str],
    actions: List[str],
    pond_id: str = "unknown"
) -> Dict[str, Any]:
    """
    Generate structured advisory using Groq LLM (production-grade).

    Returns:
        Dict with:
        - explanation (scientific)
        - farmer_message (simple)
        - advisor_note (empathetic professional)
    """

    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("Groq API key not configured")

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    risk_factors_str = ", ".join(risk_factors) if risk_factors else "standard parameters"
    actions_str = ", ".join(actions) if actions else "maintain standard practices"

    prompt  = (
    "RESPOND WITH ONLY VALID JSON. NO MARKDOWN, NO CODE BLOCKS, NO EXTRA TEXT. "
    "Output EXACTLY this JSON structure: "
    '{"explanation": "...", "farmer_message": "...", "advisor_note": "..."} '

    f"Disease Risk: {prediction}. Risk Factors: {risk_factors_str}. Actions: {actions_str}. "

    "Explanation (MANDATORY STYLE): "
    "- Write as an experienced shrimp farmer + aquaculture advisor guiding another farmer. "
    "- Maintain professional accuracy but use natural, field-level language. "
    "- 3–4 sentences ONLY, single paragraph, NO newline characters. "
    "- MUST include: "
    "(1) What is happening in the pond (risk + biological reason like stress, Vibrio, low oxygen) "
    "(2) What to do NOW (clear actions with practical detail) "
    "(3) Monitoring plan (what to check and how often: DO, ammonia, pH, turbidity) "
    "(4) Expected improvement timeline (e.g., 24–72 hours) "

    "Use domain terms naturally: dissolved oxygen, ammonia, pH, turbidity, probiotic, Vibrio, immune stress, metabolic stress. "

    "Tone rules for explanation: "
    "- Supportive, practical, field-oriented "
    "- No corporate or formal report language "
    "- No phrases like 'I recommend', 'we suggest', 'touch base' "
    "- Should feel like a farmer advising another farmer during a real situation "

    "Farmer_message: "
    "- Very simple English, 1–2 lines "
    "- Direct instructions only (what to do immediately) "

    "Advisor_note: "
    "- Calm, supportive, slightly empathetic but confident "
    "- 1–2 sentences "
    "- Reinforce urgency without panic, like an experienced advisor on-site "
)

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,  # lower = more consistent
        "max_tokens": 400,
    }

    # 🔁 Retry mechanism
    for attempt in range(3):
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"[Attempt {attempt+1}] Groq API Error: {response.status_code}")
                print(f"Response: {response.text}")
                time.sleep(2)
                continue

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            # 🧠 Parse JSON safely - handle markdown code blocks
            try:
                # Try direct JSON parse first
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError:
                # Try extracting JSON from markdown code block
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                    return parsed
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                    return parsed
                else:
                    # Try to find JSON object in content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        return parsed
                    raise json.JSONDecodeError("No valid JSON found", content, 0)

        except requests.exceptions.RequestException as e:
            print(f"[Attempt {attempt+1}] Request failed: {e}")
            time.sleep(2)
        except (KeyError, ValueError) as e:
            print(f"[Attempt {attempt+1}] Response parsing error: {e}")
            time.sleep(2)

    # Log final failure details
    print(f"❌ Groq API failed after 3 attempts. API Key present: {bool(api_key)}")
    raise RuntimeError("Failed to generate valid advisory after retries")


def generate_natural_language(
    prediction: str,
    risk_factors: List[str],
    actions: List[str],
    pond_id: str = "unknown"
) -> str:
    """
    Wrapper for backward compatibility. Returns formatted explanation string.
    Falls back to template-based explanation if Groq fails.
    """
    try:
        advisory = generate_advisory(prediction, risk_factors, actions, pond_id)
        return advisory.get("explanation", "Unable to generate explanation.")
    except Exception as e:
        print(f"Groq API failed, using fallback: {e}")
        # Template-based fallback (no LLM required)
        risk_level = prediction.lower()
        factors = ", ".join(risk_factors) if risk_factors else "monitored parameters"
        
        fallback_explanations = {
            "low": f"The aquaculture system demonstrates stable water chemistry and low disease pressure. Monitor dissolved oxygen (≥5 mg/L), ammonia (<0.5 mg/L), and pH (7.2-8.0) on alternate days. Standard probiotic application (10^8 CFU/mL) weekly maintains beneficial microbial populations. Key parameters ({factors}) remain within optimal ranges; no therapeutic intervention required. Continue standard feeding protocols and expect 95%+ survival rates.",
            
            "moderate": f"The system shows early stress indicators with moderate disease risk driven by {factors}. Immediate actions: (1) Increase aeration to maintain DO >5 mg/L 24/7; (2) Daily water quality monitoring (dissolved oxygen, ammonia, temperature); (3) Apply probiotic treatment (Bacillus/Vibrio antagonists) at 10^8 CFU/mL every 48 hours for 7 days; (4) Reduce feeding by 20% to lower organic load. Expected improvement within 5-7 days with proper management.",
            
            "high": f"Critical disease risk detected with physiological stress indicated by {factors}. URGENT: (1) Monitor water quality every 6 hours - target DO >6 mg/L, ammonia <0.3 mg/L, pH 7.5-8.2; (2) Perform 25-30% water exchange immediately; (3) Administer prophylactic antibiotic (oxytetracycline 75mg/kg or florfenicol 30mg/kg for 5 days) with enhanced probiotics (Bacillus subtilis 10^9 CFU/mL); (4) Reduce feeding to 50% ration; (5) Implement immuno-stimulants (beta-glucans, vitamin C). Immune system recovery expected in 10-14 days if interventions start immediately.",
            
            "critical": f"EMERGENCY: System parameters ({factors}) show acute collapse requiring immediate intervention. ACTIONS REQUIRED NOW: (1) Continuous aeration - target DO >8 mg/L minimum; (2) Emergency water exchange (40-50%); (3) Intensive antimicrobial therapy: oxytetracycline 100mg/kg or cephalosporin 50mg/kg daily for 7 days + probiotics (10^10 CFU/mL); (4) Stop all feeding for 24 hours, then 25% ration only; (5) Administer immunostimulants and stress hormones (cortisol management); (6) Monitor every 3-4 hours. Success depends on immediate full implementation - delay increases mortality risk significantly."
        }
        
        return fallback_explanations.get(risk_level, f"Disease Risk: {prediction}. Key Factors: {factors}.")