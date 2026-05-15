import yaml
from typing import List, Dict

# Load thresholds and other config values

def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def recommend_actions(risk: str, risk_factors: List[str], config: dict) -> List[str]:
    """Return a list of mitigation actions based on disease risk level and key factors.
    Args:
        risk: Predicted risk level ("Low", "Moderate", "High").
        risk_factors: List of top contributing feature names (e.g., "do_min_night").
        config: Loaded configuration containing thresholds.
    """
    actions = []
    thresholds = config.get("thresholds", {})

    # Generic actions per risk level
    if risk == "High":
        actions.append("Increase aeration during night hours")
        actions.append("Reduce feeding by 15%")
        actions.append("Apply probiotic treatment")
    elif risk == "Moderate":
        actions.append("Monitor dissolved oxygen closely")
        actions.append("Adjust feeding schedule if mortality rises")
    else:  # Low
        actions.append("Maintain current management practices")

    # Factor‑specific tweaks
    for factor in risk_factors:
        if factor == "do_min_night":
            if thresholds.get("do_night_min") is not None:
                actions.append(
                    f"Ensure night DO stays above {thresholds['do_night_min']} mg/L"
                )
        elif factor == "ammonia":
            if thresholds.get("ammonia_max") is not None:
                actions.append(
                    f"Keep ammonia concentration below {thresholds['ammonia_max']} mg/L"
                )
        elif factor == "nitrite":
            if thresholds.get("nitrite_max") is not None:
                actions.append(
                    f"Maintain nitrite below {thresholds['nitrite_max']} mg/L"
                )
        elif factor == "mortality":
            if thresholds.get("mortality_max") is not None:
                actions.append(
                    f"Investigate causes if mortality exceeds {thresholds['mortality_max']*100}%"
                )
        elif factor == "stress":
            if thresholds.get("stress_max") is not None:
                actions.append(
                    f"Implement stress‑reduction measures (e.g., temperature control)"
                )
    # De‑duplicate while preserving order
    seen = set()
    uniq_actions = []
    for a in actions:
        if a not in seen:
            uniq_actions.append(a)
            seen.add(a)
    return uniq_actions

# Example helper that could be called from the API
def get_mitigation_response(prediction: str, top_factors: List[str]) -> Dict:
    cfg = load_config()
    actions = recommend_actions(prediction, top_factors, cfg)
    return {"disease_risk": prediction, "recommended_actions": actions}

if __name__ == "__main__":
    # Simple demo
    cfg = load_config()
    demo = recommend_actions("High", ["do_min_night", "ammonia"], cfg)
    print(demo)
