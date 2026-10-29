"""
Recommendation Engine — Hybrid ML + Rule-Based

Takes a financial health prediction and outputs ONE concrete next action,
using the priority hierarchy established by FinScope 2024 recommendations
and EICV7 poverty-financial gap data.

Sources:
    FinScope 2024 (AFR/NISR) — Recommendations pp. 75-76.
    EICV7 Poverty Profile 2023/24 — Tables 6.6, 6.9, 6.11.
    NFIR 2026-2030 — FSDS targets.
"""


# Priority hierarchy: the highest-impact action first.
# Order follows FinScope 2024 recommendation priorities + severity of gap.
PRIORITY_RULES = [
    {
        "id": "insurance_gap",
        "trigger": lambda a: a["has_insurance"] == 0,
        "title": "Enroll in Mutuelle de Santé",
        "why": (
            "82% of Rwandans are covered by Mutuelle de Santé. Without it, a single "
            "health emergency can wipe out household savings. EICV7 shows poor households "
            "have a 12-point lower insurance coverage rate (77% vs 89%)."
        ),
        "action": "Visit your sector office or register online via Irembo.",
        "cost": "RWF 3,000/year (subsidized for vulnerable categories)",
        "source": "FinScope 2024, p.53; EICV7 Table 6.6",
    },
    {
        "id": "goal_no_action",
        "trigger": lambda a: a["has_goal"] == 1 and a["goal_action"] == "nothing",
        "title": "Open a goal-based savings account at your Umurenge SACCO",
        "why": (
            "You are in the 32% of Rwandans with a financial goal but no plan. "
            "FinScope 2024 shows the 'Achieving goals' score collapsed from 2.3 to 1.2 "
            "in 4 years. SACCOs are designed for exactly your profile."
        ),
        "action": "Visit your nearest Umurenge SACCO with your national ID.",
        "cost": "Free to open",
        "source": "FinScope 2024, p.68; EICV7 Table 6.9",
    },
    {
        "id": "no_savings_account",
        "trigger": lambda a: a["has_savings_account"] == 0,
        "title": "Open a savings account (SACCO, bank, or mobile money savings)",
        "why": (
            "Only 53% of poor households have a savings account vs 74% of non-poor. "
            "This is the single biggest financial inclusion gap in Rwanda today."
        ),
        "action": "Choose SACCO for rural areas, bank for urban, or activate mobile money savings.",
        "cost": "Free to open",
        "source": "EICV7 Table 6.9",
    },
    {
        "id": "no_pension",
        "trigger": lambda a: a["has_pension"] == 0 and a["age_group"] in ["25-34", "35-44", "45-54", "55+"],
        "title": "Register for Ejo Heza long-term savings",
        "why": (
            "Ejo Heza grew from 1% to 21% participation in 4 years — it's the easiest "
            "formal pension to join and available at any SACCO."
        ),
        "action": "Register at any SACCO or via USSD code *505#",
        "cost": "From RWF 100/month",
        "source": "FinScope 2024, p.54",
    },
    {
        "id": "no_bank_account",
        "trigger": lambda a: a["has_bank_account"] == 0 and a["area"] == "urban",
        "title": "Open a bank account to deepen your financial footprint",
        "why": (
            "You have mobile money but no bank. In urban areas, banks offer better "
            "savings rates and formal credit access."
        ),
        "action": "Visit any commercial bank with your national ID and proof of address.",
        "cost": "Free to open for most basic accounts",
        "source": "FinScope 2024, p.38",
    },
    {
        "id": "resilience_improve",
        "trigger": lambda a: a["can_raise"] != "yes" and a["has_savings_account"] == 1,
        "title": "Build an emergency fund of at least 1 month's expenses",
        "why": (
            "72% of Rwandans experienced a financial shock in the last year. "
            "Only 1% claimed insurance. An emergency fund is the strongest buffer."
        ),
        "action": "Set aside a fixed amount each week into your savings account.",
        "cost": "As little as RWF 500/week",
        "source": "FinScope 2024, p.52",
    },
    {
        "id": "maintain_and_deepen",
        "trigger": lambda a: True,  # fallback
        "title": "Maintain and deepen — consider formal credit for growth",
        "why": (
            "You are financially healthy. Next step: leverage formal credit to grow "
            "your business or assets. Only 24% of Rwandans use formal credit today."
        ),
        "action": "Speak to your SACCO or bank about a productive loan.",
        "cost": "Varies by product",
        "source": "FinScope 2024, p.47",
    },
]


def get_recommendation(answers, prediction):
    """
    answers: dict of questionnaire responses
    prediction: dict from predict_one() with score, segment
    Returns: dict with title, why, action, cost, source
    """
    for rule in PRIORITY_RULES:
        if rule["trigger"](answers):
            return {
                "id": rule["id"],
                "title": rule["title"],
                "why": rule["why"],
                "action": rule["action"],
                "cost": rule["cost"],
                "source": rule["source"],
                "segment": prediction["segment"],
                "score": prediction["score"],
            }

    # Should never reach here (fallback rule always matches)
    return {"id": "error", "title": "No recommendation available"}


if __name__ == "__main__":
    # Quick test
    sample_poor_farmer = {
        "gender": "female", "area": "rural", "age_group": "25-34",
        "education": "primary_or_below", "income_source": "independent_farmer",
        "has_mobile_money": 1, "has_bank_account": 0, "has_savings_account": 0,
        "has_insurance": 1, "has_pension": 0, "has_goal": 1,
        "goal_action": "nothing", "runs_out": "often",
        "can_raise": "no", "controls_finances": 0,
    }
    fake_prediction = {"score": 36.7, "segment": "vulnerable"}
    print("Recommendation for poor rural farmer:")
    print(get_recommendation(sample_poor_farmer, fake_prediction))