def compile_prediction_explanation(footfall_res):
    """
    Compiles traceable evidence explanations for the UI "Why?" panel.
    Returns what was observed, why it matters, sample sizes, and evidence levels.
    NO hardcoded donut weights or artificial percentage factors.
    """
    data_state = footfall_res.get("data_state")
    if data_state == "INSUFFICIENT_DATA":
        return {
            "what_was_observed": "Zero comparable historical sales records found.",
            "why_it_matters": "Predictions cannot responsibly be generated without historical evidence.",
            "evidence_level": "INSUFFICIENT DATA",
            "comparable_days_count": 0,
            "baseline_days_count": footfall_res.get("baseline_days_count", 0),
            "explanation": footfall_res.get("explanation", "Insufficient data.")
        }
        
    comp_count = footfall_res.get("comparable_days_count", 0)
    base_count = footfall_res.get("baseline_days_count", 0)
    pct_change = footfall_res.get("percentage_change", 0.0)
    comp_avg = footfall_res.get("comparable_average_orders", 0.0)
    base_avg = footfall_res.get("historical_baseline_orders", 0.0)
    day_name = footfall_res.get("event", {}).get("day_name", "Day")
    
    what_observed = f"{comp_count} comparable {day_name} operating days averaged {comp_avg:.1f} orders/day."
    direction = "above" if pct_change >= 0 else "below"
    why_matters = f"Your normal daily operating average across {base_count} days is {base_avg:.1f} orders/day, so comparable days historically ran {abs(pct_change):.1f}% {direction} baseline."
    
    return {
        "what_was_observed": what_observed,
        "why_it_matters": why_matters,
        "evidence_level": footfall_res.get("evidence_level", "LIMITED EVIDENCE"),
        "comparable_days_count": comp_count,
        "baseline_days_count": base_count,
        "explanation": footfall_res.get("explanation", "")
    }
