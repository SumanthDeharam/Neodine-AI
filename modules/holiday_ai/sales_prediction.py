def predict_sales_revenue(restaurant_id, footfall_res):
    """
    Translates footfall results into traceable revenue range forecasts.
    Derives values cleanly from footfall_res order projections and average ticket size.
    """
    if footfall_res.get("data_state") == "INSUFFICIENT_DATA":
        return {
            "expected_revenue_min": None, "expected_revenue_max": None,
            "comparable_average_revenue": 0, "historical_baseline_revenue": 0,
            "explanation": footfall_res["explanation"],
        }
    min_rev = footfall_res.get("expected_revenue_min") or 0.0
    max_rev = footfall_res.get("expected_revenue_max") or 0.0
    base_rev = footfall_res.get("historical_baseline_revenue", 0.0)
    comp_avg_rev = footfall_res.get("comparable_average_revenue", 0.0)
    
    return {
        "expected_revenue_min": min_rev,
        "expected_revenue_max": max_rev,
        "comparable_average_revenue": comp_avg_rev,
        "historical_baseline_revenue": base_rev,
        "explanation": f"Projected revenue range: ₹{min_rev:,.0f} to ₹{max_rev:,.0f} (Historical average: ₹{comp_avg_rev:,.0f})."
    }
