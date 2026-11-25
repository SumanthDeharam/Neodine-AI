from datetime import date, timedelta

def generate_predictive_alerts(target_date, predictions_payload):
    """
    Generates high-value, deduplicated, priority-ranked predictive alerts for restaurant management.
    """
    alerts = []
    seen_keys = set()
    event = predictions_payload.get("event", {})
    forecast = predictions_payload.get("forecast", {})
    menu_opps = predictions_payload.get("menu_opportunities", [])
    
    today = date.today()
    days_rem = (target_date - today).days
    target_date_str = target_date.strftime("%Y-%m-%d")
    
    # 1. Event / Festival / Long Weekend Alert
    if event.get("is_long_weekend"):
        key = ("LONG_WEEKEND", target_date_str)
        if key not in seen_keys:
            seen_keys.add(key)
            alerts.append({
                "alert_id": f"alt-lw-{target_date_str}",
                "alert_type": "LONG_WEEKEND",
                "priority": "HIGH" if days_rem <= 3 else "MEDIUM",
                "title": "Upcoming Long Weekend",
                "message": f"{event.get('long_weekend_desc')} approaching.",
                "event_date": target_date_str,
                "days_remaining": days_rem,
                "evidence": f"Extended weekend pattern detected for {target_date.strftime('%A, %d %B')}.",
                "source": "Holiday Calendar Intelligence"
            })
    elif event.get("category") in ["National Holiday", "Hindu Festival", "Muslim Festival", "Christian Festival", "Festival", "Celebration"]:
        key = ("FESTIVAL", target_date_str)
        if key not in seen_keys:
            seen_keys.add(key)
            alerts.append({
                "alert_id": f"alt-fest-{target_date_str}",
                "alert_type": "FESTIVAL",
                "priority": "HIGH" if days_rem <= 3 else "MEDIUM",
                "title": f"Festival Alert: {event.get('event_name')}",
                "message": f"{event.get('event_name')} in {days_rem} days.",
                "event_date": target_date_str,
                "days_remaining": days_rem,
                "evidence": "Authoritative Indian Holiday Calendar Engine",
                "source": "Holiday Calendar Intelligence"
            })
    elif event.get("is_weekend"):
        key = ("UPCOMING_WEEKEND", target_date_str)
        if key not in seen_keys:
            seen_keys.add(key)
            alerts.append({
                "alert_id": f"alt-wknd-{target_date_str}",
                "alert_type": "UPCOMING_WEEKEND",
                "priority": "MEDIUM" if days_rem <= 2 else "INFO",
                "title": f"{event.get('day_name')} Service",
                "message": f"Review the restaurant forecast when planning weekend service.",
                "event_date": target_date_str,
                "days_remaining": days_rem,
                "evidence": f"Weekend pattern for {event.get('day_name')}.",
                "source": "Weekend Analytics Engine"
            })
            
    # 2. Demand Outlook & Staffing Alert
    outlook = forecast.get("demand_outlook")
    pct_change = float(forecast.get("percentage_change") or 0.0)
    
    if outlook == "HIGHER THAN NORMAL" or pct_change > 10:
        key = ("STAFFING_DEMAND", target_date_str)
        if key not in seen_keys:
            seen_keys.add(key)
            alerts.append({
                "alert_id": f"alt-staff-{target_date_str}",
                "alert_type": "STAFFING",
                "priority": "HIGH",
                "title": "Kitchen & Service Staffing Recommendation",
                "message": f"Projected {forecast.get('expected_orders_min')}–{forecast.get('expected_orders_max')} orders. Ensure additional front-of-house staff are rostered.",
                "event_date": target_date_str,
                "days_remaining": days_rem,
                "evidence": f"+{pct_change}% projected surge over baseline.",
                "source": "Predictive Operations AI"
            })
            
    # 3. Top Menu Item Preparation Alert
    high_prep_items = [m["name"] for m in menu_opps if m.get("status") == "HISTORICALLY_STRONG"][:3]
    if high_prep_items:
        key = ("MENU_PREP", target_date_str)
        if key not in seen_keys:
            seen_keys.add(key)
            alerts.append({
                "alert_id": f"alt-prep-{target_date_str}",
                "alert_type": "KITCHEN_PREP",
                "priority": "MEDIUM",
                "title": "Priority Kitchen Prep Items",
                "message": f"Pre-prep key ingredients for {', '.join(high_prep_items)} by 11:30 AM to prevent order delays.",
                "event_date": target_date_str,
                "days_remaining": days_rem,
                "evidence": "Historical top-seller holiday surge rankings.",
                "source": "Menu Intelligence Engine"
            })
            
    return alerts
