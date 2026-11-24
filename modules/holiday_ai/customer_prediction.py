from datetime import date, datetime, timedelta
from sqlalchemy import func
from app import db, Order
from modules.holiday_ai.holiday_engine import get_event_for_date, get_raw_holiday_info

def predict_customer_footfall(restaurant_id, target_date, temperature=None, rainfall=None):
    """
    Empirical, evidence-based customer & demand forecasting engine.
    Calculates historical baselines from restaurant order records with intelligent
    event & surge modeling.
    Returns: expected_orders_min, expected_orders_max, expected_revenue_min,
             expected_revenue_max, demand_outlook, percentage_change, etc.
    """
    event_info = get_event_for_date(target_date)
    day_name = target_date.strftime("%A")
    
    # 1. Fetch total active operating days and overall baseline for this restaurant
    all_daily_stats = db.session.query(
        func.date(Order.date).label("order_day"),
        func.count(Order.id).label("order_count"),
        func.sum(Order.total_price).label("daily_revenue")
    ).filter(
        Order.restaurant_id == restaurant_id, Order.date < target_date
    ).group_by(
        func.date(Order.date)
    ).order_by(func.date(Order.date)).all()
    
    baseline_days_count = len(all_daily_stats)
    if not baseline_days_count:
        message = "No order history is available for this restaurant."
        return {
            "event": event_info, "data_state": "INSUFFICIENT_DATA",
            "evidence_level": "INSUFFICIENT DATA", "demand_outlook": "INSUFFICIENT DATA",
            "message": message, "explanation": message, "historical_examples": [],
            "expected_orders_min": None, "expected_orders_max": None,
            "expected_revenue_min": None, "expected_revenue_max": None,
            "historical_baseline_orders": 0, "historical_baseline_revenue": 0,
            "comparable_average_orders": 0, "comparable_average_revenue": 0,
            "comparable_days_count": 0, "baseline_days_count": 0, "percentage_change": 0,
        }
    overall_avg_orders = sum(row.order_count for row in all_daily_stats) / baseline_days_count
    overall_avg_revenue = sum(row.daily_revenue or 0 for row in all_daily_stats) / baseline_days_count

    # Back off to weekday or operating-day history when this event has too few examples.
    from statistics import median
    weekday_days = []
    event_days = []
    for row in all_daily_stats:
        day = date.fromisoformat(str(row.order_day))
        if day.weekday() == target_date.weekday():
            weekday_days.append(row)
        if event_info.get("category") not in ("Weekend", "Normal"):
            if get_raw_holiday_info(day)[0] == event_info.get("event_name"):
                event_days.append(row)
    comparable = event_days if len(event_days) >= 3 else weekday_days
    sample = comparable if len(comparable) >= 3 else all_daily_stats
    comparable_days_count = len(comparable)
    date_examples = [str(row.order_day) for row in sample[-5:]]
    counts = [row.order_count for row in sample]
    revenues = [float(row.daily_revenue or 0) for row in sample]
    projected_orders = median(counts) if len(comparable) >= 3 else counts[-1]
    projected_revenue = median(revenues) if len(comparable) >= 3 else revenues[-1]
    min_orders, max_orders = min(counts), max(counts)
    min_revenue, max_revenue = round(min(revenues), 2), round(max(revenues), 2)

    # Statistical Percentage Change from Overall Baseline
    pct_change = 0.0
    if overall_avg_orders > 0:
        pct_change = ((projected_orders - overall_avg_orders) / overall_avg_orders) * 100.0
        
    if pct_change >= 10.0:
        demand_outlook = "HIGHER THAN NORMAL"
    elif pct_change <= -10.0:
        demand_outlook = "LOWER THAN NORMAL"
    else:
        demand_outlook = "NORMAL"
        
    if comparable_days_count >= 5:
        data_state = "REAL_DATA"
        evidence_level = "HIGH EVIDENCE"
    else:
        data_state = "LIMITED_DATA"
        evidence_level = "LIMITED EVIDENCE"

    direction_str = "higher" if pct_change >= 0 else "lower"
    event_desc = f"{event_info.get('event_name')}"
    explanation = (
        f"For {day_name} ({event_desc}), projected demand is {min_orders}–{max_orders} orders "
        f"(typical volume {projected_orders:g}, {abs(pct_change):.1f}% {direction_str} than overall baseline of {overall_avg_orders:.1f} orders/day). "
        f"Estimated revenue: ₹{min_revenue:,.0f} to ₹{max_revenue:,.0f}."
    )
    
    return {
        "data_state": data_state,
        "evidence_level": evidence_level,
        "message": explanation,
        "event": event_info,
        "demand_outlook": demand_outlook,
        "expected_orders": projected_orders,
        "expected_revenue": projected_revenue,
        "expected_orders_min": min_orders,
        "expected_orders_max": max_orders,
        "expected_revenue_min": min_revenue,
        "expected_revenue_max": max_revenue,
        "historical_baseline_orders": round(overall_avg_orders, 1),
        "historical_baseline_revenue": round(overall_avg_revenue, 2),
        "comparable_average_orders": round(sum(r.order_count for r in comparable) / comparable_days_count, 1) if comparable else 0,
        "comparable_average_revenue": round(sum((r.daily_revenue or 0) for r in comparable) / comparable_days_count, 2) if comparable else 0,
        "comparable_days_count": comparable_days_count,
        "baseline_days_count": baseline_days_count,
        "percentage_change": round(pct_change, 1),
        "explanation": explanation,
        "historical_examples": date_examples
    }
