from sqlalchemy import func
from app import db, MenuItem, Order, OrderItem

def predict_menu_item_demand(restaurant_id, target_date):
    """
    Evaluates existing active menu items against actual historical OrderItem records
    and event category demand patterns.
    """
    target_dow = target_date.strftime("%w")
    day_name = target_date.strftime("%A")
    
    menu_items = MenuItem.query.filter_by(
        restaurant_id=restaurant_id,
        status='Active'
    ).all()
    
    if not menu_items:
        return []
        
    total_operating_days_res = db.session.query(
        func.count(func.distinct(func.date(Order.date)))
    ).filter(Order.date < target_date, Order.restaurant_id == restaurant_id).scalar() or 0
    
    if not total_operating_days_res:
        return []

    comp_operating_days_res = db.session.query(
        func.count(func.distinct(func.date(Order.date)))
    ).filter(
        Order.date < target_date, Order.restaurant_id == restaurant_id,
        func.strftime("%w", Order.date) == target_dow
    ).scalar() or 0
    
    menu_opportunities = []
    
    for item in menu_items:
        total_qty_sold = db.session.query(
            func.sum(OrderItem.quantity)
        ).join(Order).filter(
            Order.date < target_date, Order.restaurant_id == restaurant_id,
            OrderItem.item_name == item.name
        ).scalar() or 0
        
        comp_qty_sold = db.session.query(
            func.sum(OrderItem.quantity)
        ).join(Order).filter(
            Order.date < target_date, Order.restaurant_id == restaurant_id,
            OrderItem.item_name == item.name,
            func.strftime("%w", Order.date) == target_dow
        ).scalar() or 0
        
        normal_daily_avg = total_qty_sold / total_operating_days_res
        comp_daily_avg = comp_qty_sold / comp_operating_days_res if comp_operating_days_res >= 3 else normal_daily_avg
        diff_pct = 100 * (comp_daily_avg - normal_daily_avg) / normal_daily_avg if normal_daily_avg else 0

        if diff_pct >= 15.0:
            status = "HISTORICALLY_STRONG"
            evidence_level = "HIGH EVIDENCE" if comp_operating_days_res >= 3 else "RECOMMENDED PREP"
            explanation = f"Projected demand: {comp_daily_avg:.1f} units on {day_name} (+{diff_pct:.1f}% vs baseline). High holiday prep priority."
        elif diff_pct <= -15.0:
            status = "HISTORICALLY_WEAK"
            evidence_level = "LIMITED EVIDENCE"
            explanation = f"Projected demand: {comp_daily_avg:.1f} units ({diff_pct:.1f}% vs baseline)."
        else:
            status = "CONSISTENT_DEMAND"
            evidence_level = "STEADY DEMAND"
            explanation = f"Demand is steady ({comp_daily_avg:.1f} units expected)."
            
        menu_opportunities.append({
            "item_id": item.id,
            "name": item.name,
            "category": item.category or "General",
            "type": item.type or "Veg",
            "price": item.price,
            "normal_avg_qty": round(normal_daily_avg, 1),
            "comparable_avg_qty": round(comp_daily_avg, 1),
            "historical_diff_pct": round(diff_pct, 1),
            "comparable_days_count": comp_operating_days_res,
            "status": status,
            "evidence_level": evidence_level,
            "explanation": explanation
        })
        
    menu_opportunities.sort(key=lambda x: (x["status"] == "HISTORICALLY_STRONG", x["comparable_avg_qty"]), reverse=True)
    return menu_opportunities
