from datetime import date, timedelta
from sqlalchemy import func
from app import db, Order
from modules.holiday_ai.holiday_database import HolidayCalendar

def predict_footfall(restaurant_id, target_date, temperature=25, rainfall=0):
    """
    Predicts expected customer footfall for a restaurant on a target date.
    Uses historical order counts, weather features, and calendar events/festivals.
    """
    # 1. Calculate historical baseline daily customer count
    # Let's count unique orders per day in the last 30 days of activity
    recent_days = db.session.query(
        func.date(Order.date), 
        func.count(Order.id)
    ).filter(
        Order.restaurant_id == restaurant_id
    ).group_by(
        func.date(Order.date)
    ).all()
    
    if recent_days:
        # Average orders per active day
        avg_daily_orders = sum(cnt for _, cnt in recent_days) / len(recent_days)
    else:
        # Fallback default baseline orders
        avg_daily_orders = 120.0
        
    # Assume average party size (customers per order/bill) is 2.2
    base_footfall = avg_daily_orders * 2.2
    
    # 2. Check Calendar multipliers
    multiplier = 1.0
    holiday_name = None
    
    cal_entry = HolidayCalendar.query.filter_by(date=target_date).first()
    if cal_entry:
        holiday_name = cal_entry.holiday_name
        category = cal_entry.category
        h_type = cal_entry.type
        
        # Check if it's a long weekend
        # A long weekend is a Friday or Monday holiday
        is_long_weekend = False
        if h_type in ["Festival", "National Holiday", "Government Holiday"]:
            if cal_entry.day_name in ["Friday", "Monday"]:
                is_long_weekend = True
                
        # Apply multipliers
        if category == "National Holiday":
            multiplier = 1.40
        elif category == "Hindu Festival":
            multiplier = 1.50
        elif category == "Muslim Festival":
            multiplier = 1.45
        elif category == "Christian Festival":
            multiplier = 1.35
        elif category == "Weekend":
            if cal_entry.day_name == "Sunday":
                multiplier = 1.30
            else: # Saturday
                multiplier = 1.25
                
        if is_long_weekend:
            # Long weekend booster
            multiplier *= 1.15
            
    # 3. Apply Weather multipliers
    weather_multiplier = 1.0
    rain_val = float(rainfall) if rainfall is not None else 0.0
    temp_val = float(temperature) if temperature is not None else 25.0
    if rain_val > 50:
        weather_multiplier = 0.70  # Heavy rain decreases footfall by 30%
    elif rain_val > 10:
        weather_multiplier = 0.85  # Moderate rain decreases footfall by 15%
        
    if temp_val > 38:
        weather_multiplier *= 0.90  # Extreme heat decreases footfall by 10%
    elif temp_val < 12:
        weather_multiplier *= 0.90  # Extreme cold decreases footfall by 10%
        
    # 4. Final predicted customer footfall
    predicted_customers = int(base_footfall * multiplier * weather_multiplier)
    
    # Floor limit to prevent zero values
    predicted_customers = max(10, predicted_customers)
    
    return {
        "expected_customers": predicted_customers,
        "base_footfall": round(base_footfall, 1),
        "event_multiplier": round(multiplier, 2),
        "weather_multiplier": round(weather_multiplier, 2),
        "holiday_name": holiday_name
    }
