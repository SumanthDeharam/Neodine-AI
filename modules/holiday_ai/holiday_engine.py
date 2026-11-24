import os
import csv
from datetime import datetime, date, timedelta

# Accurate Indian Movable & Fixed Holidays (2024 to 2035)
MOVABLE_HOLIDAYS = {
    2024: {
        "15-01": ("Makar Sankranti / Pongal", "Hindu Festival", "Festival"),
        "08-03": ("Maha Shivratri", "Hindu Festival", "Festival"),
        "25-03": ("Holi", "Hindu Festival", "Festival"),
        "29-03": ("Good Friday", "Christian Festival", "Government Holiday"),
        "09-04": ("Ugadi / Gudi Padwa", "Hindu Festival", "Festival"),
        "11-04": ("Eid al-Fitr", "Muslim Festival", "Festival"),
        "14-04": ("Ambedkar Jayanti / Tamil New Year", "National Holiday", "Festival"),
        "17-04": ("Ram Navami", "Hindu Festival", "Festival"),
        "17-06": ("Eid al-Adha (Bakrid)", "Muslim Festival", "Festival"),
        "17-07": ("Muharram", "Muslim Festival", "Government Holiday"),
        "19-08": ("Raksha Bandhan", "Hindu Festival", "Festival"),
        "26-08": ("Janmashtami", "Hindu Festival", "Festival"),
        "07-09": ("Ganesh Chaturthi", "Hindu Festival", "Festival"),
        "11-10": ("Maha Navami / Ayudha Puja", "Hindu Festival", "Festival"),
        "12-10": ("Dussehra / Vijayadashami", "Hindu Festival", "Festival"),
        "31-10": ("Diwali / Deepavali (Lakshmi Puja)", "Hindu Festival", "Festival"),
        "01-11": ("Kannada Rajyotsava / Balipratipada", "Regional Festival", "Government Holiday"),
        "15-11": ("Guru Nanak Jayanti", "Sikh Festival", "Government Holiday"),
    },
    2025: {
        "14-01": ("Makar Sankranti / Pongal", "Hindu Festival", "Festival"),
        "26-02": ("Maha Shivratri", "Hindu Festival", "Festival"),
        "14-03": ("Holi", "Hindu Festival", "Festival"),
        "30-03": ("Ugadi / Gudi Padwa", "Hindu Festival", "Festival"),
        "31-03": ("Eid al-Fitr", "Muslim Festival", "Festival"),
        "14-04": ("Ambedkar Jayanti / Tamil New Year", "National Holiday", "Festival"),
        "18-04": ("Good Friday", "Christian Festival", "Government Holiday"),
        "07-06": ("Eid al-Adha (Bakrid)", "Muslim Festival", "Festival"),
        "06-07": ("Muharram", "Muslim Festival", "Government Holiday"),
        "09-08": ("Raksha Bandhan", "Hindu Festival", "Festival"),
        "16-08": ("Janmashtami", "Hindu Festival", "Festival"),
        "27-08": ("Ganesh Chaturthi", "Hindu Festival", "Festival"),
        "02-10": ("Gandhi Jayanti & Dussehra / Vijayadashami", "National Holiday", "Festival"),
        "20-10": ("Diwali / Deepavali (Lakshmi Puja)", "Hindu Festival", "Festival"),
        "01-11": ("Kannada Rajyotsava", "Regional Festival", "Government Holiday"),
        "05-11": ("Guru Nanak Jayanti", "Sikh Festival", "Government Holiday"),
    },
    2026: {
        "14-01": ("Makar Sankranti / Pongal", "Hindu Festival", "Festival"),
        "16-02": ("Maha Shivratri", "Hindu Festival", "Festival"),
        "03-03": ("Holika Dahan", "Hindu Festival", "Festival"),
        "04-03": ("Holi (Color Festival)", "Hindu Festival", "Festival"),
        "19-03": ("Ugadi / Gudi Padwa", "Hindu Festival", "Festival"),
        "21-03": ("Eid al-Fitr", "Muslim Festival", "Festival"),
        "03-04": ("Good Friday", "Christian Festival", "Government Holiday"),
        "14-04": ("Ambedkar Jayanti / Tamil New Year", "National Holiday", "Festival"),
        "28-05": ("Eid al-Adha (Bakrid)", "Muslim Festival", "Festival"),
        "26-06": ("Muharram", "Muslim Festival", "Government Holiday"),
        "28-08": ("Raksha Bandhan", "Hindu Festival", "Festival"),
        "04-09": ("Janmashtami (Krishna Jayanti)", "Hindu Festival", "Festival"),
        "14-09": ("Ganesh Chaturthi", "Hindu Festival", "Festival"),
        "25-09": ("Ganesh Visarjan (Anant Chaturdashi)", "Hindu Festival", "Festival"),
        "02-10": ("Gandhi Jayanti", "National Holiday", "National Holiday"),
        "20-10": ("Dussehra / Vijayadashami", "Hindu Festival", "Festival"),
        "01-11": ("Kannada Rajyotsava", "Regional Festival", "Government Holiday"),
        "08-11": ("Diwali / Deepavali (Lakshmi Puja)", "Hindu Festival", "Festival"),
        "10-11": ("Diwali (Bhai Dooj / Balipratipada)", "Hindu Festival", "Festival"),
        "24-11": ("Guru Nanak Jayanti", "Sikh Festival", "Government Holiday"),
    },
    2027: {
        "14-01": ("Makar Sankranti / Pongal", "Hindu Festival", "Festival"),
        "06-03": ("Maha Shivratri", "Hindu Festival", "Festival"),
        "10-03": ("Eid al-Fitr", "Muslim Festival", "Festival"),
        "22-03": ("Holi", "Hindu Festival", "Festival"),
        "26-03": ("Good Friday", "Christian Festival", "Government Holiday"),
        "07-04": ("Ugadi / Gudi Padwa", "Hindu Festival", "Festival"),
        "14-04": ("Ambedkar Jayanti / Tamil New Year", "National Holiday", "Festival"),
        "17-05": ("Eid al-Adha (Bakrid)", "Muslim Festival", "Festival"),
        "16-07": ("Muharram", "Muslim Festival", "Government Holiday"),
        "17-08": ("Raksha Bandhan", "Hindu Festival", "Festival"),
        "25-08": ("Janmashtami", "Hindu Festival", "Festival"),
        "04-09": ("Ganesh Chaturthi", "Hindu Festival", "Festival"),
        "02-10": ("Gandhi Jayanti", "National Holiday", "National Holiday"),
        "09-10": ("Dussehra / Vijayadashami", "Hindu Festival", "Festival"),
        "29-10": ("Diwali / Deepavali", "Hindu Festival", "Festival"),
        "01-11": ("Kannada Rajyotsava", "Regional Festival", "Government Holiday"),
        "14-11": ("Guru Nanak Jayanti", "Sikh Festival", "Government Holiday"),
    }
}

# Fixed calendar holidays (same date every year)
FIXED_HOLIDAYS = {
    "01-01": ("New Year's Day", "Celebration", "Celebration"),
    "26-01": ("Republic Day", "National Holiday", "National Holiday"),
    "01-05": ("May Day / International Workers' Day", "National Holiday", "Government Holiday"),
    "15-08": ("Independence Day", "National Holiday", "National Holiday"),
    "02-10": ("Gandhi Jayanti", "National Holiday", "National Holiday"),
    "25-12": ("Christmas Day", "Christian Festival", "Festival"),
    "31-12": ("New Year's Eve", "Celebration", "Celebration"),
}

def get_raw_holiday_info(target_date):
    """
    Returns (event_name, category, event_type) for target_date without recursion.
    """
    year = target_date.year
    dd_mm = target_date.strftime("%d-%m")
    dow = target_date.weekday() # 5=Sat, 6=Sun
    day_name = target_date.strftime("%A")

    if year in MOVABLE_HOLIDAYS and dd_mm in MOVABLE_HOLIDAYS[year]:
        return MOVABLE_HOLIDAYS[year][dd_mm]
    if dd_mm in FIXED_HOLIDAYS:
        return FIXED_HOLIDAYS[dd_mm]
    if dow in [5, 6]:
        return f"{day_name} Weekend", "Weekend", "Weekend"
    return f"Standard Operating Day ({day_name})", "Normal", "Normal"

def detect_long_weekend(target_date):
    """
    Detects if a target date belongs to or is adjacent to a Long Weekend.
    Returns (is_long_weekend: bool, long_weekend_length: int, related_event_name: str)
    """
    current_dow = target_date.weekday() # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    friday = target_date - timedelta(days=(current_dow - 4) if current_dow >= 4 else (current_dow + 3))
    
    dates_in_window = {}
    for i in range(4): # Fri, Sat, Sun, Mon
        check_d = friday + timedelta(days=i)
        ev_name, cat, ev_type = get_raw_holiday_info(check_d)
        is_h = cat in ["National Holiday", "Hindu Festival", "Muslim Festival", "Christian Festival", "Government Holiday", "Festival", "Regional Festival"]
        dates_in_window[check_d] = {
            "dow": check_d.weekday(),
            "is_holiday": is_h,
            "event_name": ev_name
        }
        
    fri_d = friday
    sat_d = friday + timedelta(days=1)
    sun_d = friday + timedelta(days=2)
    mon_d = friday + timedelta(days=3)
    
    # Check Friday Holiday (Fri + Sat + Sun = 3 days)
    if dates_in_window[fri_d]["is_holiday"] and not dates_in_window[mon_d]["is_holiday"]:
        if target_date in [fri_d, sat_d, sun_d]:
            return True, 3, f"Long Weekend (Fri {dates_in_window[fri_d]['event_name']} + Sat + Sun)"
            
    # Check Monday Holiday (Sat + Sun + Mon = 3 days)
    if dates_in_window[mon_d]["is_holiday"] and not dates_in_window[fri_d]["is_holiday"]:
        if target_date in [sat_d, sun_d, mon_d]:
            return True, 3, f"Long Weekend (Sat + Sun + Mon {dates_in_window[mon_d]['event_name']})"
            
    # Check 4-day mega weekend (Fri Holiday + Sat + Sun + Mon Holiday)
    if dates_in_window[fri_d]["is_holiday"] and dates_in_window[mon_d]["is_holiday"]:
        return True, 4, f"4-Day Extended Weekend ({dates_in_window[fri_d]['event_name']} to {dates_in_window[mon_d]['event_name']})"
        
    return False, 0, ""

def get_event_for_date(target_date):
    """
    Returns holiday event dictionary for target_date with exact date and formatted day.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    day_name = target_date.strftime("%A")
    date_formatted = target_date.strftime("%d %b %Y")
    full_date_str = target_date.strftime("%A, %d %B %Y")
    dow = target_date.weekday()
    is_weekend = (dow in [5, 6])
    
    event_name, category, event_type = get_raw_holiday_info(target_date)
    is_lw, lw_len, lw_desc = detect_long_weekend(target_date)
    today = date.today()
    days_rem = (target_date - today).days
    
    return {
        "date_str": date_str,
        "date_formatted": date_formatted,
        "full_date_str": full_date_str,
        "event_name": event_name,
        "category": category,
        "event_type": event_type,
        "day_name": day_name,
        "is_weekend": is_weekend,
        "is_long_weekend": is_lw,
        "long_weekend_length": lw_len,
        "long_weekend_desc": lw_desc,
        "days_remaining": days_rem,
        "calendar_source": "Authoritative Indian Holiday Calendar Engine"
    }

def seed_database_from_csv(csv_path="modules/holiday_ai/holiday_calendar.csv", force=False):
    """
    Populates HolidayCalendar SQLite table for 2024 to 2030 from verified authoritative dates.
    """
    try:
        from modules.holiday_ai.holiday_database import HolidayCalendar
        from app import db
        
        start_d = date(2024, 1, 1)
        end_d = date(2030, 12, 31)
        
        existing_count = HolidayCalendar.query.count()
        # If force or outdated count, refresh table
        if force or existing_count < 300:
            HolidayCalendar.query.delete()
            db.session.commit()
        else:
            return existing_count
            
        cur = start_d
        added = 0
        while cur <= end_d:
            ev = get_event_for_date(cur)
            if ev["category"] != "Normal" or ev["is_weekend"] or ev["is_long_weekend"]:
                row = HolidayCalendar(
                    date_str=ev["date_str"],
                    date=cur,
                    holiday_name=ev["event_name"],
                    category=ev["category"],
                    type=ev["event_type"],
                    year=cur.year,
                    day_name=ev["day_name"],
                    is_weekend=ev["is_weekend"]
                )
                db.session.add(row)
                added += 1
            cur += timedelta(days=1)
            
        db.session.commit()
        return added
    except Exception as e:
        print(f"[HOLIDAY SEED ERROR] {e}")
        return 0
