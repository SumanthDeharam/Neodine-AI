from app import db
from datetime import datetime, timezone

class HolidayCalendar(db.Model):
    __tablename__ = 'holiday_calendar'
    id = db.Column(db.Integer, primary_key=True)
    date_str = db.Column(db.String(20), nullable=False)  # format: "DD-MM-YYYY"
    date = db.Column(db.Date, unique=True, nullable=False, index=True)
    holiday_name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # National Holiday, Hindu Festival, Muslim Festival, Christian Festival, Weekend, etc.
    type = db.Column(db.String(50), nullable=False)      # Festival, National Holiday, Weekend, Government Holiday
    day_name = db.Column(db.String(20), nullable=False)  # Saturday, Sunday, Monday, etc.
    is_weekend = db.Column(db.Boolean, default=False)
    year = db.Column(db.Integer, nullable=False)

class HolidayPredictionHistory(db.Model):
    __tablename__ = 'holiday_prediction_history'
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    holiday_name = db.Column(db.String(100), nullable=True)
    expected_customers = db.Column(db.Integer, nullable=False)
    expected_revenue = db.Column(db.Float, nullable=False)
    expected_bills = db.Column(db.Integer, nullable=False)
    expected_aov = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
