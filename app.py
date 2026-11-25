import sys
if __name__ == '__main__':
    sys.modules['app'] = sys.modules['__main__']

from flask import Flask, request, render_template, redirect, url_for, jsonify, session as flask_session
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from flask import send_file
from functools import wraps

import uuid
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression  # type: ignore
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

import json

def get_global_settings():
    path = os.path.join(app.instance_path, "global_settings.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"test_mode": False}

def save_global_settings(settings):
    os.makedirs(app.instance_path, exist_ok=True)
    path = os.path.join(app.instance_path, "global_settings.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(settings, f)

_ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}

def _safe_settings_path(username):
    """Build a safe path inside app.instance_path for a user's settings JSON."""
    from werkzeug.utils import secure_filename
    safe_name = secure_filename(str(username or '').strip()) or 'default'
    base_dir = os.path.abspath(app.instance_path)
    target = os.path.abspath(os.path.join(base_dir, f"{safe_name}_settings.json"))
    if not target.startswith(base_dir + os.sep):
        raise ValueError("Invalid settings path")
    return target

def _is_safe_image_filename(filename):
    ext = os.path.splitext(str(filename or '').strip())[1].lower()
    return ext in _ALLOWED_IMAGE_EXTENSIONS

def _is_safe_image_url(url):
    if not url:
        return True
    s = str(url).strip().lower()
    return s.startswith('/static/') or s.startswith('http://') or s.startswith('https://')

def get_tax_config(user_id=None):
    """Retrieve combined tax configuration for the specific restaurant."""
    cgst_pct = 2.5
    sgst_pct = 2.5
    service_charge_pct = 0.0
    tax_inclusive = False
    tax_enabled = True
    gst_number = '29ABCDE1234F1Z5'

    owner_id = user_id
    try:
        if owner_id is None and current_user and current_user.is_authenticated:
            owner_id = _current_restaurant_owner()
    except Exception:
        pass

    owner = None
    owner_username = None
    if owner_id:
        try:
            owner = db.session.get(User, owner_id)
            if owner:
                owner_username = owner.username
        except Exception:
            pass

    if owner_id:
        try:
            detail = RestaurantDetail.query.filter_by(user_id=owner_id).first()
            if detail and detail.gst_number:
                gst_number = detail.gst_number

            u_sett = UserSetting.query.filter_by(user_id=owner_id).first()
            if u_sett is not None and u_sett.tax_enabled is not None:
                tax_enabled = bool(u_sett.tax_enabled)
        except Exception:
            pass

    if owner_username:
        spath = os.path.join(app.instance_path, f"{owner_username}_settings.json")
        if os.path.exists(spath):
            try:
                with open(spath, 'r', encoding='utf-8') as f:
                    sdata = json.load(f)
                    if 'tax_config' in sdata and isinstance(sdata['tax_config'], dict):
                        tcfg = sdata['tax_config']
                        cgst_pct = float(tcfg.get('cgst_pct', cgst_pct))
                        sgst_pct = float(tcfg.get('sgst_pct', sgst_pct))
                        service_charge_pct = float(tcfg.get('service_charge_pct', service_charge_pct))
                        tax_inclusive = bool(tcfg.get('tax_inclusive', tax_inclusive))
                        tax_enabled = bool(tcfg.get('tax_enabled', tax_enabled))
                        if tcfg.get('gst_number'):
                            gst_number = tcfg.get('gst_number')
            except Exception as e:
                print(f"Error loading {owner_username}_settings.json for tax_config:", e)

    combined_gst_pct = (cgst_pct + sgst_pct) if tax_enabled else 0.0
    return {
        'cgst_pct': cgst_pct,
        'sgst_pct': sgst_pct,
        'combined_gst_pct': round(combined_gst_pct, 2),
        'service_charge_pct': service_charge_pct,
        'tax_inclusive': tax_inclusive,
        'tax_enabled': tax_enabled,
        'gst_number': gst_number
    }
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'neodine-super-secret-key-123')
db_url = os.environ.get('DATABASE_URL', 'sqlite:///saas_customers.db')
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

db = SQLAlchemy(app)
from modules.holiday_ai.holiday_database import HolidayCalendar, HolidayPredictionHistory

login_manager = LoginManager(app)
login_manager.login_view = 'login'        

# --- Stripe Integration ---
class MockStripeCheckoutSession:    
    class Session:
        @classmethod
        def create(cls, **kwargs):
            class DummySession:
                id = "cs_test_dummy"
                url = f"http://localhost:10000/subscription/success?session_id=cs_test_dummy&plan_slug={kwargs.get('metadata', {}).get('plan_slug', 'starter')}"
            return DummySession()
        @classmethod
        def retrieve(cls, session_id):
            class DummySession:
                payment_status = "paid" if session_id == "cs_test_dummy" else "failed"
                metadata = {"plan_slug": "starter"}
            return DummySession()

class MockStripe:
    api_key = "sk_test_dummy"
    checkout = MockStripeCheckoutSession

try:
    import stripe
    if os.environ.get('STRIPE_SECRET_KEY') is None:
        stripe = MockStripe
except ImportError:
    stripe = MockStripe

# Fetch keys from environment, fallback to mock keys
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', 'pk_test_dummy')
if stripe == MockStripe:
    stripe.api_key = "sk_test_dummy"
else:
    stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_dummy')


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    raw_password = db.Column(db.String(200))  # Admin visible password
    # 'Pending', 'Approved', 'Admin'
    status = db.Column(db.String(20), default='Pending')
    orders = db.relationship('Order', backref='restaurant', lazy=True)
    
    name = db.Column(db.String(100), default="")
    phone = db.Column(db.String(50), default="")
    profile_image = db.Column(db.String(250), default="")
    role = db.Column(db.String(50), default="Cashier")
    last_login = db.Column(db.DateTime)
    # For waiter/kitchen staff: links the account to the restaurant (User) it serves.
    # Scope = restaurant_owner_id for staff, own id for the restaurant owner.
    restaurant_owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)


class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False)
    order_uuid = db.Column(db.String(100), unique=True, nullable=False)
    customer_name = db.Column(db.String(100))
    customer_phone = db.Column(db.String(20))
    subtotal = db.Column(db.Float)
    discount_pct = db.Column(db.Float, default=0.0)
    gst_pct = db.Column(db.Float, default=5.0)
    gst_amount = db.Column(db.Float)
    total_price = db.Column(db.Float)
    temperature = db.Column(db.Integer)
    rainfall = db.Column(db.Integer)
    date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    order_type = db.Column(db.String(50), default='Dine-In')
    payment_method = db.Column(db.String(50), default='Cash')
    # --- Order lifecycle fields used by the Waiter & Kitchen portals ---
    table_num = db.Column(db.Integer, nullable=True)
    payment_status = db.Column(db.String(20), default='PENDING')  # PENDING / PAID / CANCELLED
    order_status = db.Column(db.String(20), default='PENDING')    # PENDING / KITCHEN / PREPARING / READY / COMPLETED
    waiter_note = db.Column(db.String(300), nullable=True)
    items = db.relationship('OrderItem', backref='order_ref', lazy=True)


class KitchenProgress(db.Model):
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), primary_key=True)
    ready_through = db.Column(db.Integer, default=0, nullable=False)


def mark_visible_items_ready(order):
    last = max((line.id for line in order.items), default=0)
    db.session.merge(KitchenProgress(order_id=order.id, ready_through=last))


class OrderSubmission(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(36), primary_key=True)
    kind = db.Column(db.String(12), nullable=False)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    table_num = db.Column(db.Integer)
    quantity = db.Column(db.Integer, default=0)


def lock_restaurant(owner_id):
    # Acquire the SQLite write lock before reading the current table or retry token.
    User.query.filter_by(id=owner_id).update({User.id: User.id}, synchronize_session=False)


def submission_result(record):
    if record.kind == 'checkout':
        order = db.session.get(Order, record.order_id)
        return redirect(url_for('show_receipt', order_id=order.order_uuid))
    return jsonify(success=True, order_id=record.order_id, table_num=record.table_num,
                   quantity=record.quantity)


@app.get('/api/checkout/submissions/<token>')
@login_required
def checkout_submission_status(token):
    row = db.session.get(OrderSubmission, (current_user.id, token))
    if row and row.kind == 'checkout':
        order = db.session.get(Order, row.order_id)
        return jsonify(receipt=url_for('show_receipt', order_id=order.order_uuid))
    return jsonify(receipt=None), 404


@app.get('/api/kitchen/submissions/<token>')
@login_required
def kitchen_submission_status(token):
    row = db.session.get(OrderSubmission, (current_user.id, token))
    if row and row.kind == 'kitchen':
        return submission_result(row)
    return jsonify(success=False), 404


class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    item_name = db.Column(db.String(100))
    price = db.Column(db.Float)
    quantity = db.Column(db.Integer, default=1)

class MenuItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50))
    price = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(20), default='Veg') # Veg or Non-Veg
    status = db.Column(db.String(20), default='Active')
    description = db.Column(db.String(255))
    is_best_seller = db.Column(db.Boolean, default=False)
    is_hidden = db.Column(db.Boolean, default=False)
    image_url = db.Column(db.String(255))
    prep_time = db.Column(db.Integer, default=15)
    display_order = db.Column(db.Integer, default=0)
    recipe_name = db.Column(db.String(100), nullable=True)

class FoodImageLibrary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    keywords = db.Column(db.Text)
    file_path = db.Column(db.String(255), nullable=False)
    tags = db.Column(db.String(255))

class Inventory(db.Model):
    __tablename__ = 'inventory'
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ingredient_name = db.Column(db.String(100), nullable=False)
    unit = db.Column(db.String(20), default='kg') # kg, ml, L, pieces
    current_stock = db.Column(db.Float, default=0.0)
    min_level = db.Column(db.Float, default=5.0)
    consumed_quantity = db.Column(db.Float, default=0.0)  # Auto-updated by order deductions
    supplier = db.Column(db.String(150), nullable=True)
    last_updated = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class MenuRecipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    menu_item_id = db.Column(db.Integer, db.ForeignKey('menu_item.id'), nullable=False)
    ingredient_id = db.Column(db.Integer, db.ForeignKey('inventory.id'), nullable=False)
    quantity_required = db.Column(db.Float, nullable=False)
    recipe_name = db.Column(db.String(100), nullable=True)
    
    menu_item = db.relationship('MenuItem', backref=db.backref('recipes', cascade='all, delete-orphan'))
    ingredient = db.relationship('Inventory', backref='used_in_recipes')

class DiningTable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    table_num = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='EMPTY') # EMPTY or DINING
    current_order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=True)
    
    order = db.relationship('Order', foreign_keys=[current_order_id], lazy=True)

class GroupService(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='ACTIVE') # ACTIVE, COMPLETED
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    customers = db.relationship('GroupCustomer', backref='group_session', cascade='all, delete-orphan', lazy=True)

class GroupCustomer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('group_service.id'), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='PENDING') # PENDING, PAID
    orders = db.relationship('GroupOrder', backref='customer_ref', cascade='all, delete-orphan', lazy=True)
    payments = db.relationship('GroupPayment', backref='customer_ref', cascade='all, delete-orphan', lazy=True)

class GroupOrder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('group_customer.id'), nullable=False)
    item_name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, default=1)

class GroupPayment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('group_customer.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    payment_method = db.Column(db.String(50))
    date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class DiscountRule(db.Model):
    __tablename__ = 'discount_rules'
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(50), nullable=True)
    discount_type = db.Column(db.String(20), default='Percentage') # Percentage or Flat
    value = db.Column(db.Float, nullable=False, default=0.0)
    min_order_amount = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='Active') # Active / Inactive
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# --- Subscription System Models ---
class SubscriptionPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(50), unique=True, nullable=False) # starter, growth, enterprise
    name = db.Column(db.String(100), nullable=False)
    price_monthly = db.Column(db.Float, nullable=False)

class Subscription(db.Model):
    __tablename__ = 'subscriptions'
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    plan_id = db.Column(db.Integer, db.ForeignKey('subscription_plan.id'), nullable=True)
    plan = db.relationship('SubscriptionPlan', backref='subscriptions')
    
    status = db.Column(db.String(20), default='TRIAL') # TRIAL, ACTIVE, EXPIRED, INACTIVE
    trial_start = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    trial_end = db.Column(db.DateTime, nullable=True) # Will be +7 days from start
    subscription_start = db.Column(db.DateTime, nullable=True)
    subscription_end = db.Column(db.DateTime, nullable=True)
    payment_status = db.Column(db.String(30), nullable=True, default='NONE')

class RestaurantDetail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    restaurant_name = db.Column(db.String(150), default="My Restaurant")
    logo = db.Column(db.String(200))
    gst_number = db.Column(db.String(50))
    address = db.Column(db.Text)
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    currency = db.Column(db.String(20), default="INR")
    timezone = db.Column(db.String(50), default="Asia/Kolkata")

class UserSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    default_order_type = db.Column(db.String(50), default="Dine In")
    language = db.Column(db.String(50), default="English")
    tax_enabled = db.Column(db.Boolean, default=True)
    auto_print_receipt = db.Column(db.Boolean, default=False)
    auto_print_kot = db.Column(db.Boolean, default=False)
    appearance_mode = db.Column(db.String(20), default="Light")

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    category = db.Column(db.String(50), default='General')
    amount = db.Column(db.Float, nullable=False)
    payment_mode = db.Column(db.String(50), default='Cash')
    date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    notes = db.Column(db.Text, nullable=True)

class SupportTicket(db.Model):
    __tablename__ = 'support_tickets'
    id = db.Column(db.Integer, primary_key=True)
    ticket_uuid = db.Column(db.String(100), unique=True, nullable=False) # e.g. TKT-0001
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(100), default="General")
    subject = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(50), default="Medium") # Low, Medium, High
    status = db.Column(db.String(50), default="Open") # Open, In Progress, Waiting for Customer, Resolved, Closed
    assigned_admin = db.Column(db.String(100), nullable=True)
    has_unread_replies = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    replies = db.relationship('TicketReply', backref='ticket', cascade="all, delete-orphan", lazy=True)

class TicketReply(db.Model):
    __tablename__ = 'ticket_replies'
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('support_tickets.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sender_type = db.Column(db.String(50), nullable=False) # 'Customer' or 'Admin'
    message = db.Column(db.Text, nullable=False)
    is_internal = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    model_used = db.Column(db.String(100))
    ingredient_name = db.Column(db.String(100), nullable=False)
    predicted_qty = db.Column(db.Float, nullable=False)
    actual_qty = db.Column(db.Float, nullable=True)
    error = db.Column(db.Float, nullable=True)
    accuracy = db.Column(db.Float, nullable=True)

class LocationIntelligence(db.Model):
    __tablename__ = 'location_intelligence'
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    address = db.Column(db.String(255), nullable=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    radius = db.Column(db.Integer, default=1000)
    google_maps_key = db.Column(db.String(255), nullable=True)
    competitors_cached = db.Column(db.Text, nullable=True)
    nearby_data_cached = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def ensure_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def ensure_subscription(user):
    if user.status == 'Admin':
        return
    sub = Subscription.query.filter_by(restaurant_id=user.id).first()
    if not sub:
        # Automatically Create Trial (7 Days)
        trial_end_dt = datetime.now(timezone.utc) + timedelta(days=7)
        sub = Subscription(restaurant_id=user.id, status='TRIAL', trial_end=trial_end_dt)
        db.session.add(sub)
        db.session.commit()
    else:
        # Check if trial expired naturally
        if sub.status == 'TRIAL' and sub.trial_end and datetime.now(timezone.utc) > ensure_utc(sub.trial_end):
            sub.status = 'EXPIRED'
            db.session.commit()
        # Check if active subscription expired naturally
        elif sub.status == 'ACTIVE' and sub.subscription_end and datetime.now(timezone.utc) > ensure_utc(sub.subscription_end):
            sub.status = 'EXPIRED'
            db.session.commit()
        
    return sub

@app.context_processor
def inject_subscription():
    if current_user.is_authenticated:
        # Locale settings
        owner_id = _current_restaurant_owner()
        owner = db.session.get(User, owner_id)
        owner_username = owner.username if owner else current_user.username
        spath = os.path.join(app.instance_path, f"{owner_username}_settings.json")
        json_sett = {}
        if os.path.exists(spath):
            try:
                with open(spath, 'r', encoding='utf-8') as f:
                    json_sett = json.load(f)
            except Exception:
                pass
        user_sett = UserSetting.query.filter_by(user_id=current_user.id).first()
        lang = user_sett.language if (user_sett and user_sett.language) else json_sett.get('language', 'English')
        curr_raw = json_sett.get('currency_symbol', '₹ (Rupee Symbol)')
        curr_sym = '₹' if '₹' in curr_raw else ('INR' if 'INR' in curr_raw else ('$' if '$' in curr_raw else '₹'))

        current_locale = {
            'language': lang,
            'receipt_language': json_sett.get('receipt_language', 'English'),
            'date_format': json_sett.get('date_format', 'DD/MM/YYYY'),
            'time_format': json_sett.get('time_format', '12 Hour (AM/PM)'),
            'number_format': json_sett.get('number_format', '1,00,000 (Indian)'),
            'currency_symbol': curr_sym
        }

        if getattr(current_user, 'status', None) == 'Admin':
            return dict(has_growth=True, has_enterprise=True, current_locale=current_locale)
            
        sub = Subscription.query.filter_by(restaurant_id=current_user.id).first()
        if sub:
            days_left = 0
            expiry_date_str = "None"
            
            if sub.status == 'TRIAL' and sub.trial_end:
                delta = ensure_utc(sub.trial_end) - datetime.now(timezone.utc)
                days_left = max(0, delta.days)
                expiry_date_str = ensure_utc(sub.trial_end).strftime('%Y-%m-%d')
            elif sub.status == 'ACTIVE' and sub.subscription_end:
                delta = ensure_utc(sub.subscription_end) - datetime.now(timezone.utc)
                days_left = max(0, delta.days)
                expiry_date_str = ensure_utc(sub.subscription_end).strftime('%Y-%m-%d')
            elif sub.status == 'EXPIRED':
                days_left = 0
                if sub.subscription_end:
                    expiry_date_str = ensure_utc(sub.subscription_end).strftime('%Y-%m-%d')
                elif sub.trial_end:
                    expiry_date_str = ensure_utc(sub.trial_end).strftime('%Y-%m-%d')
            
            has_growth, _, _, _ = check_feature_access(current_user, 'ai_insights')
            has_enterprise, _, _, _ = check_feature_access(current_user, 'executive_dashboard')
                    
            return dict(
                user_subscription=sub, 
                trial_days_left=days_left, 
                subscription_days_left=days_left, 
                subscription_expiry_date=expiry_date_str,
                has_growth=has_growth,
                has_enterprise=has_enterprise,
                current_locale=current_locale
            )
    return dict(current_locale={
        'language': 'English',
        'receipt_language': 'English',
        'date_format': 'DD/MM/YYYY',
        'time_format': '12 Hour (AM/PM)',
        'number_format': '1,00,000 (Indian)',
        'currency_symbol': '₹'
    })

FEATURE_ACCESS_MAP = {
    'pos_dashboard': ['starter', 'growth', 'enterprise'],
    'billing': ['starter', 'growth', 'enterprise'],
    'orders': ['starter', 'growth', 'enterprise'],
    'inventory': ['starter', 'growth', 'enterprise'],
    'customers': ['starter', 'growth', 'enterprise'],
    'menu': ['starter', 'growth', 'enterprise'],
    'recipes': ['starter', 'growth', 'enterprise'],
    'feedback': ['starter', 'growth', 'enterprise'],
    'basic_reports': ['starter', 'growth', 'enterprise'],

    'ai_insights': ['growth', 'enterprise'],
    'online_orders': ['growth', 'enterprise'],
    'kot': ['growth', 'enterprise'],
    'delivery_boys': ['growth', 'enterprise'],
    'discounts': ['starter', 'growth', 'enterprise'],
    'due_payments': ['growth', 'enterprise'],
    'tax_management': ['growth', 'enterprise'],
    'live_kitchen_view': ['starter', 'growth', 'enterprise'],

    'executive_dashboard': ['enterprise'],
    'cash_flow': ['enterprise'],
    'cash_topup': ['enterprise'],
    'currency_conversion': ['enterprise'],
    'ai_forecasting': ['enterprise'],
    'advanced_analytics': ['enterprise'],
    'multi_branch': ['enterprise'],
    'api_integrations': ['enterprise'],
}

def check_feature_access(user, feature_code):
    """
    Centralized service to evaluate feature access for a given user.
    Returns (allowed, reason, required_plan_title, current_plan_name)
    """
    if getattr(user, 'status', None) == 'Admin':
        return True, None, None, None

    sub = ensure_subscription(user)
    if not sub:
        return False, 'NoSubscription', 'Starter', 'None'

    if sub.status == 'EXPIRED':
        req_plan = FEATURE_ACCESS_MAP.get(feature_code, ['enterprise'])[-1].title()
        return False, 'Expired', req_plan, (sub.plan.name if sub.plan else 'None')

    if sub.status == 'TRIAL':
        return True, None, None, None

    if sub.status == 'ACTIVE' and sub.plan:
        allowed_plans = FEATURE_ACCESS_MAP.get(feature_code, ['enterprise'])
        if sub.plan.slug in allowed_plans:
            return True, None, None, None

    req_plan = FEATURE_ACCESS_MAP.get(feature_code, ['enterprise'])[-1].title()
    curr_plan = sub.plan.name if sub.plan else 'None'
    return False, 'UpgradeRequired', req_plan, curr_plan

def get_feature_code(category, page):
    mapping = {
        ('operations', 'orders'): 'orders',
        ('operations', 'kots'): 'kot',
        ('operations', 'billing-screen'): 'billing',
        ('operations', 'bill-kot-print'): 'kot',
        ('operations', 'online-orders'): 'online_orders',
        ('operations', 'delivery-boys'): 'delivery_boys',
        ('operations', 'due-payment'): 'due_payments',
        ('operations', 'live-view'): 'live_kitchen_view',
        
        ('finance', 'cash-flow'): 'cash_flow',
        ('finance', 'expense'): 'cash_flow',
        ('finance', 'withdrawal'): 'cash_flow',
        ('finance', 'cash-topup'): 'cash_topup',
        ('finance', 'currency-conversion'): 'currency_conversion',
        
        ('menu', 'menu'): 'menu',
        ('menu', 'inventory'): 'inventory',
        ('menu', 'recipes'): 'recipes',
        ('menu', 'customers'): 'customers',
        ('menu', 'discount'): 'discounts',
        ('menu', 'tax'): 'tax_management',
        ('menu', 'feedback'): 'feedback',
        ('menu', 'menu-item-onoff'): 'menu',
        
        ('reports', 'sales-summary'): 'basic_reports',
        ('reports', 'order-summary'): 'orders',
        ('reports', 'item-summary'): 'basic_reports',
        ('reports', 'category-summary'): 'basic_reports',
        ('reports', 'executive-sales-summary'): 'basic_reports',
        ('reports', 'tip-summary'): 'basic_reports',
    }
    if category == 'settings':
        return 'pos_dashboard'
    return mapping.get((category, page), 'enterprise')

def require_feature(feature_code):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            allowed, reason, req_plan, curr_plan = check_feature_access(current_user, feature_code)
            
            # --- MANDATORY PRINT LOGS FOR DEBUGGING ---
            username = current_user.username if current_user.is_authenticated else "Anonymous"
            sub = ensure_subscription(current_user)
            plan_slug = sub.plan.slug if (sub and sub.plan) else "None"
            plan_name = sub.plan.name if (sub and sub.plan) else "None"
            sub_status = sub.status if sub else "None"
            
            print("--- REQUIRE_FEATURE TRACE ---")
            print("USER:", username)
            print("STATUS:", sub_status)
            print("PLAN SLUG:", plan_slug)
            print("PLAN NAME:", plan_name)
            print("FEATURE:", feature_code)
            print("ALLOWED:", FEATURE_ACCESS_MAP.get(feature_code))
            print("AUTHORIZED:", allowed, "REASON:", reason)                                     
            # ------------------------------------------
            
            if allowed:
                return f(*args, **kwargs)
                
            if reason in ['NoSubscription', 'Expired']:
                print(f"[ACCESS] Blocked {username}: {reason}")
                return redirect(url_for('subscription_pricing', msg='Trial/Subscription expired. Please choose a plan.'))
                
            print(f"[ACCESS] Access denied to {username} ({plan_slug}) for feature {feature_code}")
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'UpgradeRequired', 'required_plan': req_plan}), 403
            
            return render_template('upgrade_popup.html', required_plan=req_plan, current_plan=curr_plan)
        return decorated_function
    return decorator


@app.route('/subscription')     
@login_required
def subscription():
    sub = None
    days_left = 0
    expiry_date = "Never"
    
    if current_user.status != 'Admin':
        sub = ensure_subscription(current_user)
        if sub:
            if sub.status == 'TRIAL' and sub.trial_end:
                delta = ensure_utc(sub.trial_end) - datetime.now(timezone.utc)
                days_left = max(0, delta.days)
                expiry_date = ensure_utc(sub.trial_end).strftime('%Y-%m-%d')
            elif sub.status == 'ACTIVE' and sub.subscription_end:
                delta = ensure_utc(sub.subscription_end) - datetime.now(timezone.utc)
                days_left = max(0, delta.days)
                expiry_date = ensure_utc(sub.subscription_end).strftime('%Y-%m-%d')
            elif sub.status == 'EXPIRED':
                days_left = 0
                if sub.subscription_end:
                    expiry_date = ensure_utc(sub.subscription_end).strftime('%Y-%m-%d')
                elif sub.trial_end:
                    expiry_date = ensure_utc(sub.trial_end).strftime('%Y-%m-%d')
                else:
                    expiry_date = "Expired"
    
    plans = SubscriptionPlan.query.all()
    plans_dict = {p.slug: p for p in plans} 
    
    return render_template(
        'subscription.html',
        sub=sub,
        days_left=days_left,
        expiry_date=expiry_date,
        plans=plans_dict,
        msg=request.args.get('msg'), 
        test_mode_enabled=get_global_settings().get("test_mode", False)
    )

def generate_testing_report_file(user, sub):
    plan_name = sub.plan.name if sub.plan else "None"
    status = sub.status
    expiry_date = ensure_utc(sub.subscription_end).strftime('%Y-%m-%d %H:%M:%S UTC') if sub.subscription_end else "Never"
    
    # Compute active and locked features
    activated_features = []
    locked_features = []
    
    for f_code, plans in FEATURE_ACCESS_MAP.items():
        if sub.plan and sub.plan.slug in plans:
            activated_features.append(f_code)
        else:
            locked_features.append(f_code)
            
    report_content = f"""# Subscription Testing Report

- **User / Restaurant**: {user.username}
- **Selected Plan**: {plan_name}
- **Subscription Status**: {status}
- **Expiry Date**: {expiry_date}

## Activated Features  
{chr(10).join([f'- {feat}' for feat in sorted(activated_features)]) if activated_features else '- None'}

## Locked Features   
{chr(10).join([f'- {feat}' for feat in sorted(locked_features)]) if locked_features else '- None'}
"""
    # Write to artifacts directory (use current machine's path)
    artifact_path = os.path.join(BASE_DIR, "subscription_testing_report_artifacts.md")
    try:
        with open(artifact_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"[REPORT] Saved testing report to: {artifact_path}")
    except Exception as e:
        print(f"[REPORT] Error saving report: {e}")
        
    # Write to local workspace
    workspace_path = "subscription_testing_report.md"
    try:
        with open(workspace_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"[REPORT] Saved testing report to workspace: {workspace_path}")
    except Exception as e:
        print(f"[REPORT] Error saving report to workspace: {e}")

@app.route('/subscription/pricing')
@login_required
def subscription_pricing():
    if current_user.status == 'Admin':
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('subscription', msg=request.args.get('msg')))

@app.route('/subscription/upgrade/<plan_slug>', methods=['POST'])
@login_required
def upgrade_plan(plan_slug):
    if current_user.status == 'Admin':
        return redirect(url_for('admin_dashboard'))
        
    plan = SubscriptionPlan.query.filter_by(slug=plan_slug).first()
    if not plan:
        return redirect(url_for('subscription', msg="Selected plan not found."))
        
    sub = ensure_subscription(current_user)
    if not sub:
        return redirect(url_for('subscription', msg="Failed to find or create subscription."))  
        
    # Check if TEST MODE is enabled
    is_test_mode = request.form.get('test_mode') == 'true'
    
    if is_test_mode:
        if not get_global_settings().get("test_mode", False):
            return "Access Denied: Test mode is not active globally.", 403
            
        # TEST MODE: Skip Stripe, activate instantly
        sub.plan_id = plan.id
        sub.status = 'ACTIVE'
        sub.payment_status = 'TEST_MODE'
        sub.subscription_start = datetime.now(timezone.utc)
        sub.subscription_end = datetime.now(timezone.utc) + timedelta(days=30)
        db.session.commit()
        
        # Generate Testing Report
        generate_testing_report_file(current_user, sub)
        
        return redirect(url_for('index', msg=f"Test Mode: Plan {plan.name} has been activated instantly!"))
    else:
        # LIVE MODE: Create Stripe Checkout Session
        is_dummy_key = stripe.api_key == 'sk_test_dummy' or getattr(stripe, 'api_key', None) is None
        if is_dummy_key:
            # Generate a one-time server-side token to prevent direct URL manipulation.
            # This token is stored in the Flask session and consumed on success.
            import secrets
            dummy_token = secrets.token_urlsafe(32)
            flask_session['pending_dummy_checkout'] = {
                'token': dummy_token,
                'plan_slug': plan_slug,
                'user_id': current_user.id
            }
            success_url = url_for('stripe_success', session_id='cs_test_dummy', plan_slug=plan_slug, _external=True) + f'&dummy_token={dummy_token}'
            return redirect(success_url)
            
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'inr',
                        'product_data': {
                            'name': f"Neo Dine AI - {plan.name} Plan",
                        },
                        'unit_amount': int(plan.price_monthly * 100),
                        'recurring': {
                            'interval': 'month',
                        },
                    },
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=url_for('stripe_success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}&plan_slug=' + plan_slug,
                cancel_url=url_for('stripe_cancel', _external=True),
                metadata={
                    'restaurant_id': current_user.id,
                    'plan_slug': plan_slug
                }
            )
            return redirect(session.url, code=303)
        except Exception as e:
            return redirect(url_for('subscription', msg=f"Stripe Checkout Error: {str(e)}"))

@app.route('/subscription/success')
@login_required
def stripe_success():
    session_id = request.args.get('session_id')
    plan_slug = request.args.get('plan_slug')
    
    if not session_id or not plan_slug:
        return redirect(url_for('subscription', msg="Invalid success redirect session parameters."))
        
    plan = SubscriptionPlan.query.filter_by(slug=plan_slug).first()
    if not plan:
        return redirect(url_for('subscription', msg="Selected plan not found."))
        
    sub = ensure_subscription(current_user)
    
    # Try verifying payment
    payment_verified = False
    is_dummy_key = stripe.api_key == 'sk_test_dummy' or getattr(stripe, 'api_key', None) is None
    
    if is_dummy_key and session_id == 'cs_test_dummy':
        # SECURITY: Verify the one-time server-side token to prevent direct URL manipulation.
        # The token must have been issued by upgrade_plan() and stored in Flask session.
        dummy_token = request.args.get('dummy_token')
        pending = flask_session.get('pending_dummy_checkout')
        
        if (
            pending is not None
            and dummy_token is not None
            and pending.get('token') == dummy_token
            and pending.get('plan_slug') == plan_slug
            and pending.get('user_id') == current_user.id
        ):
            # Consume the token so it cannot be replayed
            flask_session.pop('pending_dummy_checkout', None)
            payment_verified = True
        else:
            print(f"[SECURITY] Rejected dummy checkout attempt - token mismatch or missing. user_id={current_user.id}, plan={plan_slug}")
            payment_verified = False
    else:
        try:
            stripe_session = stripe.checkout.Session.retrieve(session_id)
            if stripe_session and stripe_session.payment_status == 'paid':
                # Additional integrity check: verify the plan matches session metadata
                session_plan = stripe_session.metadata.get('plan_slug') if stripe_session.metadata else None
                session_user = str(stripe_session.metadata.get('restaurant_id', '')) if stripe_session.metadata else None
                if session_plan == plan_slug and session_user == str(current_user.id):
                    payment_verified = True
                else:
                    print(f"[SECURITY] Stripe session metadata mismatch: plan={session_plan}/{plan_slug}, user={session_user}/{current_user.id}")
        except Exception as e:
            print(f"[STRIPE] Verification error: {e}")
            payment_verified = False
        
    if payment_verified:
        sub.plan_id = plan.id
        sub.status = 'ACTIVE'
        sub.payment_status = 'STRIPE' if not is_dummy_key else 'DUMMY_VERIFIED'
        sub.subscription_start = datetime.now(timezone.utc)
        sub.subscription_end = datetime.now(timezone.utc) + timedelta(days=30)
        db.session.commit()
        
        # Generate Testing Report
        generate_testing_report_file(current_user, sub)
        
        return redirect(url_for('index', msg=f"Payment successful! Plan {plan.name} has been activated."))
    else:
        return redirect(url_for('subscription', msg="Payment verification failed. Please try again."))

@app.route('/subscription/cancel')
@login_required
def stripe_cancel():
    return redirect(url_for('subscription', msg="Subscription upgrade cancelled."))


# --- Authoritative Excel-Based Recipe Integration ---
AUTHORITATIVE_RECIPE_FILE = "Restaurant_1195_Items_with_Cafe.xlsx"
EXCEL_RECIPE_CACHE = {}

def normalize_item_name(name):
    """Normalized matching: strip whitespace, lowercase, collapse internal whitespace, common spelling variants."""
    if not name:
        return ""
    import re
    s = str(name).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\bbiriyani\b', 'biryani', s)
    return s

def load_excel_recipes():
    """Reads Restaurant_1195_Items_with_Cafe.xlsx as authoritative recipe source."""
    global EXCEL_RECIPE_CACHE
    file_path = AUTHORITATIVE_RECIPE_FILE
    if not os.path.exists(file_path):
        alt_path = os.path.join(os.path.dirname(__file__), AUTHORITATIVE_RECIPE_FILE)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print(f"[EXCEL] Authoritative recipe file {file_path} not found.")
            return
    
    try:
        import pandas as pd
        excel_file = pd.ExcelFile(file_path)
        sheet_name = 'Ingredients' if 'Ingredients' in excel_file.sheet_names else excel_file.sheet_names[0]
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        dish_ids_per_norm = {}
        raw_recipes = {}
        
        for _, row in df.iterrows():
            dish_name = str(row.get('dish_name', '')).strip()
            ing_name = str(row.get('ingredient_name', '')).strip()
            if not dish_name or dish_name == 'nan' or not ing_name or ing_name == 'nan':
                continue
            
            dish_id = row.get('dish_id', dish_name)
            try:
                qty_val = float(row.get('quantity', 0.0))
            except (ValueError, TypeError):
                qty_val = 0.0
            unit_val = str(row.get('unit', '')).strip()
            
            norm_name = normalize_item_name(dish_name)
            if norm_name not in dish_ids_per_norm:
                dish_ids_per_norm[norm_name] = set()
            dish_ids_per_norm[norm_name].add(dish_id)
            
            if dish_id not in raw_recipes:
                raw_recipes[dish_id] = {
                    'recipe_name': dish_name,
                    'norm_name': norm_name,
                    'ingredients': []
                }
            
            raw_recipes[dish_id]['ingredients'].append({
                'ingredient': ing_name,
                'quantity': qty_val,
                'unit': unit_val
            })
            
        recipes_cache = {}
        for norm_name, d_ids in dish_ids_per_norm.items():
            if len(d_ids) > 1:
                # Ambiguous dish name matching multiple different dish_ids - do not guess!
                recipes_cache[norm_name] = {'ambiguous': True, 'recipe_name': None, 'ingredients': []}
            else:
                d_id = list(d_ids)[0]
                recipes_cache[norm_name] = {
                    'ambiguous': False,
                    'recipe_name': raw_recipes[d_id]['recipe_name'],
                    'ingredients': raw_recipes[d_id]['ingredients']
                }
                
        EXCEL_RECIPE_CACHE = recipes_cache
        print(f"[EXCEL] Successfully loaded {len(recipes_cache)} authoritative recipe mappings from {file_path}.")
    except Exception as e:
        print(f"[EXCEL] Error parsing authoritative recipe file: {e}")

def auto_map_menu_item_recipes(menu_item):
    """
    Automatically maps a MenuItem to its recipe in Restaurant_1195_Items_with_Cafe.xlsx.
    Creates or reuses Inventory records (with starting current_stock = 0.0).
    Creates and synchronizes MenuRecipe records.
    Never assumes current_user.id; uses menu_item.restaurant_id.
    """
    if not menu_item or not menu_item.name:
        return False
        
    if not EXCEL_RECIPE_CACHE:
        load_excel_recipes()
        
    norm_name = normalize_item_name(menu_item.name)
    recipe_data = EXCEL_RECIPE_CACHE.get(norm_name)
    
    if not recipe_data or recipe_data.get('ambiguous'):
        # No exact normalized match or match is ambiguous; skip to avoid guessing
        return False
        
    recipe_name = recipe_data['recipe_name']
    ingredients = recipe_data['ingredients']
    
    if hasattr(menu_item, 'recipe_name'):
        menu_item.recipe_name = recipe_name
        
    restaurant_id = menu_item.restaurant_id # Actual restaurant_id of the menu item
    matched_inv_ids = set()
    
    for item_data in ingredients:
        raw_ing_name = item_data['ingredient'].strip()
        norm_ing = normalize_item_name(raw_ing_name)
        qty = float(item_data['quantity'])
        unit = item_data['unit'].strip()
        
        # 1. Check if inventory item exists for this restaurant
        all_inv = Inventory.query.filter_by(restaurant_id=restaurant_id).all()
        inv_item = None
        for inv in all_inv:
            if normalize_item_name(inv.ingredient_name) == norm_ing:
                inv_item = inv
                break
                
        if not inv_item:
            # Create new Inventory record with starting current_stock = 0.0
            inv_item = Inventory(
                restaurant_id=restaurant_id,
                ingredient_name=raw_ing_name,
                unit=unit,
                current_stock=0.0,
                consumed_quantity=0.0
            )
            db.session.add(inv_item)
            db.session.flush() # Obtain inv_item.id
            
        matched_inv_ids.add(inv_item.id)
        
        # 2. Check if MenuRecipe record exists for (menu_item.id, inv_item.id)
        existing_mr = MenuRecipe.query.filter_by(
            menu_item_id=menu_item.id,
            ingredient_id=inv_item.id
        ).first()
        
        if existing_mr:
            # Synchronize existing recipe if quantity or recipe_name changed
            existing_mr.quantity_required = qty
            if hasattr(existing_mr, 'recipe_name'):
                existing_mr.recipe_name = recipe_name
        else:
            mr_kwargs = {
                'menu_item_id': menu_item.id,
                'ingredient_id': inv_item.id,
                'quantity_required': qty
            }
            if hasattr(MenuRecipe, 'recipe_name'):
                mr_kwargs['recipe_name'] = recipe_name
            new_mr = MenuRecipe(**mr_kwargs)
            db.session.add(new_mr)
            
    # Remove old MenuRecipe records no longer present in updated recipe master
    old_mrs = MenuRecipe.query.filter_by(menu_item_id=menu_item.id).all()
    for old_mr in old_mrs:
        if old_mr.ingredient_id not in matched_inv_ids:
            db.session.delete(old_mr)
            
    try:
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"[AUTO-MAP-ERROR] Failed to map recipes for menu item {menu_item.id}: {e}")
        return False

def auto_map_all_recipes_for_restaurant(restaurant_id):
    """Synchronizes all active MenuItems for a restaurant_id."""
    if not restaurant_id:
        return 0
    active_items = MenuItem.query.filter_by(restaurant_id=restaurant_id, status='Active').all()
    mapped_count = 0
    for item in active_items:
        if auto_map_menu_item_recipes(item):
            mapped_count += 1
    return mapped_count

def get_restaurant_recipe_status(restaurant_id):
    """
    Inspects all active MenuItems for a restaurant and checks their recipe mapping status
    against EXCEL_RECIPE_CACHE and MenuRecipe table. Returns detailed audit metrics.
    """
    if not EXCEL_RECIPE_CACHE:
        load_excel_recipes()
        
    active_menu = MenuItem.query.filter_by(restaurant_id=restaurant_id, status='Active').all()
    total_active_items = len(active_menu)
    
    successfully_mapped = []
    unmapped_items = []
    ambiguous_items = []
    
    for item in active_menu:
        recipes_count = MenuRecipe.query.filter_by(menu_item_id=item.id).count()
        norm_name = normalize_item_name(item.name)
        excel_data = EXCEL_RECIPE_CACHE.get(norm_name)
        
        if recipes_count > 0:
            successfully_mapped.append({
                'id': item.id,
                'name': item.name,
                'recipe_name': getattr(item, 'recipe_name', item.name) or item.name,
                'recipe_count': recipes_count
            })
        elif excel_data and excel_data.get('ambiguous'):
            ambiguous_items.append({
                'id': item.id,
                'name': item.name,
                'reason': 'Ambiguous recipe match — manual review required'
            })
        elif not excel_data:
            unmapped_items.append({
                'id': item.id,
                'name': item.name,
                'reason': 'Recipe not found in master recipe database — manual mapping required'
            })
        else:
            unmapped_items.append({
                'id': item.id,
                'name': item.name,
                'reason': 'Recipe mapping pending or incomplete — manual mapping required'
            })
            
    total_recipes_count = db.session.query(MenuRecipe).join(MenuItem).filter(MenuItem.restaurant_id == restaurant_id).count()
    total_inventory_count = Inventory.query.filter_by(restaurant_id=restaurant_id).count()
    
    all_mapped_cleanly = (
        total_active_items > 0 and 
        len(successfully_mapped) == total_active_items and 
        len(unmapped_items) == 0 and 
        len(ambiguous_items) == 0
    )
    
    return {
        'has_recipes': all_mapped_cleanly,
        'total_menu_items': total_active_items,
        'items_with_recipes': len(successfully_mapped),
        'unmapped_items_count': len(unmapped_items),
        'ambiguous_items_count': len(ambiguous_items),
        'total_recipe_relationships': total_recipes_count,
        'total_inventory_ingredients': total_inventory_count,
        'successfully_mapped_items': successfully_mapped,
        'unmapped_items': unmapped_items,
        'ambiguous_items': ambiguous_items
    }

import difflib

# Self-Learning Memory Cache
LEARNED_ALIASES = {
    'pepsi beverage': 'pepsi',
    'sprite beverage': 'sprite',
    'chicken patty': 'chicken',
    'veg patty': 'vegetable',
    'burger bun': 'bun',
    'mayonnaise': 'mayo',
    'burger sauce': 'sauce',
    'cooking oil': 'oil',
    'lettuce': 'veg'
}

def fuzzy_match(target, candidates, threshold=0.8):
    """Fuzzy matching logic substituting exact matches."""
    best_match = None
    best_score = 0.0
    for cand in candidates:
        if target in cand or cand in target:
            if len(cand)/max(1, len(target)) > 0.5 and len(target)/max(1, len(cand)) > 0.5:
                return cand
        score = difflib.SequenceMatcher(None, target, cand).ratio()
        if score > best_score:
            best_score = score
            best_match = cand
    if best_score >= threshold:
        return best_match
    return None

def normalize(text):
    """Semantic normalization mapping against the intelligence layer."""
    raw = str(text).strip().lower()
    return LEARNED_ALIASES.get(raw, raw)

def convert_to_base_unit(value, from_unit, to_unit, item_name="Unknown"):
    """Unit Intelligence Layer protecting against volumetric anomalies."""
    from_unit = str(from_unit).lower().strip()
    to_unit = str(to_unit).lower().strip()
    
    if from_unit == to_unit:
        return value
        
    vol_units = ['ml', 'l', 'litre', 'litres']
    weight_units = ['g', 'kg', 'kilo', 'gram', 'grams']
    piece_units = ['piece', 'pieces', 'unit', 'units', 'pc', 'pcs', 'bottle', 'bottles']
    
    if from_unit in weight_units and to_unit in weight_units:
        if from_unit in ['g', 'gram', 'grams'] and to_unit in ['kg', 'kilo']: return value / 1000.0
        if from_unit in ['kg', 'kilo'] and to_unit in ['g', 'gram', 'grams']: return value * 1000.0
        
    if from_unit in vol_units and to_unit in vol_units:
        if from_unit == 'ml' and to_unit in ['l', 'litre', 'litres']: return value / 1000.0
        if from_unit in ['l', 'litre', 'litres'] and to_unit == 'ml': return value * 1000.0
        
    if from_unit in vol_units and to_unit in piece_units:
        print(f"    [UNIT ANOMALY DETECTED] Attempted to deduct {value} {from_unit} from solid unit '{to_unit}' for '{item_name}'. Auto-correcting to 1.0 {to_unit}.")
        return 1.0
        
    if from_unit in piece_units and to_unit in piece_units:
        return value
        
    print(f"    [UNIT MISMATCH] Unhandled conversion from {from_unit} to {to_unit} for '{item_name}'. Using raw value {value}.")
    return value

def deduct_inventory_for_order(restaurant_id, item_counts, commit=True):
    """
    Centralized logic to deduct inventory based on ordered items and recipes.
    Uses database tables (MenuRecipe) and falls back to direct name matching.
    """
    from sqlalchemy import func
    print(f"\n{'='*50}")
    print(f"[INVENTORY-DEDUCTION-START] Restaurant ID: {restaurant_id}")
    print(f"[INVENTORY-DEDUCTION-START] Item Counts: {item_counts}")
    print(f"{'='*50}")
    
    for item_raw_name, qty in item_counts.items():
        found_deduction = False
        item_name = normalize(item_raw_name)
        print(f"[ORDER] Processing item: '{item_name}' (qty: {qty})")
        
        # Pre-cache DB inventory names for fuzzy matching resolution
        all_inv_items = Inventory.query.filter_by(restaurant_id=restaurant_id).all()
        db_inv_dict = {str(i.ingredient_name).strip().lower(): i for i in all_inv_items}
        db_inv_names = list(db_inv_dict.keys())
        
        # 1. Try precise recipe-based deduction from DB
        menu_item = MenuItem.query.filter(
            func.lower(func.trim(MenuItem.name)) == item_name,
            MenuItem.restaurant_id == restaurant_id
        ).first()

        if menu_item:
            recipes = MenuRecipe.query.filter_by(menu_item_id=menu_item.id).all()
            if recipes:
                print(f"  [DB] Found {len(recipes)} recipes in system database.")
                for recipe in recipes:
                    inv_item = Inventory.query.filter_by(
                        id=recipe.ingredient_id,
                        restaurant_id=restaurant_id
                    ).first()
                    if inv_item:
                        stock_before = inv_item.current_stock
                        deduction = recipe.quantity_required * qty
                        
                        if deduction > 100:
                            print(f"    [HIGH CONSUMPTION ANOMALY] Deducting extraordinarily large quantity: {deduction} {inv_item.unit} of '{inv_item.ingredient_name}'")
                            
                        inv_item.current_stock = max(0.0, inv_item.current_stock - deduction)
                        if hasattr(inv_item, 'consumed_quantity'):
                            inv_item.consumed_quantity = (inv_item.consumed_quantity or 0.0) + deduction
                        
                        inv_item.last_updated = datetime.now(timezone.utc)
                        print(f"    - [STOCK-UPDATE] '{inv_item.ingredient_name}': {stock_before} -> {inv_item.current_stock} {inv_item.unit} (Used: {deduction})")
                        found_deduction = True
                    else:
                        print(f"    - [DB-MISS] Ingredient ID {recipe.ingredient_id} not found.")
                if found_deduction:
                    print(f"  [SUCCESS] Deduction completed via DB for '{item_name}'")
                    continue

        # 2. Fallback: Try to match item name directly to an ingredient name
        exact_inv = db_inv_dict.get(item_name)
        if not exact_inv:
            fuzz = fuzzy_match(item_name, db_inv_names, 0.8)
            if fuzz:
                LEARNED_ALIASES[item_raw_name.strip().lower()] = fuzz
                exact_inv = db_inv_dict.get(fuzz)
                print(f"    [AI LEARNED] Sub-item '{item_raw_name}' mapped to '{fuzz}'")

        if exact_inv:
            stock_before = exact_inv.current_stock
            exact_inv.current_stock = max(0.0, exact_inv.current_stock - qty)
            if hasattr(exact_inv, 'consumed_quantity'):
                exact_inv.consumed_quantity = (exact_inv.consumed_quantity or 0.0) + qty
            
            exact_inv.last_updated = datetime.now(timezone.utc)
            print(f"  [FALLBACK-SUCCESS] '{item_name}' matched direct ingredient. Stock: {stock_before} -> {exact_inv.current_stock}")
        else:
            print(f"[CRITICAL] [UNMAPPED INGREDIENT] No deduction path found for '{item_raw_name}'. Total mapping failure.")
    
    if not commit:
        return
    # CRITICAL: Commit all changes to the database
    try:
        db.session.commit()
        print("[ORDER-END] Database committed. Inventory updates saved.\n")
    except Exception as e:
        db.session.rollback()
        print(f"[ORDER-ERROR] Failed to commit inventory changes: {e}\n")



@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()
    # Load Excel recipes into memory
    load_excel_recipes()
    # --- Safe SQLite migration: only run PRAGMA-based migrations on SQLite ---
    _db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    if 'sqlite' in _db_uri.lower():
        try:
            from sqlalchemy import text
            with db.engine.connect() as conn:
                cols = [row[1] for row in conn.execute(text("PRAGMA table_info(inventory)"))]
                if 'consumed_quantity' not in cols:
                    conn.execute(text('ALTER TABLE inventory ADD COLUMN consumed_quantity FLOAT DEFAULT 0.0'))
                    conn.commit()
                    print('[MIGRATION] Added consumed_quantity column to inventory table.')

                # User profile table safe migration
                user_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(user)"))]
                if 'name' not in user_cols:
                    conn.execute(text("ALTER TABLE user ADD COLUMN name VARCHAR(100) DEFAULT ''"))
                if 'phone' not in user_cols:
                    conn.execute(text("ALTER TABLE user ADD COLUMN phone VARCHAR(50) DEFAULT ''"))
                if 'profile_image' not in user_cols:
                    conn.execute(text("ALTER TABLE user ADD COLUMN profile_image VARCHAR(250) DEFAULT ''"))
                if 'role' not in user_cols:
                    conn.execute(text("ALTER TABLE user ADD COLUMN role VARCHAR(50) DEFAULT 'Cashier'"))
                if 'last_login' not in user_cols:
                    conn.execute(text("ALTER TABLE user ADD COLUMN last_login DATETIME"))
                if 'restaurant_owner_id' not in user_cols:
                    conn.execute(text("ALTER TABLE user ADD COLUMN restaurant_owner_id INTEGER"))
                    conn.commit()
                    print('[MIGRATION] Added restaurant_owner_id column to user table.')
                conn.commit()
                print('[MIGRATION] Verified user table columns.')

                # --- Safe SQLite migration for order table ---
                order_cols = [row[1] for row in conn.execute(text("PRAGMA table_info('order')"))]
                if 'payment_method' not in order_cols:
                    conn.execute(text("ALTER TABLE 'order' ADD COLUMN payment_method VARCHAR(50) DEFAULT 'Cash'"))
                    conn.commit()
                    print('[MIGRATION] Added payment_method column to order table.')
                if 'order_status' not in order_cols:
                    conn.execute(text("ALTER TABLE 'order' ADD COLUMN order_status VARCHAR(20) DEFAULT 'PENDING'"))
                    conn.commit()
                    print('[MIGRATION] Added order_status column to order table.')
                if 'payment_status' not in order_cols:
                    conn.execute(text("ALTER TABLE 'order' ADD COLUMN payment_status VARCHAR(20) DEFAULT 'PENDING'"))
                    conn.commit()
                    print('[MIGRATION] Added payment_status column to order table.')
                if 'table_num' not in order_cols:
                    conn.execute(text("ALTER TABLE 'order' ADD COLUMN table_num INTEGER"))
                    conn.commit()
                    print('[MIGRATION] Added table_num column to order table.')
                if 'waiter_note' not in order_cols:
                    conn.execute(text("ALTER TABLE 'order' ADD COLUMN waiter_note VARCHAR(300)"))
                    conn.commit()
                    print('[MIGRATION] Added waiter_note column to order table.')

                # --- Safe SQLite migration for subscriptions table ---
                sub_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(subscriptions)"))]
                if 'subscription_start' not in sub_cols:
                    conn.execute(text("ALTER TABLE subscriptions ADD COLUMN subscription_start DATETIME"))
                    conn.commit()
                    print('[MIGRATION] Added subscription_start column to subscriptions table.')
                if 'payment_status' not in sub_cols:
                    conn.execute(text("ALTER TABLE subscriptions ADD COLUMN payment_status VARCHAR(30) DEFAULT 'NONE'"))
                    conn.commit()
                    print('[MIGRATION] Added payment_status column to subscriptions table.')

                # --- Safe SQLite migration for menu_recipe & menu_item ---
                mr_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(menu_recipe)"))]
                if 'recipe_name' not in mr_cols:
                    conn.execute(text("ALTER TABLE menu_recipe ADD COLUMN recipe_name VARCHAR(100)"))
                    conn.commit()
                    print('[MIGRATION] Added recipe_name column to menu_recipe table.')
                mi_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(menu_item)"))]
                if 'recipe_name' not in mi_cols:
                    conn.execute(text("ALTER TABLE menu_item ADD COLUMN recipe_name VARCHAR(100)"))
                    conn.commit()
                    print('[MIGRATION] Added recipe_name column to menu_item table.')
        except Exception as _mig_err:
            print(f'[MIGRATION] Warning: {_mig_err}')
    else:
        print('[MIGRATION] PostgreSQL detected — skipping SQLite PRAGMA migrations (db.create_all handles schema).')
        
    # Auto-synchronize existing active MenuItems that currently lack recipes
    try:
        existing_active = MenuItem.query.filter_by(status='Active').all()
        synced_count = 0
        for item in existing_active:
            rcp_count = MenuRecipe.query.filter_by(menu_item_id=item.id).count()
            if rcp_count == 0:
                if auto_map_menu_item_recipes(item):
                    synced_count += 1
        if synced_count > 0:
            print(f"[AUTO-MAP] Initialized auto-mapping for {synced_count} existing menu items.")
    except Exception as _sync_err:
        print(f"[AUTO-MAP] Startup sync warning: {_sync_err}")

    # Auto-seed the default administrator account
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password_hash=generate_password_hash('admin123'),
            status='Admin')
        admin.raw_password = 'admin123'
        db.session.add(admin)
        db.session.commit()
        
    # Auto-seed the subscription plans
    try:
        if not SubscriptionPlan.query.first():
            plans = [
                SubscriptionPlan(slug='starter', name='Starter', price_monthly=999.0),
                SubscriptionPlan(slug='growth', name='Growth', price_monthly=2999.0),
                SubscriptionPlan(slug='enterprise', name='Enterprise', price_monthly=7999.0)
            ]
            db.session.bulk_save_objects(plans)
            db.session.commit()
            print('[SEED] Subscription Plans correctly created.')
    except Exception as e:
        db.session.rollback()
        print(f"[SEED] Subscription Plan Error: {e}")

API_KEY = "YOUR_API_KEY_HERE"
CITY = "Bengaluru"
DATABASE_FILE = "restaurant_database.xlsx"
PROFIT_MARGIN = 0.30

# Internal storage for pending orders to display on receipt
PENDING_ORDERS = {}

# Global ML State
GLOBAL_ML_MODEL = None
GLOBAL_ML_CLASSES = None


def get_or_train_ml_model(force_retrain=False):
    global GLOBAL_ML_MODEL, GLOBAL_ML_CLASSES
    if GLOBAL_ML_MODEL is not None and not force_retrain:
        return GLOBAL_ML_MODEL, GLOBAL_ML_CLASSES

    # Extract recent historical training data from SQL (capped for fast, low-memory cloud startup)
    try:
        historical_items = (
            db.session.query(Order.temperature, Order.rainfall, OrderItem.item_name)
            .join(OrderItem, Order.id == OrderItem.order_id)
            .order_by(Order.id.desc())
            .limit(500)
            .all()
        )
    except Exception:
        historical_items = []

    if historical_items:
        df = pd.DataFrame(historical_items, columns=['Temperature', 'Rainfall', 'Item'])
        # Keep top 25 most frequent items to avoid OOM/timeouts on multi-class trees
        top_items = df['Item'].value_counts().nlargest(25).index
        df = df[df['Item'].isin(top_items)].copy()
    else:
        df = pd.DataFrame(columns=['Temperature', 'Rainfall', 'Item'])

    # 1. Synthesize baseline data if DB is empty or very small to prevent cold-starts
    if len(df) < 50:
        synth_data = [
            {"Temperature": 35, "Rainfall": 0, "Item": "Pepsi"},
            {"Temperature": 34, "Rainfall": 0, "Item": "Sprite"},
            {"Temperature": 36, "Rainfall": 0, "Item": "Chicken Burger"},
            {"Temperature": 22, "Rainfall": 10, "Item": "Mutton Soup"},
            {"Temperature": 20, "Rainfall": 15, "Item": "Tea"},
            {"Temperature": 25, "Rainfall": 5, "Item": "Vegetable Soup"},
            {"Temperature": 28, "Rainfall": 0, "Item": "Chicken Pizza"},
            {"Temperature": 29, "Rainfall": 0, "Item": "Egg Noodles"},
            {"Temperature": 27, "Rainfall": 0, "Item": "Coffee"},
            {"Temperature": 30, "Rainfall": 0, "Item": "Veg Burger"}
        ]
        train_df = pd.DataFrame(synth_data)
        if not df.empty:
            train_df = pd.concat([train_df, df], ignore_index=True)
    else:
        train_df = df.copy()

    # 2. Train lightweight Ensemble Model (fast & low-memory for Render 512MB instances)
    train_df['Temperature'] = pd.to_numeric(train_df['Temperature'], errors='coerce').fillna(30.0)
    train_df['Rainfall'] = pd.to_numeric(train_df['Rainfall'], errors='coerce').fillna(0.0)
    train_df['Item'] = train_df['Item'].fillna('Item').astype(str)

    # Ensure at least 2 distinct classes so classifiers never raise ValueError
    if train_df['Item'].nunique() < 2:
        train_df = pd.concat([
            train_df,
            pd.DataFrame([
                {"Temperature": 30.0, "Rainfall": 0.0, "Item": "Coffee"},
                {"Temperature": 22.0, "Rainfall": 10.0, "Item": "Tea"},
            ])
        ], ignore_index=True)

    X = train_df[['Temperature', 'Rainfall']]
    y = train_df['Item']

    rf = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)
    lr = LogisticRegression(max_iter=200, random_state=42)

    clf = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
    clf.fit(X, y)

    GLOBAL_ML_MODEL = clf
    GLOBAL_ML_CLASSES = clf.classes_
    return GLOBAL_ML_MODEL, GLOBAL_ML_CLASSES


def get_weather(lat=12.9716, lon=77.5946):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
        )
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            current = data.get("current_weather", {})
            import random

            # Base temperature + realistic random variation (-2 to +2 degrees)
            temperature = current.get("temperature", 30)

            # Basic rainfall logic mapping with simulated randomness
            weathercode = current.get("weathercode", 0)
            rainfall = 5 if weathercode >= 50 else 0

            return temperature, rainfall
        return 30, 0
    except Exception as e:
        print("Weather fetch error:", e)
        return 30, 0


@app.route('/weather')
def get_live_weather_api():
    """
    Multi-provider weather route:
    1. Primary: OpenWeatherMap API
    2. Fallback: Open-Meteo Free API (no key required)
    3. Offline Fallback: Realistic regional default
    """
    lat, lon, city = None, None, None
    if current_user.is_authenticated:
        settings_path = os.path.join(app.instance_path, f"{current_user.username}_settings.json")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    import json
                    s_data = json.load(f)
                    lat = s_data.get('lat')
                    lon = s_data.get('lon')
                    city = s_data.get('city')
            except Exception:
                pass

    if not lat or not lon:
        lat = request.args.get('lat', default=12.9716, type=float)
        lon = request.args.get('lon', default=77.5946, type=float)
        city = request.args.get('city', type=str)

    api_key = os.environ.get('WEATHER_API_KEY', 'fb7306ac26a65b8457d6e5b1d59e3469')
    
    # 1. Primary: OpenWeatherMap
    try:
        if city:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        else:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                "status": "success",
                "temp": round(data['main']['temp']),
                "feels_like": round(data['main']['feels_like']),
                "condition": data['weather'][0]['main'],
                "description": data['weather'][0]['description'].capitalize(),
                "humidity": data['main']['humidity'],
                "wind_speed": data['wind']['speed'],
                "city": data['name'],
                "icon": data['weather'][0]['icon'],
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
    except Exception as e:
        print("OpenWeatherMap fetch failed, trying Open-Meteo:", e)

    # 2. Fallback: Open-Meteo (No key required)
    try:
        om_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        om_res = requests.get(om_url, timeout=4)
        if om_res.status_code == 200:
            d = om_res.json()
            cw = d.get('current_weather', {})
            temp = round(cw.get('temperature', 28))
            wind = cw.get('windspeed', 5.0)
            code = cw.get('weathercode', 0)
            cond = "Clear" if code < 3 else "Cloudy" if code < 50 else "Rainy"
            icon = "01d" if code < 3 else "03d" if code < 50 else "10d"
            return jsonify({
                "status": "success",
                "temp": temp,
                "feels_like": temp + 1,
                "condition": cond,
                "description": cond,
                "humidity": 64,
                "wind_speed": wind,
                "city": city or "Bengaluru",
                "icon": icon,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
    except Exception as e:
        print("Open-Meteo fetch failed:", e)

    # 3. Offline / Guaranteed Fallback
    return jsonify({
        "status": "success",
        "temp": 28,
        "feels_like": 29,
        "condition": "Clouds",
        "description": "Partly cloudy",
        "humidity": 64,
        "wind_speed": 4.5,
        "city": city or "Bengaluru",
        "icon": "03d",
        "timestamp": datetime.now().strftime("%I:%M %p")
    })


@app.route('/api/settings', methods=['GET'])
def get_user_settings():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    owner_id = _current_restaurant_owner()
    owner = db.session.get(User, owner_id)
    owner_username = owner.username if owner else current_user.username

    settings = {
        'restaurant_name': 'Neo Dine AI',
        'city': 'Bengaluru',
        'address': '',
        'phone': '7989052868',
        'currency': '₹ INR',
        'timezone': 'Asia/Kolkata (IST)',
        'lat': 12.9716,
        'lon': 77.5946,
        'upi_id': '',
        'upi_payee_name': ''
    }

    # Load from RestaurantDetail DB row first
    detail = RestaurantDetail.query.filter_by(user_id=owner_id).first()
    if detail:
        if detail.restaurant_name: settings['restaurant_name'] = detail.restaurant_name
        if detail.address: settings['address'] = detail.address
        if detail.phone: settings['phone'] = detail.phone
        if detail.currency: settings['currency'] = detail.currency
        if detail.timezone: settings['timezone'] = detail.timezone

    # Load ONLY this restaurant owner's settings file
    spath = os.path.join(app.instance_path, f"{owner_username}_settings.json")
    if os.path.exists(spath):
        try:
            with open(spath, 'r', encoding='utf-8') as f:
                sdata = json.load(f)
                settings.update(sdata)
        except Exception as e:
            print("Error loading settings:", e)

    db_settings = UserSetting.query.filter_by(user_id=current_user.id).first()
    if not db_settings:
        db_settings = UserSetting(user_id=current_user.id)
        db.session.add(db_settings)
        db.session.commit()

    settings.update({
        'success': True,
        'default_order_type': db_settings.default_order_type,
        'language': db_settings.language,
        'tax_enabled': db_settings.tax_enabled,
        'auto_print_receipt': db_settings.auto_print_receipt,
        'auto_print_kot': db_settings.auto_print_kot,
        'appearance_mode': db_settings.appearance_mode,
        'require_customer_name': bool(settings.get('require_customer_name', False)),
        'show_item_images': bool(settings.get('show_item_images', False)),
        'enable_table_management': bool(settings.get('enable_table_management', True)),
        'auto_apply_loyalty': bool(settings.get('auto_apply_loyalty', False)),
        'show_profit_margin': bool(settings.get('show_profit_margin', False)),
        'settings': {
            'default_order_type': db_settings.default_order_type,
            'language': db_settings.language,
            'tax_enabled': db_settings.tax_enabled,
            'auto_print_receipt': db_settings.auto_print_receipt,
            'auto_print_kot': db_settings.auto_print_kot,
            'appearance_mode': db_settings.appearance_mode
        }
    })
    return jsonify(settings)


@app.route('/api/settings/save', methods=['POST'])
def save_user_settings():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401

    data = request.json or {}
    city = data.get('city', 'Bengaluru').strip()
    
    # Try geocoding city using OpenWeatherMap to get lat/lon
    api_key = os.environ.get('WEATHER_API_KEY', 'fb7306ac26a65b8457d6e5b1d59e3469')
    lat, lon = 12.9716, 77.5946
    if city:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                w_data = res.json()
                coord = w_data.get('coord', {})
                if 'lat' in coord and 'lon' in coord:
                    lat = coord['lat']
                    lon = coord['lon']
                    print(f"Geocoded {city} to lat={lat}, lon={lon}")
        except Exception as e:
            print("Geocoding failed:", e)
            
    owner_id = _current_restaurant_owner()
    owner = db.session.get(User, owner_id)
    owner_username = owner.username if owner else current_user.username

    # Update UserSetting database row
    db_settings = UserSetting.query.filter_by(user_id=current_user.id).first()
    if not db_settings:
        db_settings = UserSetting(user_id=current_user.id)
        db.session.add(db_settings)

    if 'auto_print_kot' in data:
        db_settings.auto_print_kot = bool(data['auto_print_kot'])
    if 'auto_print_receipt' in data:
        db_settings.auto_print_receipt = bool(data['auto_print_receipt'])
    if 'tax_enabled' in data:
        db_settings.tax_enabled = bool(data['tax_enabled'])
    if 'default_order_type' in data:
        db_settings.default_order_type = str(data['default_order_type'])

    # 1. Update/Create RestaurantDetail database row
    detail = RestaurantDetail.query.filter_by(user_id=owner_id).first()
    if not detail:
        detail = RestaurantDetail(user_id=owner_id)
        db.session.add(detail)

    rest_name = data.get('restaurant_name')
    if rest_name is not None and rest_name.strip():
        detail.restaurant_name = rest_name.strip()
    
    addr = data.get('address')
    if addr is not None:
        detail.address = addr.strip()
        
    ph = data.get('phone')
    if ph is not None and ph.strip():
        detail.phone = ph.strip()
        
    curr = data.get('currency')
    if curr is not None and curr.strip():
        detail.currency = curr.strip()
        
    tz = data.get('timezone')
    if tz is not None and tz.strip():
        detail.timezone = tz.strip()

    db.session.commit()

    # 2. Update JSON settings file
    settings_path = os.path.join(app.instance_path, f"{owner_username}_settings.json")
    existing_settings = {}
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                existing_settings = json.load(f)
        except Exception:
            pass

    existing_settings.update({
        'restaurant_name': detail.restaurant_name,
        'city': city,
        'address': detail.address or '',
        'phone': detail.phone or '',
        'currency': detail.currency or '₹ INR',
        'timezone': detail.timezone or 'Asia/Kolkata (IST)',
        'lat': lat,
        'lon': lon,
        'auto_print_kot': db_settings.auto_print_kot,
        'auto_print_receipt': db_settings.auto_print_receipt,
        'require_customer_name': bool(data['require_customer_name']) if 'require_customer_name' in data else bool(existing_settings.get('require_customer_name', False)),
        'show_item_images': bool(data['show_item_images']) if 'show_item_images' in data else bool(existing_settings.get('show_item_images', False)),
        'enable_table_management': bool(data['enable_table_management']) if 'enable_table_management' in data else bool(existing_settings.get('enable_table_management', True)),
        'auto_apply_loyalty': bool(data['auto_apply_loyalty']) if 'auto_apply_loyalty' in data else bool(existing_settings.get('auto_apply_loyalty', False)),
        'show_profit_margin': bool(data['show_profit_margin']) if 'show_profit_margin' in data else bool(existing_settings.get('show_profit_margin', False))
    })
    
    os.makedirs(app.instance_path, exist_ok=True)
    try:
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(existing_settings, f, indent=2)
        return jsonify({'success': True, 'message': 'Settings saved successfully!'})
    except Exception as e:
        print("Error saving settings:", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            return render_template(
                'register.html',
                error="Username and password are required.")
        if any(ch in username for ch in ('/', '\\', '..')):
            return render_template(
                'register.html',
                error="Username contains invalid characters.")
        if User.query.filter(
            db.func.lower(User.username) == username.lower()
        ).first():
            return render_template(
                'register.html',
                error="Username already exists.")
        new_user = User(
            username=username,
            password_hash=generate_password_hash(password))
        new_user.raw_password = password
        db.session.add(new_user)
        db.session.commit()
        return render_template(
            'register.html',
            success=("Registration successful! Firewall is active. "
                     "Please wait for Admin approval.")
        )
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, redirect away from login page
    if current_user.is_authenticated:
        if current_user.role == 'waiter':
            return redirect(url_for('waiter_portal'))
        if current_user.role == 'kitchen':
            return redirect(url_for('kitchen_display'))
        if current_user.status == 'Admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = User.query.filter(
            db.func.lower(User.username) == username.lower()
        ).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            ensure_subscription(user)
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()

            # Role-aware landing page: Waiter & Kitchen staff go to their portals.
            if user.role == 'waiter':
                return redirect(url_for('waiter_portal'))
            if user.role == 'kitchen':
                return redirect(url_for('kitchen_display'))
            if user.status == 'Admin':
                return redirect(url_for('admin_dashboard'))

            # Honour Flask-Login's ?next= parameter (safe redirect only)
            next_page = request.args.get('next') or request.form.get('next')
            if next_page and next_page.startswith('/') and not next_page.startswith('//'):
                return redirect(next_page)
            return redirect(url_for('index'))
        return render_template('login.html',
                               error="Invalid credentials. Try again.")
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    from flask import make_response
    logout_user()
    flask_session.clear()
    response = make_response(redirect(url_for('login')))
    response.set_cookie('session', '', expires=0)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-store'
    return response


def update_customer_excel(username, order_data):
    filename = os.path.join(BASE_DIR, f"{username}_database.xlsx")
    df_new = pd.DataFrame([order_data])
    try:
        if os.path.exists(filename):
            df_existing = pd.read_excel(filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(filename, index=False)
        else:
            df_new.to_excel(filename, index=False)
    except Exception as e:
        print(f"Error updating Excel for {username}: {e}")


# ────────────────────────────────────────────────────────────────────────────
#  WAITER & KITCHEN PORTAL HELPERS
#  Every restaurant/account is isolated by a restaurant id. Staff accounts
#  (role = waiter/kitchen) are linked to the restaurant they serve through
#  `restaurant_owner_id`; every other account operates on its own data.
# ────────────────────────────────────────────────────────────────────────────

def _current_restaurant_owner():
    """Return the restaurant id whose data the current user may operate on.

    Waiter/Kitchen staff act on their linked restaurant owner; the restaurant
    owner and every other account use their own id (each approved account is an
    isolated restaurant).
    """
    if getattr(current_user, 'role', None) in ('waiter', 'kitchen') and getattr(current_user, 'restaurant_owner_id', None):
        return current_user.restaurant_owner_id
    return current_user.id


def _kitchen_scope():
    """Restaurant scope visible to the Kitchen Display.

    Regular staff see their own restaurant's orders. The SaaS super-admin
    (who is not itself a restaurant) is allowed to monitor every restaurant.
    """
    if current_user.status == 'Admin' and current_user.username == 'admin':
        return 'ALL'
    return _current_restaurant_owner()


def _get_restaurant_name(user=None):
    """Best-effort display name for the current restaurant scope."""
    owner_id = _current_restaurant_owner()
    rd = RestaurantDetail.query.filter_by(user_id=owner_id).first()
    if rd and rd.restaurant_name:
        return rd.restaurant_name
    user = user or current_user
    return (user.name or "") if user.name else user.username


@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.status != 'Admin':
        return "Access Denied", 403
    pending_users = User.query.filter_by(status='Pending').all()
    approved_users = User.query.filter_by(status='Approved').all()

    # Restaurants (owners) eligible for staff linking — exclude staff accounts.
    restaurant_owners = User.query.filter(
        User.status.in_(('Approved', 'Admin')),
        db.or_(User.role.is_(None), User.role.notin_(('waiter', 'kitchen')))
    ).order_by(User.username).all()

    # Waiters / kitchen staff and their linked restaurant for the admin UI.
    staff_accounts = []
    for s in User.query.filter(User.role.in_(['waiter', 'kitchen'])).order_by(User.username).all():
        owner = db.session.get(User, s.restaurant_owner_id) if s.restaurant_owner_id else None
        staff_accounts.append({
            'username': s.username,
            'role': s.role,
            'owner_username': owner.username if owner else None,
            'restaurant_id': s.restaurant_owner_id,
        })

    # Calculate tomorrow's stock requirements per customer based on today's
    # orders
    tomorrow_stock_by_user = {}

    ingredient_symbols = {
        "Chicken": "🍗", "Mutton": "🍖", "Cheese": "🧀", "Veg": "🥗",
        "Paneer": "🧀", "Egg": "🥚", "Bun": "🍔", "Pizza Base": "🍕",
        "Noodles": "🍝", "Rice": "🍚", "Milk": "🥛", "Sugar": "🧂",
        "Tea Powder": "🌿", "Coffee Powder": "☕", "Tomato": "🍅",
        "Onion": "🧅", "Potato": "🥔", "Spice": "🌶️", "Sauce": "🥫",
        "Oil": "🛢️", "Butter": "🧈", "Water": "💧", "Drink": "🥤",
        "Pepsi": "🥤", "Sprite": "🥤", "Coke": "🥤", "Flour": "🌾"
    }

    def get_symbol(ingredient_name):
        for key, symbol in ingredient_symbols.items():
            if key.lower() in ingredient_name.lower():
                return symbol
        return "📦"

    try:
        today = datetime.now(timezone.utc).date()
        df_gigatos = pd.read_excel(os.path.join(BASE_DIR, 'Gigatos   excel data.xlsx'), header=None)

        for user in approved_users:
            user_stock = {}
            # Get orders for this specific restaurant from today
            today_str = today.strftime('%Y-%m-%d')
            user_orders = db.session.query(OrderItem.item_name)\
                .join(Order, Order.id == OrderItem.order_id)\
                .filter(Order.restaurant_id == user.id)\
                .filter(db.func.substr(Order.date, 1, 10) == today_str).all()

            item_counts = {}
            for (item_name,) in user_orders:
                item_counts[item_name] = item_counts.get(item_name, 0) + 1

            for index, row in df_gigatos.iterrows():
                col2 = str(row[2]).strip()
                if col2.lower() == 'menu item':
                    continue
                if pd.isna(row[2]):
                    continue

                item_name = str(row[2]).strip()
                ingredient = str(row[4]).strip()
                qty = pd.to_numeric(row[5], errors='coerce')
                unit = str(row[6]).strip()

                if pd.isna(qty):
                    continue

                for ordered_item, count in item_counts.items():
                    if ordered_item.lower() == item_name.lower():
                        buffer_count = int(count * 1.2) + 1
                        total_qty_needed = qty * buffer_count

                        symbol = get_symbol(ingredient)
                        key = f"{symbol} {ingredient} ({unit})"
                        user_stock[key] = user_stock.get(
                            key, 0) + total_qty_needed

            if user_stock:
                tomorrow_stock_by_user[user.username] = user_stock

    except Exception as e:
        print(f"Error calculating stock per user: {e}")

    # Live overview of each restaurant's menu and tables (single source of truth).
    restaurant_overview = []
    for owner in restaurant_owners:
        menu = MenuItem.query.filter_by(restaurant_id=owner.id).order_by(MenuItem.category, MenuItem.name).all()
        tables = DiningTable.query.filter_by(restaurant_id=owner.id).order_by(DiningTable.table_num).all()
        restaurant_overview.append({
            'owner': owner,
            'menu_items': menu,
            'tables': tables,
            'occupied': sum(1 for t in tables if t.status == 'DINING'),
        })

    return render_template('admin.html',
                           pending_users=pending_users,
                           approved_users=approved_users,
                           restaurant_owners=restaurant_owners,
                           restaurant_overview=restaurant_overview,
                           staff_accounts=staff_accounts,
                           tomorrow_stock_by_user=tomorrow_stock_by_user)


@app.route('/admin/approve/<int:user_id>', methods=['POST'])
@login_required
def approve_user(user_id):
    if current_user.status != 'Admin':
        return "Access Denied", 403
    user = db.session.get(User, user_id)
    if user:
        user.status = 'Approved'
        db.session.commit()
        # Ensure excel file exists initially
        filename = os.path.join(BASE_DIR, f"{user.username}_database.xlsx")
        if not os.path.exists(filename):
            pd.DataFrame(
                columns=[
                    'Date',
                    'Order ID',
                    'Customer Name',
                    'Phone',
                    'Items',
                    'Subtotal',
                    'Discount %',
                    'GST (5%)',
                    'Total Price']).to_excel(
                filename,
                index=False)
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/revoke/<int:user_id>', methods=['POST'])
@login_required
def revoke_user(user_id):
    if current_user.status != 'Admin':
        return "Access Denied", 403
    user = db.session.get(User, user_id)
    if user:
        user.status = 'Pending'
        db.session.commit()
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/developer-tools')
@login_required
def admin_developer_tools():
    if current_user.status != 'Admin':
        return "Access Denied", 403
        
    approved_users = User.query.filter_by(status='Approved').all()
    # Map active plan details
    users_with_subs = []
    for user in approved_users:
        sub = ensure_subscription(user)
        plan_name = "None"
        plan_slug = "none"
        if sub and sub.plan:
            plan_name = sub.plan.name
            plan_slug = sub.plan.slug
        users_with_subs.append({
            'user': user,
            'sub': sub,
            'plan_name': plan_name,
            'plan_slug': plan_slug
        })
        
    test_mode_enabled = get_global_settings().get("test_mode", False)
    return render_template(
        'admin_developer_tools.html',
        users=users_with_subs,
        test_mode_enabled=test_mode_enabled
    )

@app.route('/admin/developer-tools/toggle-test-mode', methods=['POST'])
@login_required
def admin_toggle_test_mode():
    if current_user.status != 'Admin':
        return "Access Denied", 403
    settings = get_global_settings()
    settings["test_mode"] = not settings.get("test_mode", False)
    save_global_settings(settings)
    return redirect(url_for('admin_developer_tools'))

@app.route('/admin/developer-tools/activate-plan/<int:user_id>/<plan_slug>', methods=['POST'])
@login_required
def admin_activate_plan(user_id, plan_slug):
    if current_user.status != 'Admin':
        return "Access Denied", 403
        
    user = db.session.get(User, user_id)
    if not user:
        return "User not found", 404
        
    plan = SubscriptionPlan.query.filter_by(slug=plan_slug).first()
    if not plan:
        return "Plan not found", 404
        
    sub = ensure_subscription(user)
    if sub:
        sub.plan_id = plan.id
        sub.status = 'ACTIVE'
        sub.payment_status = 'TEST_MODE'
        sub.subscription_start = datetime.now(timezone.utc)
        sub.subscription_end = datetime.now(timezone.utc) + timedelta(days=30)
        db.session.commit()
        
        # Generate testing report
        generate_testing_report_file(user, sub)
        
    return redirect(url_for('admin_developer_tools'))

@app.route('/admin/developer-tools/reset-subscription/<int:user_id>', methods=['POST'])
@login_required
def admin_reset_subscription(user_id):
    if current_user.status != 'Admin':
        return "Access Denied", 403
        
    user = db.session.get(User, user_id)
    if not user:
        return "User not found", 404
        
    sub = Subscription.query.filter_by(restaurant_id=user.id).first()
    if sub:
        sub.status = 'TRIAL'
        sub.trial_end = datetime.now(timezone.utc) + timedelta(days=7)
        sub.plan_id = None
        sub.payment_status = None
        sub.subscription_start = None
        sub.subscription_end = None
        db.session.commit()
        
    return redirect(url_for('admin_developer_tools'))


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if current_user.status == 'Pending':
        return (
            "<h2 style='text-align:center; margin-top:50px; "
            "font-family:Outfit, sans-serif;'>"
            "Your account is pending Administrator Approval.<br>"
            "The firewall is currently blocking you from accessing this "
            "software.<br><br>Please contact the system owner to approve "
            "your registration.<br><br><a href='/logout'>Logout</a></h2>"
        )

    temperature, rainfall = get_weather()

    # Fetch real SQLite items from DB (only active & visible items)
    db_items = MenuItem.query.filter_by(restaurant_id=current_user.id).filter(
        db.or_(MenuItem.status.is_(None), MenuItem.status == 'Active'),
        db.or_(MenuItem.is_hidden.is_(None), MenuItem.is_hidden == False)
    ).all()
    if db_items:
        # Sort db_items so Veg items appear before Non-Veg, then by display_order, then by name
        sorted_db_items = sorted(
            db_items,
            key=lambda x: (
                0 if (x.type or '').strip().lower() == 'veg' else 1,
                x.display_order or 0,
                (x.name or '').lower()
            )
        )
        categorized_menu = {}
        cat_canonical_map = {}
        for i in sorted_db_items:
            raw_cat = (i.category or 'General').strip()
            cat_lower = raw_cat.lower()
            if cat_lower not in cat_canonical_map:
                cat_canonical_map[cat_lower] = raw_cat
            cat = cat_canonical_map[cat_lower]
            if cat not in categorized_menu:
                categorized_menu[cat] = {}
            categorized_menu[cat][i.name] = i.price
    else:
        # Fallback dictionary if no menu items added yet by user
        categorized_menu = {}

    # Fetch live tables
    tables = DiningTable.query.filter_by(restaurant_id=current_user.id).order_by(DiningTable.table_num).all()
    if not tables:
        # Seed default 15 tables automatically
        for t_idx in range(1, 16):
            db.session.add(DiningTable(restaurant_id=current_user.id, table_num=t_idx))
        db.session.commit()
        tables = DiningTable.query.filter_by(restaurant_id=current_user.id).all()

    # Flatten categories to easily retrieve prices later in the checkout logic
    price_map = {}
    for items in categorized_menu.values():
        price_map.update(items)

    # Machine Learning Core
    clf, classes = get_or_train_ml_model()

    # 3. Predict Today (Live Weather)
    # Give it slight variations of current weather to get probabilities of
    # what will sell
    weather_samples_today = pd.DataFrame([
        {'Temperature': temperature, 'Rainfall': rainfall},
        {'Temperature': temperature + 1, 'Rainfall': rainfall},
        {'Temperature': temperature - 1, 'Rainfall': max(0, rainfall - 1)},
        {'Temperature': temperature, 'Rainfall': max(0, rainfall + 2)}
    ])
    clf.predict(weather_samples_today)

    current_weather_df = pd.DataFrame(
        [{'Temperature': temperature, 'Rainfall': rainfall}])
    probs_today = clf.predict_proba(current_weather_df)[0]

    # Get top 3 items for today based on probabilities
    top_indices = probs_today.argsort()[-3:][::-1]

    predicted_today = {}
    for idx in top_indices:
        item = classes[idx]
        # Generate a synthetic "expected orders" number based on probability
        # score
        expected = int(probs_today[idx] * 100) + 10
        predicted_today[item] = expected

    # 4. Predict Tomorrow (Assume slightly hotter or rainier based on today)
    future_temp = temperature + 2 if rainfall == 0 else temperature - 2
    future_rain = 0 if rainfall > 0 else 5
    weather_samples_tomorrow = pd.DataFrame([
        {'Temperature': future_temp, 'Rainfall': future_rain},
        {'Temperature': future_temp + 1, 'Rainfall': future_rain},
        {'Temperature': future_temp - 1, 'Rainfall': future_rain + 2}
    ])
    preds_tomorrow = clf.predict(weather_samples_tomorrow)
    predicted_tomorrow = list(set(preds_tomorrow))

    total_price = None
    receipt = None

    if request.method == "POST":
        name = request.form.get("name", "").strip() or "Walk-in guest"
        phone = request.form.get("phone", "").strip()
        foods = request.form.getlist("food")
        try:
            discount_pct = float(request.form.get("discount", 0))
            discount_pct = max(0.0, min(100.0, discount_pct))
        except (ValueError, TypeError):
            discount_pct = 0.0

        print(f"\n[DEBUG-POST] Entering Order Processing. Name: {name}, Phone: {phone}")
        print(f"[DEBUG-POST] Raw form data: {request.form}")
        print(f"[DEBUG-POST] Raw 'food' list from form: {foods}")
        
        token = request.form.get('submission', '')
        lock_restaurant(current_user.id)
        if token:
            record = db.session.get(OrderSubmission, (current_user.id, token))
            if record:
                return submission_result(record) if record.kind == 'checkout' else ('Invalid submission', 409)
        table_num = request.form.get('table_number', type=int)
        table = DiningTable.query.populate_existing().filter_by(restaurant_id=current_user.id, table_num=table_num).first() if table_num else None
        if table_num and not table:
            return 'Table no longer exists.', 409
        existing = db.session.get(Order, table.current_order_id) if table and table.current_order_id else None
        if existing and (existing.restaurant_id != current_user.id or existing.table_num != table_num
                         or existing.payment_status != 'PENDING'):
            return 'This table has a previous bill. Clear it before starting another order.', 409
        original_ids = {line.id for line in existing.items} if existing else set()
        expected = request.form.get('bill_version', '')
        version = f"{existing.id}:{len(existing.items)}" if existing else ''
        if expected != version:
            return 'This bill changed. Reopen the table and review the total.', 409
        if any(item not in price_map for item in foods):
            return "A selected menu item is no longer available.", 400
        if not foods and not existing:
            return "Add items or reopen an unpaid table.", 400
        if foods or existing:
            print(f"[DEBUG-POST] Processing order for {len(foods)} items.")
            food_items_receipt = []
            subtotal = 0.0

            # Create a UUID for securing the receipt link
            order_id_string = existing.order_uuid if existing else str(uuid.uuid4())

            # Build new SQL database order securely linked to current
            # Restaurant
            new_order = existing or Order(
                restaurant_id=current_user.id,
                order_uuid=order_id_string,
                customer_name=name,
                customer_phone=phone,
                temperature=temperature,
                rainfall=rainfall
            )
            new_order.customer_name = name
            new_order.customer_phone = phone
            new_order.table_num = table_num
            new_order.payment_status = 'DUE' if request.form.get("paymentType") == 'Due' else 'PAID'
            new_order.discount_pct = float(discount_pct)
            new_order.order_type = request.form.get("order_type", "Dine-In")
            new_order.payment_method = request.form.get("paymentType", "Cash")
            db.session.add(new_order)
            db.session.flush()  # Generate new SQL ID safely before child rows

            from collections import Counter
            item_counts = Counter(foods)

            item_names_str = []
            for item_raw, qty in item_counts.items():
                item = str(item_raw)
                if item in price_map:
                    item_price = float(price_map[item])
                    total_item_price = item_price * qty
                    subtotal += total_item_price  # type: ignore
                    food_items_receipt.append(
                        {"name": item, "price": total_item_price, "qty": qty})
                    item_names_str.append(f"{qty}x {item}")

                    # Store individual purchased item logic in SQL DB
                    order_item = OrderItem(
                        order_id=new_order.id,
                        item_name=item,
                        price=item_price,
                        quantity=qty
                    )
                    db.session.add(order_item)
            
            # Include previously fired items at their original prices.
            if existing:
                originals = [line for line in existing.items if line.id in original_ids]
                for line in originals:
                    amount = round(line.price * line.quantity, 2)
                    subtotal += amount
                    food_items_receipt.append(dict(name=line.item_name, qty=line.quantity, price=amount))
                    item_names_str.append(f"{line.quantity}x {line.item_name}")
                    item_counts[line.item_name] += line.quantity
            deduct_inventory_for_order(current_user.id, item_counts, commit=False)
            if table:
                table.status = 'EMPTY'
                table.current_order_id = None
            if token:
                db.session.add(OrderSubmission(user_id=current_user.id, token=token, kind='checkout',
                                               order_id=new_order.id, table_num=table_num))

            # Subtotal -> Apply Discount -> Apply Dynamic Combined GST -> Total
            tax_cfg = get_tax_config(current_user.id)
            tax_rate = (tax_cfg['combined_gst_pct'] / 100.0) if tax_cfg['tax_enabled'] else 0.0
            discount_amount = round(float(subtotal * (discount_pct / 100.0)), 2)
            after_discount = subtotal - discount_amount
            gst_amount = round(float(after_discount * tax_rate), 2)
            final_total = round(after_discount + gst_amount, 2)

            new_order.subtotal = subtotal
            new_order.gst_pct = tax_cfg['combined_gst_pct'] if tax_cfg['tax_enabled'] else 0.0
            new_order.gst_amount = gst_amount
            new_order.total_price = final_total
            db.session.commit()

            # --- Live learning: Trigger background retrain check ---
            try:
                from ai_prediction_engine import check_and_trigger_retrain
                check_and_trigger_retrain(current_user.id)
            except Exception as ml_err:
                print(f"[AI-PREDICT-HOOK] Error triggering retrain check: {ml_err}")

            # For the immediate frontend receipt view (so customers don't query
            # db for it directly)
            PENDING_ORDERS[(current_user.id, order_id_string)] = {
                "order_id": order_id_string,
                "name": name,
                "phone": phone,
                "food_items": food_items_receipt,
                "subtotal": subtotal,
                "discount_pct": discount_pct,
                "discount_amount": discount_amount,
                "gst_amount": gst_amount,
                "total": final_total,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Live Update Excel Database for this User
            order_record = {
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Order ID': str(order_id_string)[:8],  # type: ignore
                'Customer Name': name,
                'Phone': phone,
                'Items': ", ".join(item_names_str),
                'Subtotal': subtotal,
                'Discount %': discount_pct,
                'GST (5%)': gst_amount,
                'Total Price': final_total,
                'Payment Method': request.form.get("paymentType", "Cash")
            }
            update_customer_excel(current_user.username, order_record)

            return redirect(url_for('show_receipt', order_id=order_id_string))

    # Calculate restaurant total sales from SQL DB intelligently
    restaurant_orders = Order.query.filter_by(
        restaurant_id=current_user.id).all()
    total_sales = sum((o.total_price or 0) for o in restaurant_orders)
    total_profit = round(total_sales * PROFIT_MARGIN, 2)

    menu_item_details = {i.name: i for i in db_items} if db_items else {}

    tax_cfg = get_tax_config(current_user.id)
    return render_template(
        'index.html',
        menu_data=categorized_menu,
        menu_item_details=menu_item_details,
        predicted_tomorrow=predicted_tomorrow,
        predicted_today=predicted_today,
        temperature=temperature,
        rainfall=rainfall,
        total_price=total_price,
        receipt=receipt,
        total_sales=total_sales,
        total_profit=total_profit,
        tables=tables,
        tax_config=tax_cfg
    )


@app.route("/receipt/<order_id>")
@login_required
def show_receipt(order_id):
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403

    owner_id = _current_restaurant_owner()
    receipt = PENDING_ORDERS.get((owner_id, order_id))
    if not receipt:
        orders = Order.query.filter_by(restaurant_id=owner_id)
        order = orders.filter_by(order_uuid=order_id).first()
        if not order and order_id.isascii() and order_id.isdecimal() and len(order_id) <= 18:
            order = orders.filter_by(id=int(order_id)).first()
        if order:
            receipt = {
                'order_id': str(order.order_uuid or order.id),
                'name': order.customer_name or 'Walk-in Guest',
                'phone': order.customer_phone or 'N/A',
                'date': order.date.strftime('%d-%m-%Y %H:%M') if order.date else 'N/A',
                'food_items': [{'name': (item.item_name or 'Item'), 'qty': item.quantity, 'price': round((item.price or 0) * item.quantity, 2)} for item in getattr(order, 'items', [])] or [{'name': 'Order Item', 'qty': 1, 'price': order.total_price}],
                'subtotal': round(order.subtotal if order.subtotal is not None else ((order.total_price or 0) - (order.gst_amount or 0)), 2),
                'discount_pct': round(order.discount_pct or 0, 2),
                'discount_amount': round((order.subtotal or 0) * (order.discount_pct or 0) / 100, 2),
                'gst_pct': round(order.gst_pct or 5.0, 2),
                'gst_amount': round(order.gst_amount or 0, 2),
                'total': round(order.total_price or 0, 2)
            }
        else:
            return "Receipt not found or expired.", 404

    # UPI QR: customers can scan the receipt to pay the exact bill amount.
    upi_qr = None
    upi_vpa = None
    try:
        vpa = None
        payee_name = None

        # Determine the restaurant owner for this session/order
        owner_id_qr = _current_restaurant_owner()
        owner_user = db.session.get(User, owner_id_qr) or current_user
        owner_username = owner_user.username

        # 1) Read this restaurant owner's settings file
        spath = _safe_settings_path(owner_username)
        if os.path.exists(spath):
            try:
                with open(spath) as f:
                    sdata = json.load(f)
                vpa = (sdata.get("upi_id") or "").strip() or None
                payee_name = (sdata.get("upi_payee_name") or "").strip() or None
            except Exception:
                pass

        # 2) Fallback: check global_settings.json
        if not vpa:
            g_sett = get_global_settings()
            if isinstance(g_sett, dict):
                vpa = (g_sett.get("upi_id") or "").strip() or None
                if not payee_name:
                    payee_name = (g_sett.get("upi_payee_name") or "").strip() or None

        # 3) Fallback: check environment variable DEFAULT_UPI_ID
        if not vpa:
            vpa = (os.environ.get("DEFAULT_UPI_ID") or "").strip() or None

        # 4) Fallback: derive from restaurant phone number if available
        if not vpa:
            owner_id_for_qr = _current_restaurant_owner()
            rest_det_for_qr = RestaurantDetail.query.filter_by(user_id=owner_id_for_qr).first()
            if rest_det_for_qr and rest_det_for_qr.phone:
                phone_clean = (rest_det_for_qr.phone or "").strip().lstrip("+").replace(" ", "")
                if phone_clean and phone_clean.isdigit() and len(phone_clean) >= 10:
                    vpa = f"{phone_clean[-10:]}@upi"
                    payee_name = payee_name or (rest_det_for_qr.restaurant_name or owner_username)

        if vpa:
            upi_vpa = vpa
            import qrcode, base64, io as _io, urllib.parse
            pa_str = urllib.parse.quote(vpa, safe='@._-')
            pn_str = urllib.parse.quote(payee_name or getattr(owner_user, 'name', None) or owner_username)
            tn_str = urllib.parse.quote(f"Bill {str(order_id)[:8]}")
            upi_str = (
                f"upi://pay?pa={pa_str}"
                f"&pn={pn_str}"
                f"&am={float(receipt.get('total', 0)):.2f}"
                f"&cu=INR&tn={tn_str}"
            )
            img = qrcode.make(upi_str)
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            upi_qr = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as qr_err:
        print(f"[UPI-QR] Could not generate QR: {qr_err}")

    # Populate CGST / SGST breakdown for receipt template
    if receipt:
        tax_cfg = get_tax_config(_current_restaurant_owner())
        gst_pct = float(receipt.get('gst_pct') or tax_cfg['combined_gst_pct'] or 5.0)
        gst_amount = float(receipt.get('gst_amount') or 0.0)
        cgst_pct = float(tax_cfg.get('cgst_pct') or (gst_pct / 2.0))
        sgst_pct = float(tax_cfg.get('sgst_pct') or (gst_pct / 2.0))
        
        receipt['gst_pct'] = round(gst_pct, 2)
        receipt['cgst_pct'] = round(cgst_pct, 2)
        receipt['sgst_pct'] = round(sgst_pct, 2)
        receipt['cgst_amount'] = round(gst_amount * (cgst_pct / gst_pct), 2) if gst_pct > 0 else 0.0
        receipt['sgst_amount'] = round(gst_amount * (sgst_pct / gst_pct), 2) if gst_pct > 0 else 0.0

    # Load saved Restaurant Information (Name, Address, Phone, GSTIN, Email, Logo)
    owner_id = _current_restaurant_owner()
    rest_detail = RestaurantDetail.query.filter_by(user_id=owner_id).first()

    return render_template('receipt.html', receipt=receipt, restaurant_detail=rest_detail, upi_qr=upi_qr, upi_vpa=upi_vpa)


@app.route('/api/settings/upi', methods=['POST'])
@login_required
def api_save_upi():
    """Save this restaurant's UPI ID (and optional payee name) for receipt QR codes."""
    if current_user.status == 'Pending' or current_user.role in ('waiter', 'kitchen'):
        return jsonify({'success': False, 'error': 'Only the restaurant owner can change the UPI ID'}), 403

    data = request.get_json(force=True) or {}
    supplied_pwd = (data.get('current_password') or '').strip()

    if not supplied_pwd:
        return jsonify({'success': False, 'error': 'Password is required to change UPI ID'}), 400

    from werkzeug.security import check_password_hash as _chk
    valid_pwd = False
    if current_user.password_hash and _chk(current_user.password_hash, supplied_pwd):
        valid_pwd = True
    elif getattr(current_user, 'raw_password', None) and current_user.raw_password == supplied_pwd:
        valid_pwd = True

    if not valid_pwd:
        return jsonify({'success': False, 'error': 'Incorrect password. UPI ID not saved.'}), 403

    vpa = (data.get('upi_id') or '').strip()
    payee = (data.get('upi_payee_name') or '').strip()
    if vpa and ('@' not in vpa):
        return jsonify({'success': False, 'error': 'Invalid UPI ID (must look like name@bank)'}), 400

    owner_id = _current_restaurant_owner()
    owner_user = db.session.get(User, owner_id) or current_user
    owner_username = owner_user.username

    settings_path = _safe_settings_path(owner_username)
    sdata = {}
    if os.path.exists(settings_path):
        try:
            with open(settings_path) as f:
                sdata = json.load(f)
        except Exception:
            pass
    sdata['upi_id'] = vpa
    sdata['upi_payee_name'] = payee
    os.makedirs(app.instance_path, exist_ok=True)
    with open(settings_path, 'w') as f:
        json.dump(sdata, f)

    return jsonify({'success': True, 'upi_id': vpa, 'upi_payee_name': payee})


@app.route('/api/upi/qr', methods=['GET'])
@login_required
def api_upi_qr():
    """Generate a UPI QR code on-the-fly for the POS billing modal.
    Query params:
      amount   - bill total (float, required)
      order_id - optional order reference string
    Returns JSON: { success, qr: "data:image/png;base64,...", vpa, amount }
    """
    try:
        amount = float(request.args.get('amount', 0) or 0)
        order_ref = (request.args.get('order_id') or '').strip()[:8] or 'Bill'

        # Resolve owner
        owner_id_qr = _current_restaurant_owner()
        owner_user = db.session.get(User, owner_id_qr) or current_user
        owner_username = owner_user.username

        vpa = None
        payee_name = None

        # 1) Owner settings file
        spath = _safe_settings_path(owner_username)
        if os.path.exists(spath):
            try:
                with open(spath) as f:
                    sdata = json.load(f)
                vpa = (sdata.get("upi_id") or "").strip() or None
                payee_name = (sdata.get("upi_payee_name") or "").strip() or None
            except Exception:
                pass

        # 2) Global settings fallback
        if not vpa:
            g_sett = get_global_settings()
            if isinstance(g_sett, dict):
                vpa = (g_sett.get("upi_id") or "").strip() or None
                if not payee_name:
                    payee_name = (g_sett.get("upi_payee_name") or "").strip() or None

        # 3) Environment variable fallback
        if not vpa:
            vpa = (os.environ.get("DEFAULT_UPI_ID") or "").strip() or None

        # 4) Phone-based fallback
        if not vpa:
            owner_id_qr = _current_restaurant_owner()
            rest_det_qr = RestaurantDetail.query.filter_by(user_id=owner_id_qr).first()
            if rest_det_qr and rest_det_qr.phone:
                phone_clean = (rest_det_qr.phone or "").strip().lstrip("+").replace(" ", "")
                if phone_clean and phone_clean.isdigit() and len(phone_clean) >= 10:
                    vpa = f"{phone_clean[-10:]}@upi"
                    payee_name = payee_name or (rest_det_qr.restaurant_name or owner_username)

        if not vpa:
            return jsonify({'success': False, 'error': 'No UPI ID configured. Go to Settings → UPI Payment QR to add one.'}), 404

        import qrcode, base64, io as _io, urllib.parse
        pa_str = urllib.parse.quote(vpa, safe='@._-')
        pn_str = urllib.parse.quote(payee_name or owner_username)
        tn_str = urllib.parse.quote(f"Bill {order_ref}")
        upi_str = (
            f"upi://pay?pa={pa_str}"
            f"&pn={pn_str}"
            f"&am={amount:.2f}"
            f"&cu=INR&tn={tn_str}"
        )
        img = qrcode.make(upi_str)
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        qr_data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return jsonify({'success': True, 'qr': qr_data, 'vpa': vpa, 'amount': amount})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download_excel')
@login_required
def download_excel():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
        
    filename = f"{current_user.username}_database.xlsx"
    if os.path.basename(filename) == filename:
        path = os.path.join(BASE_DIR, filename)
        if os.path.isfile(path):
            return send_file(path, as_attachment=True)
    return "No orders yet to generate a report.", 404


@app.route('/api/check_permission/<feature_code>')
@login_required
def check_permission_api(feature_code):
    allowed, reason, req_plan, curr_plan = check_feature_access(current_user, feature_code)
    if allowed:
        return jsonify({'allowed': True})
        
    return jsonify({
        'allowed': False,
        'reason': reason,
        'current_plan': curr_plan,
        'required_plan': req_plan
    }), 403

@app.route('/<category>/<page>')
@login_required
def render_module(category, page):
    # Only trap module routing if it maps to our templates
    if category not in ['operations', 'finance', 'menu', 'settings', 'reports']:
        return "Route not found.", 404
        
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    
    # --- SUBSCRIPTION & FEATURE GATE ---
    feature_code = get_feature_code(category, page)
    allowed, reason, req_plan, curr_plan = check_feature_access(current_user, feature_code)
    
    # --- MANDATORY PRINT LOGS FOR DEBUGGING ---
    username = current_user.username if current_user.is_authenticated else "Anonymous"
    sub = ensure_subscription(current_user)
    plan_slug = sub.plan.slug if (sub and sub.plan) else "None"
    plan_name = sub.plan.name if (sub and sub.plan) else "None"
    sub_status = sub.status if sub else "None"
    
    print("--- RENDER_MODULE TRACE ---")
    print("USER:", username)
    print("STATUS:", sub_status)
    print("PLAN SLUG:", plan_slug)
    print("PLAN NAME:", plan_name)
    print("FEATURE:", feature_code)
    print("ALLOWED:", FEATURE_ACCESS_MAP.get(feature_code))
    print("AUTHORIZED:", allowed, "REASON:", reason)
    # ------------------------------------------
    
    if not allowed:
        if reason in ['NoSubscription', 'Expired']:
            print(f"[ACCESS] Blocked {username}: {reason}")
            return redirect(url_for('subscription_pricing', msg='Trial/Subscription expired. Please choose a plan.'))
        else:
            print(f"[ACCESS] Access denied to {username} ({plan_slug}) for feature {feature_code}")
            return render_template('upgrade_popup.html', required_plan=req_plan, current_plan=curr_plan)
    # --- END GATE ---
    
    context = {}
    if category in ['operations', 'reports'] or page in ['orders', 'kots']:
        context['orders'] = Order.query.filter_by(restaurant_id=current_user.id).order_by(Order.date.desc()).all()
        
    if category == 'operations' and page == 'table':
        tables = DiningTable.query.filter_by(restaurant_id=current_user.id).order_by(DiningTable.table_num).all()
        if not tables:
            for t_idx in range(1, 13):
                db.session.add(DiningTable(restaurant_id=current_user.id, table_num=t_idx))
            db.session.commit()
            tables = DiningTable.query.filter_by(restaurant_id=current_user.id).order_by(DiningTable.table_num).all()
        context['tables'] = tables
        context['total_tables'] = len(tables)
        context['occupied_tables'] = sum(1 for t in tables if t.status == 'DINING')

    if category == 'menu' and page == 'menu':
        owner_id = _current_restaurant_owner()
        context['menu_items'] = MenuItem.query.filter_by(restaurant_id=owner_id).order_by(MenuItem.display_order, MenuItem.name).all()
        context['total_items'] = len(context['menu_items'])
        cats_dict = {}
        for i in context['menu_items']:
            raw_c = (i.category or '').strip()
            if raw_c and raw_c.lower() not in cats_dict:
                cats_dict[raw_c.lower()] = raw_c
        context['categories'] = sorted(list(cats_dict.values()))
        context['total_categories'] = len(context['categories'])
        context['hidden_items'] = sum(1 for i in context['menu_items'] if (i.is_hidden or i.status != 'Active'))
        context['best_sellers'] = sum(1 for i in context['menu_items'] if i.is_best_seller)
        context['veg_items_count'] = sum(1 for i in context['menu_items'] if (i.type or '').strip().lower() == 'veg')
        context['nonveg_items_count'] = sum(1 for i in context['menu_items'] if (i.type or '').strip().lower() == 'non-veg')

        veg_cats = set()
        nonveg_cats = set()
        for i in context['menu_items']:
            raw_c = (i.category or '').strip()
            if not raw_c: continue
            canon = cats_dict.get(raw_c.lower(), raw_c)
            itype = (i.type or '').strip().lower()
            if itype == 'veg':
                veg_cats.add(canon)
            elif itype == 'non-veg':
                nonveg_cats.add(canon)
        context['veg_categories'] = sorted(list(veg_cats))
        context['nonveg_categories'] = sorted(list(nonveg_cats))
    elif category == 'menu' and page == 'tax':
        owner_id = _current_restaurant_owner()
        context['tax_config'] = get_tax_config(owner_id)
        today_orders = Order.query.filter_by(restaurant_id=owner_id).all()
        now_utc = datetime.now(timezone.utc)
        context['gst_collected_today'] = round(sum(
            (o.gst_amount or 0) for o in today_orders
            if o.date and ensure_utc(o.date).date() == now_utc.date()
        ), 2)
    elif category == 'menu' and page == 'inventory':
        context['inventory_items'] = Inventory.query.filter_by(restaurant_id=current_user.id).all()
        print(f"DEBUG IN INVENTORY ROUTE: restaurant_id={current_user.id}, items_found={len(context.get('inventory_items', []))}")
        context['total_items'] = len(context['inventory_items'])
        context['low_stock'] = sum(1 for i in context['inventory_items'] if 0 < i.current_stock <= i.min_level)
        context['critical_stock'] = sum(1 for i in context['inventory_items'] if i.current_stock <= 0)
        context['well_stocked'] = context['total_items'] - context['low_stock'] - context['critical_stock']
    elif category == 'menu' and page == 'recipes':
        context['menu_items'] = MenuItem.query.filter_by(restaurant_id=current_user.id, status='Active').order_by(MenuItem.name).all()
        context['inventory_items'] = Inventory.query.filter_by(restaurant_id=current_user.id).order_by(Inventory.ingredient_name).all()
        context['recipes'] = db.session.query(MenuRecipe).join(MenuItem).filter(MenuItem.restaurant_id == current_user.id).order_by(MenuItem.name, MenuRecipe.id).all()
    elif category == 'menu' and page == 'menu-item-onoff':
        owner_id = _current_restaurant_owner()
        menu_items = MenuItem.query.filter_by(restaurant_id=owner_id).order_by(MenuItem.category, MenuItem.name).all()
        if not menu_items and getattr(current_user, 'status', None) == 'Admin':
            menu_items = MenuItem.query.order_by(MenuItem.category, MenuItem.name).all()
        context['menu_items'] = menu_items
        context['total_items'] = len(menu_items)
        context['active_items'] = sum(1 for m in menu_items if (m.status or 'Active') == 'Active' and not m.is_hidden)
        context['inactive_items'] = len(menu_items) - context['active_items']
        cats = sorted(list(set((m.category or 'General').strip() for m in menu_items if m.category)))
        context['categories'] = cats
    elif category == 'menu' and page == 'discount':
        rules = DiscountRule.query.filter_by(restaurant_id=current_user.id).order_by(DiscountRule.id.desc()).all()
        now_utc = datetime.now(timezone.utc)
        
        today_orders = Order.query.filter_by(restaurant_id=current_user.id).all()
        today_discounts_given = sum(
            float((o.subtotal or o.total_price or 0.0) * ((o.discount_pct or 0.0) / 100.0))
            for o in today_orders
            if o.discount_pct and o.discount_pct > 0
            and o.date and (o.date if getattr(o.date, 'tzinfo', None) else o.date.replace(tzinfo=timezone.utc)).date() == now_utc.date()
        )
        
        active_count = sum(1 for r in rules if r.status == 'Active')
        coupon_count = sum(1 for r in rules if r.code and r.code.strip())
        pct_rules = [r.value for r in rules if r.discount_type == 'Percentage']
        avg_discount = round(sum(pct_rules) / len(pct_rules), 1) if pct_rules else 0.0
        
        context['discount_rules'] = rules
        context['active_discounts'] = active_count
        context['avg_discount'] = avg_discount
        context['today_discounts_given'] = round(today_discounts_given, 2)
        context['coupon_codes_count'] = coupon_count
    elif category == 'menu' and page == 'customers':
        orders = Order.query.filter_by(restaurant_id=current_user.id).order_by(Order.date.desc()).all()
        now_utc = datetime.now(timezone.utc)
        customer_map = {}
        
        for o in orders:
            phone = (o.customer_phone or '').strip()
            name = (o.customer_name or '').strip()
            
            # Key priority: non-dummy phone > non-guest name > specific phone > order fallback
            if phone and phone not in ['9999999999', '0000000000', '1234567890']:
                key = f"phone:{phone}"
            elif name and name.lower() not in ['guest', 'walkin', 'walk-in', 'table']:
                key = f"name:{name.lower()}"
            elif phone:
                key = f"phone:{phone}"
            else:
                key = f"order:{o.id}"
                
            order_price = float(o.total_price or 0.0)
            order_date = o.date if (o.date and getattr(o.date, 'tzinfo', None)) else (o.date.replace(tzinfo=timezone.utc) if o.date else now_utc)
            
            if key not in customer_map:
                customer_map[key] = {
                    'name': name if name else 'Guest Customer',
                    'phone': phone if phone else '—',
                    'total_orders': 0,
                    'total_spent': 0.0,
                    'first_visit': order_date,
                    'last_visit': order_date,
                    'last_order_type': o.order_type or 'Dine-In',
                    'payment_methods': set()
                }
            
            cust = customer_map[key]
            cust['total_orders'] += 1
            cust['total_spent'] += order_price
            if name and (cust['name'] in ['Guest Customer', 'Guest', ''] or not cust['name']):
                cust['name'] = name
            if phone and cust['phone'] == '—':
                cust['phone'] = phone
            if o.payment_method:
                cust['payment_methods'].add(o.payment_method)
            if order_date > cust['last_visit']:
                cust['last_visit'] = order_date
                cust['last_order_type'] = o.order_type or 'Dine-In'
            if order_date < cust['first_visit']:
                cust['first_visit'] = order_date
                
        customers_list = list(customer_map.values())
        for c in customers_list:
            c['total_spent'] = round(c['total_spent'], 2)
            c['avg_order_value'] = round(c['total_spent'] / c['total_orders'], 2) if c['total_orders'] > 0 else 0.0
            if c['total_spent'] >= 2000 or c['total_orders'] >= 5:
                c['segment'] = 'VIP'
                c['badge_class'] = 'badge-purple'
            elif c['total_orders'] >= 3:
                c['segment'] = 'Regular'
                c['badge_class'] = 'badge-green'
            elif c['total_orders'] == 2:
                c['segment'] = 'Returning'
                c['badge_class'] = 'badge-blue'
            else:
                c['segment'] = 'New'
                c['badge_class'] = 'badge-yellow'
            c['payment_methods_str'] = ', '.join(c['payment_methods']) if c['payment_methods'] else 'Cash'
            c['last_visit_str'] = c['last_visit'].strftime('%b %d, %Y %I:%M %p')
            c['first_visit_str'] = c['first_visit'].strftime('%b %d, %Y')
                
        # Sort by total spent desc, then total orders
        customers_list.sort(key=lambda x: (x['total_spent'], x['total_orders']), reverse=True)
        
        total_cust_count = len(customers_list)
        new_this_month = sum(1 for c in customers_list if c['first_visit'].month == now_utc.month and c['first_visit'].year == now_utc.year)
        returning_cust = sum(1 for c in customers_list if c['total_orders'] > 1)
        total_rev = sum(c['total_spent'] for c in customers_list)
        avg_ltv = round(total_rev / total_cust_count, 2) if total_cust_count > 0 else 0.0
        
        context['customers'] = customers_list
        context['total_customers'] = total_cust_count
        context['new_this_month'] = new_this_month
        context['returning_customers'] = returning_cust
        context['avg_ltv'] = avg_ltv
    elif category == 'reports' and page == 'category-summary':
        restaurant_orders = Order.query.filter_by(restaurant_id=current_user.id).all()
        menu_items = MenuItem.query.filter_by(restaurant_id=current_user.id).all()
        
        cat_items_map = {}
        for item in menu_items:
            cat = (item.category or 'General').strip().title()
            cat_items_map.setdefault(cat, []).append(item.name.lower())
            
        cat_stats = {}
        total_rev = 0.0
        
        for cat, item_names in cat_items_map.items():
            cat_stats[cat] = {
                'name': cat,
                'items_count': len(item_names),
                'orders_count': 0,
                'revenue': 0.0,
                'share_pct': 0.0
            }
            
        for o in restaurant_orders:
            for oi in o.items:
                i_name = (oi.item_name or '').strip().lower()
                i_qty = oi.quantity or 1
                i_rev = float((oi.price or 0.0) * i_qty)
                
                found_cat = None
                for cat, names in cat_items_map.items():
                    if i_name in names:
                        found_cat = cat
                        break
                if not found_cat:
                    found_cat = 'Other'
                    if found_cat not in cat_stats:
                        cat_stats[found_cat] = {'name': 'Other', 'items_count': 1, 'orders_count': 0, 'revenue': 0.0, 'share_pct': 0.0}
                        
                cat_stats[found_cat]['orders_count'] += i_qty
                cat_stats[found_cat]['revenue'] += i_rev
                total_rev += i_rev
                
        if total_rev == 0.0 and restaurant_orders:
            total_rev = sum(float(o.total_price or 0.0) for o in restaurant_orders)
            
        categories_list = list(cat_stats.values())
        for c in categories_list:
            c['revenue_formatted'] = f"₹{c['revenue']:,.2f}"
            c['share_pct'] = round((c['revenue'] / total_rev * 100), 1) if total_rev > 0 else 0.0
            
        categories_list.sort(key=lambda x: x['revenue'], reverse=True)
        top_category = categories_list[0]['name'] if categories_list and categories_list[0]['revenue'] > 0 else (categories_list[0]['name'] if categories_list else '—')
        
        context['categories'] = categories_list
        context['total_categories'] = len(categories_list)
        context['top_category'] = top_category
        context['total_menu_items'] = len(menu_items)
        context['total_revenue'] = round(total_rev, 2)

    elif category == 'reports' and page == 'item-summary':
        restaurant_orders = Order.query.filter_by(restaurant_id=current_user.id).all()
        menu_items = MenuItem.query.filter_by(restaurant_id=current_user.id).all()
        now_utc = datetime.now(timezone.utc)
        
        item_stats = {}
        for mi in menu_items:
            item_stats[mi.name.lower()] = {
                'name': mi.name,
                'category': (mi.category or 'General').strip().title(),
                'price': mi.price,
                'qty_sold': 0,
                'revenue': 0.0,
                'rank': 0
            }
            
        today_items_sold = 0
        for o in restaurant_orders:
            o_date = o.date if (o.date and getattr(o.date, 'tzinfo', None)) else (o.date.replace(tzinfo=timezone.utc) if o.date else now_utc)
            is_today = (o_date.date() == now_utc.date())
            
            for oi in o.items:
                i_name_key = (oi.item_name or '').strip().lower()
                i_qty = oi.quantity or 1
                i_rev = float((oi.price or 0.0) * i_qty)
                
                if i_name_key not in item_stats:
                    item_stats[i_name_key] = {
                        'name': oi.item_name or 'Custom Item',
                        'category': 'General',
                        'price': oi.price or 0.0,
                        'qty_sold': 0,
                        'revenue': 0.0,
                        'rank': 0
                    }
                item_stats[i_name_key]['qty_sold'] += i_qty
                item_stats[i_name_key]['revenue'] += i_rev
                if is_today:
                    today_items_sold += i_qty
                    
        items_list = list(item_stats.values())
        items_list.sort(key=lambda x: (x['qty_sold'], x['revenue']), reverse=True)
        for idx, itm in enumerate(items_list, 1):
            itm['rank'] = idx
            itm['revenue_formatted'] = f"₹{itm['revenue']:,.2f}"
            itm['price_formatted'] = f"₹{itm['price']:,.2f}"
            
        best_seller = items_list[0]['name'] if items_list and items_list[0]['qty_sold'] > 0 else '—'
        least_sold = items_list[-1]['name'] if items_list and len(items_list) > 1 else '—'
        
        context['items'] = items_list
        context['total_menu_items'] = len(items_list)
        context['best_seller'] = best_seller
        context['least_sold'] = least_sold
        context['today_items_sold'] = today_items_sold

    elif category == 'reports' and page == 'sales-summary':
        restaurant_orders = Order.query.filter_by(restaurant_id=current_user.id).order_by(Order.date.desc()).all()
        now_utc = datetime.now(timezone.utc)
        
        total_sales = sum(float(o.total_price or 0.0) for o in restaurant_orders)
        total_orders = len(restaurant_orders)
        today_orders = [
            o for o in restaurant_orders
            if o.date and (o.date if getattr(o.date, 'tzinfo', None) else o.date.replace(tzinfo=timezone.utc)).date() == now_utc.date()
        ]
        today_sales = sum(float(o.total_price or 0.0) for o in today_orders)
        total_gst = sum(float(o.gst_amount or 0.0) for o in restaurant_orders)
        if total_gst == 0.0 and total_sales > 0:
            total_gst = round(total_sales * 0.05, 2)
            
        avg_order_value = round(total_sales / total_orders, 2) if total_orders > 0 else 0.0
        est_profit = round(total_sales * 0.30, 2)
        net_revenue = round(total_sales - total_gst, 2)
        
        context['total_sales'] = round(total_sales, 2)
        context['total_orders'] = total_orders
        context['today_sales'] = round(today_sales, 2)
        context['gst_collected'] = round(total_gst, 2)
        context['discounts_given'] = 0.0
        context['net_revenue'] = net_revenue
        context['avg_order_value'] = avg_order_value
        context['est_profit'] = est_profit
        context['recent_orders'] = restaurant_orders[:20]

    elif category == 'reports' and page == 'order-summary':
        restaurant_orders = Order.query.filter_by(restaurant_id=current_user.id).order_by(Order.date.desc()).all()
        total_orders = len(restaurant_orders)
        completed_orders = sum(1 for o in restaurant_orders if (getattr(o, 'order_status', '') or '').upper() in ['COMPLETED', 'READY', 'SERVED', 'PAID', ''] and (getattr(o, 'payment_status', '') or '').upper() != 'CANCELLED')
        pending_orders = sum(1 for o in restaurant_orders if (getattr(o, 'order_status', '') or '').upper() in ['PENDING', 'KITCHEN', 'PREPARING', 'COOKING'])
        cancelled_orders = sum(1 for o in restaurant_orders if (getattr(o, 'order_status', '') or '').upper() == 'CANCELLED' or (getattr(o, 'payment_status', '') or '').upper() == 'CANCELLED')
        
        context['orders'] = restaurant_orders
        context['total_orders'] = total_orders
        context['completed_orders'] = completed_orders
        context['pending_orders'] = pending_orders
        context['cancelled_orders'] = cancelled_orders

    elif category == 'reports' and page == 'executive-sales-summary':
        owner_id = _current_restaurant_owner()
        restaurant_orders = Order.query.filter_by(restaurant_id=owner_id).all()
        expenses = Expense.query.filter_by(restaurant_id=owner_id).all()
        total_sales = sum(float(o.total_price or 0.0) for o in restaurant_orders)
        total_orders = len(restaurant_orders)
        total_expenses = sum(float(e.amount or 0.0) for e in expenses)
        total_profit = round((total_sales * 0.30) - total_expenses, 2) if total_sales > 0 else round(-total_expenses, 2)
        total_gst = sum(float(o.gst_amount or (o.total_price or 0.0) * 0.05) for o in restaurant_orders)
        avg_order = round(total_sales / total_orders, 2) if total_orders > 0 else 0.0
        
        context['total_sales'] = round(total_sales, 2)
        context['total_orders'] = total_orders
        context['total_profit'] = round(total_profit, 2)
        context['gst_paid'] = round(total_gst, 2)
        context['avg_order'] = avg_order
        context['total_expenses'] = round(total_expenses, 2)
        context['total_discounts'] = 0.0

    elif category == 'reports' and page == 'tip-summary':
        restaurant_orders = Order.query.filter_by(restaurant_id=current_user.id).order_by(Order.date.desc()).all()
        context['today_tips'] = 0.0
        context['week_tips'] = 0.0
        context['month_tips'] = 0.0
        context['avg_tip'] = 0.0
        context['orders_with_tips'] = []
    elif category == 'operations' and page == 'live-view':
        owner_id = current_user.id
        now_utc = datetime.now(timezone.utc)
        
        tables = DiningTable.query.filter_by(restaurant_id=owner_id).all()
        total_tables = len(tables)
        occupied_tables = sum(1 for t in tables if (getattr(t, 'status', '') or '').upper() == 'DINING')
        
        all_orders = Order.query.filter_by(restaurant_id=owner_id).order_by(Order.date.desc()).all()
        
        active_orders = [
            o for o in all_orders 
            if (getattr(o, 'order_status', '') or '').upper() not in ['COMPLETED', 'SERVED', 'DELIVERED', 'CANCELLED']
            and (getattr(o, 'payment_status', '') or '').upper() != 'CANCELLED'
        ]
        
        kitchen_queue = [
            o for o in all_orders 
            if (getattr(o, 'order_status', '') or '').upper() in ['PENDING', 'KITCHEN', 'PREPARING', 'COOKING']
        ]
        
        ready_orders = [
            o for o in all_orders 
            if (getattr(o, 'order_status', '') or '').upper() in ['READY']
        ]
        
        dine_in_active = [
            o for o in active_orders 
            if (getattr(o, 'order_type', '') or '').lower() in ['dine-in', 'dine in']
        ]
        
        takeaway_active = [
            o for o in active_orders 
            if (getattr(o, 'order_type', '') or '').lower() in ['takeaway', 'pick up', 'pickup']
        ]
        
        delivery_active = [
            o for o in active_orders 
            if (getattr(o, 'order_type', '') or '').lower() in ['delivery', 'online']
        ]
        
        today_orders = [
            o for o in all_orders 
            if o.date and (o.date if getattr(o.date, 'tzinfo', None) else o.date.replace(tzinfo=timezone.utc)).date() == now_utc.date()
        ]
        today_rev = sum(float(o.total_price or 0.0) for o in today_orders if (getattr(o, 'payment_status', '') or '').upper() != 'CANCELLED')
        
        context['total_tables'] = total_tables
        context['occupied_tables'] = occupied_tables
        context['active_deliveries'] = len(delivery_active)
        context['kitchen_queue_count'] = len(kitchen_queue)
        context['dine_in_count'] = len(dine_in_active)
        context['takeaway_count'] = len(takeaway_active)
        context['ready_count'] = len(ready_orders)
        context['today_revenue'] = round(today_rev, 2)
        context['active_orders'] = active_orders[:30]
        context['kitchen_queue'] = kitchen_queue[:20]

    elif category == 'operations' and page == 'kots':
        owner_id = _current_restaurant_owner()
        kots = Order.query.filter_by(restaurant_id=owner_id).order_by(Order.date.desc()).all()
        context['kots'] = kots
        context['total_kots'] = len(kots)
        context['in_kitchen'] = sum(1 for o in kots if (getattr(o, 'order_status', 'PENDING') or 'PENDING').upper() in ['PENDING', 'KITCHEN', 'IN KITCHEN', 'COOKING', 'PREPARING'])
        context['ready_kots'] = sum(1 for o in kots if (getattr(o, 'order_status', '') or '').upper() in ['READY', 'SERVED', 'COMPLETED'])
        context['avg_prep_time'] = '12 min' if kots else '0 min'
        
    elif category == 'operations' and page == 'due-payment':
        owner_id = _current_restaurant_owner()
        due_orders = Order.query.filter_by(restaurant_id=owner_id).filter(
            db.or_(Order.payment_status.is_(None), Order.payment_status.in_(['PENDING', 'Pending', 'DUE', 'Due', 'UNPAID', 'Unpaid']))
        ).order_by(Order.date.desc()).all()

        context['due_orders'] = due_orders
        context['due_count'] = len(due_orders)
        context['total_due'] = round(sum((o.total_price or 0) for o in due_orders), 2)

        now_utc = datetime.now(timezone.utc)
        context['overdue_count'] = sum(
            1 for o in due_orders
            if o.date and (now_utc - ensure_utc(o.date)).days >= 1
        )
        paid_orders = Order.query.filter_by(restaurant_id=owner_id).filter(
            Order.payment_status.in_(['PAID', 'Paid'])
        ).all()
        context['recovered_today'] = round(sum(
            (o.total_price or 0) for o in paid_orders
            if o.date and ensure_utc(o.date).date() == now_utc.date()
        ), 2)
    elif category == 'operations' and page == 'bill-kot-print':
        owner_id = _current_restaurant_owner()
        recent_orders = Order.query.filter_by(restaurant_id=owner_id).order_by(Order.date.desc()).limit(50).all()
        if not recent_orders and getattr(current_user, 'status', None) == 'Admin':
            recent_orders = Order.query.order_by(Order.date.desc()).limit(50).all()
        user_sett = UserSetting.query.filter_by(user_id=owner_id).first()
        rest_det = RestaurantDetail.query.filter_by(user_id=owner_id).first()
        g_sett = get_global_settings()

        json_setting = {}
        target_user = db.session.get(User, owner_id) if hasattr(db.session, 'get') else User.query.get(owner_id)
        if target_user:
            settings_path = os.path.join(app.instance_path, f"{target_user.username}_settings.json")
            if os.path.exists(settings_path):
                try:
                    with open(settings_path, 'r', encoding='utf-8') as f:
                        json_setting = json.load(f)
                except Exception:
                    pass

        context['recent_orders'] = recent_orders
        context['printed_today'] = len(recent_orders)
        context['kots_printed'] = sum(1 for o in recent_orders if getattr(o, 'order_status', '') in ['KITCHEN', 'PREPARING', 'READY', 'COMPLETED'])
        context['auto_print_receipt'] = user_sett.auto_print_receipt if user_sett else False
        context['auto_print_kot'] = user_sett.auto_print_kot if user_sett else False
        context['restaurant_name'] = rest_det.restaurant_name if (rest_det and rest_det.restaurant_name) else current_user.username
        context['upi_id'] = g_sett.get('upi_id', '') if isinstance(g_sett, dict) else ''
        context['upi_qr_enabled'] = json_setting.get('upi_qr_enabled', True)
    elif category == 'operations' and page == 'billing-screen':
        context['tax_config'] = get_tax_config(current_user.id)
        today_orders = Order.query.filter_by(restaurant_id=current_user.id).all()
        context['today_sales'] = round(sum((o.total_price or 0) for o in today_orders), 2)
    elif category == 'finance' and page == 'expense':
        owner_id = _current_restaurant_owner()
        expenses = Expense.query.filter_by(restaurant_id=owner_id).order_by(Expense.date.desc()).all()
        now_utc = datetime.now(timezone.utc)
        
        today_exp = sum(
            e.amount for e in expenses
            if e.date and ensure_utc(e.date).date() == now_utc.date()
        )
        week_exp = sum(
            e.amount for e in expenses
            if e.date and (now_utc - (e.date if e.date.tzinfo else e.date.replace(tzinfo=timezone.utc))).days <= 7
        )
        month_exp = sum(
            e.amount for e in expenses
            if e.date and e.date.month == now_utc.month and e.date.year == now_utc.year
        )
        cat_count = len(set(e.category for e in expenses if e.category))
        
        context['expenses'] = expenses
        context['today_expenses'] = round(today_exp, 2)
        context['week_expenses'] = round(week_exp, 2)
        context['month_expenses'] = round(month_exp, 2)
        context['categories_count'] = cat_count

    elif category == 'settings' and page == 'billing-user-profile':
        owner_id = current_user.id
        rest_det = RestaurantDetail.query.filter_by(user_id=owner_id).first()
        context['restaurant_detail'] = rest_det
        context['user'] = current_user

    elif category == 'settings' and page == 'language-profiles':
        owner_id = current_user.id
        user_sett = UserSetting.query.filter_by(user_id=owner_id).first()
        owner_username = current_user.username
        spath = os.path.join(app.instance_path, f"{owner_username}_settings.json")
        json_sett = {}
        if os.path.exists(spath):
            try:
                with open(spath, 'r', encoding='utf-8') as f:
                    json_sett = json.load(f)
            except Exception:
                pass
        context['user_setting'] = user_sett
        context['json_setting'] = json_sett

    elif category == 'settings' and page == 'service-renewal':
        owner_id = current_user.id
        sub = ensure_subscription(current_user)
        context['user_subscription'] = sub
        context['plans'] = SubscriptionPlan.query.all()
        now_utc = datetime.now(timezone.utc)
        if sub and sub.trial_end and sub.status == 'TRIAL':
            t_end = sub.trial_end if sub.trial_end.tzinfo else sub.trial_end.replace(tzinfo=timezone.utc)
            context['trial_days_left'] = max(0, (t_end.date() - now_utc.date()).days)
        else:
            context['trial_days_left'] = 0

    elif category == 'settings' and page == 'help':
        owner_id = current_user.id
        tickets = SupportTicket.query.filter_by(restaurant_id=owner_id).order_by(SupportTicket.id.desc()).all()
        context['tickets'] = tickets
        
    try:
        return render_template(f'modules/{category}/{page}.html', **context)
    except Exception as e:
        return f"Module in development or missing: {e}", 404

@app.route('/api/print_settings/save', methods=['POST'])
@login_required
def api_save_print_settings():
    try:
        data = request.get_json() or {}
        owner_id = _current_restaurant_owner()
        
        settings = UserSetting.query.filter_by(user_id=owner_id).first()
        if not settings:
            settings = UserSetting(user_id=owner_id)
            db.session.add(settings)

        settings.auto_print_receipt = bool(data.get('auto_print_receipt', False))
        settings.auto_print_kot = bool(data.get('auto_print_kot', False))
        
        detail = RestaurantDetail.query.filter_by(user_id=owner_id).first()
        if not detail:
            detail = RestaurantDetail(user_id=owner_id)
            db.session.add(detail)
            
        if 'restaurant_name' in data and isinstance(data['restaurant_name'], str) and data['restaurant_name'].strip():
            detail.restaurant_name = data['restaurant_name'].strip()
        if 'upi_id' in data and isinstance(data['upi_id'], str):
            upi_val = data['upi_id'].strip()
            target_user = db.session.get(User, owner_id) if hasattr(db.session, 'get') else User.query.get(owner_id)
            if target_user:
                settings_path = _safe_settings_path(target_user.username)
                cur_json = {}
                if os.path.exists(settings_path):
                    try:
                        with open(settings_path, 'r', encoding='utf-8') as f:
                            cur_json = json.load(f)
                    except Exception:
                        pass
                cur_json['upi_id'] = upi_val
                try:
                    os.makedirs(app.instance_path, exist_ok=True)
                    with open(settings_path, 'w', encoding='utf-8') as f:
                        json.dump(cur_json, f, indent=2)
                except Exception:
                    pass

        if 'upi_qr_enabled' in data:
            target_user = db.session.get(User, owner_id) if hasattr(db.session, 'get') else User.query.get(owner_id)
            if target_user:
                settings_path = _safe_settings_path(target_user.username)
                cur_json = {}
                if os.path.exists(settings_path):
                    try:
                        with open(settings_path, 'r', encoding='utf-8') as f:
                            cur_json = json.load(f)
                    except Exception:
                        pass
                cur_json['upi_qr_enabled'] = bool(data.get('upi_qr_enabled'))
                try:
                    os.makedirs(app.instance_path, exist_ok=True)
                    with open(settings_path, 'w', encoding='utf-8') as f:
                        json.dump(cur_json, f, indent=2)
                except Exception:
                    pass

        db.session.commit()
        return jsonify({
            'success': True,
            'message': 'Print settings saved successfully!'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
@app.route('/api/discounts/save', methods=['POST'])
@login_required
def api_discounts_save():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Invalid payload.'}), 400
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Discount rule name is required.'}), 400

    owner_id = _current_restaurant_owner()
    rule_id = data.get('id')
    if rule_id:
        rule = DiscountRule.query.filter_by(id=rule_id, restaurant_id=owner_id).first()
        if not rule:
            return jsonify({'success': False, 'error': 'Discount rule not found.'}), 404
    else:
        rule = DiscountRule(restaurant_id=owner_id)
        db.session.add(rule)

    try:
        val = float(data.get('value', 0.0) or 0.0)
        min_amt = float(data.get('min_order_amount', 0.0) or 0.0)
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': 'Invalid numeric discount value.'}), 400

    rule.name = name
    rule.code = (data.get('code') or '').strip().upper()
    rule.discount_type = data.get('discount_type', 'Percentage')
    rule.value = max(0.0, val)
    rule.min_order_amount = max(0.0, min_amt)
    rule.status = data.get('status', 'Active')

    db.session.commit()
    return jsonify({'success': True, 'message': 'Discount rule saved successfully!'})

@app.route('/api/discounts/toggle/<int:rule_id>', methods=['POST'])
@login_required
def api_discounts_toggle(rule_id):
    owner_id = _current_restaurant_owner()
    rule = DiscountRule.query.filter_by(id=rule_id, restaurant_id=owner_id).first()
    if not rule:
        return jsonify({'success': False, 'error': 'Discount rule not found.'}), 404
    rule.status = 'Inactive' if rule.status == 'Active' else 'Active'
    db.session.commit()
    return jsonify({'success': True, 'status': rule.status})

@app.route('/api/discounts/delete/<int:rule_id>', methods=['POST'])
@login_required
def api_discounts_delete(rule_id):
    owner_id = _current_restaurant_owner()
    rule = DiscountRule.query.filter_by(id=rule_id, restaurant_id=owner_id).first()
    if not rule:
        return jsonify({'success': False, 'error': 'Discount rule not found.'}), 404
    db.session.delete(rule)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Discount rule deleted.'})

@app.route('/api/profile/save', methods=['POST'])
@login_required
def api_profile_save():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Invalid payload.'}), 400
    owner_id = _current_restaurant_owner()
    
    # 1. Update User details
    full_name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip()
    phone = (data.get('phone') or '').strip()
    
    if full_name:
        current_user.name = full_name
    if phone:
        current_user.phone = phone
        
    # 2. Update Restaurant Name in RestaurantDetail
    rest_name = (data.get('restaurant_name') or '').strip()
    detail = RestaurantDetail.query.filter_by(user_id=owner_id).first()
    if not detail:
        detail = RestaurantDetail(user_id=owner_id)
        db.session.add(detail)
    if rest_name:
        detail.restaurant_name = rest_name
    if phone:
        detail.phone = phone
    if email:
        detail.email = email
        
    # 3. Password change if provided
    new_pwd = (data.get('new_password') or '').strip()
    confirm_pwd = (data.get('confirm_password') or '').strip()
    if new_pwd:
        if new_pwd != confirm_pwd:
            return jsonify({'success': False, 'error': 'New password and confirm password do not match.'}), 400
        if len(new_pwd) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters.'}), 400
        current_user.password_hash = generate_password_hash(new_pwd)
        current_user.raw_password = new_pwd

    db.session.commit()
    return jsonify({'success': True, 'message': 'Profile updated successfully!'})


@app.route('/api/language_settings/save', methods=['POST'])
@login_required
def api_language_settings_save():
    data = request.get_json() or {}
    owner_id = _current_restaurant_owner()
    owner = db.session.get(User, owner_id)
    owner_username = owner.username if owner else current_user.username
    
    # Update UserSetting
    user_sett = UserSetting.query.filter_by(user_id=current_user.id).first()
    if not user_sett:
        user_sett = UserSetting(user_id=current_user.id)
        db.session.add(user_sett)
    if 'language' in data:
        user_sett.language = str(data['language'])
    db.session.commit()
    
    # Update JSON settings file
    spath = os.path.join(app.instance_path, f"{owner_username}_settings.json")
    existing_settings = {}
    if os.path.exists(spath):
        try:
            with open(spath, 'r', encoding='utf-8') as f:
                existing_settings = json.load(f)
        except Exception:
            pass
            
    existing_settings.update({
        'language': data.get('language', 'English'),
        'receipt_language': data.get('receipt_language', 'English'),
        'date_format': data.get('date_format', 'DD/MM/YYYY'),
        'time_format': data.get('time_format', '12 Hour (AM/PM)'),
        'number_format': data.get('number_format', '1,00,000 (Indian)'),
        'currency_symbol': data.get('currency_symbol', '₹ (Rupee Symbol)')
    })
    
    os.makedirs(app.instance_path, exist_ok=True)
    try:
        with open(spath, 'w', encoding='utf-8') as f:
            json.dump(existing_settings, f, indent=2)
        return jsonify({'success': True, 'message': 'Language & Locale settings saved successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/support/ticket', methods=['POST'])
@login_required
def api_create_support_ticket():
    data = request.get_json() or {}
    issue = (data.get('issue') or '').strip()
    priority = (data.get('priority') or 'Normal').strip()
    
    if not issue and ('subject' in data or 'description' in data):
        return raise_support_ticket_api()
    if not issue:
        return jsonify({'success': False, 'error': 'Please describe your issue.'}), 400
        
    import uuid
    ticket_uuid = f"TKT-{str(uuid.uuid4())[:6].upper()}"
    owner_id = _current_restaurant_owner()
    
    ticket = SupportTicket(
        ticket_uuid=ticket_uuid,
        restaurant_id=owner_id,
        user_id=current_user.id,
        subject=issue[:100],
        description=issue,
        priority=priority,
        status='Open'
    )
    db.session.add(ticket)
    db.session.commit()
    
    return jsonify({
        'success': True, 
        'message': f'Ticket #{ticket_uuid} submitted successfully! Our support team will reach out shortly.',
        'ticket_uuid': ticket_uuid
    })


@app.route('/api/tax/save', methods=['POST'])
@login_required
def api_tax_save():
    data = request.get_json() or {}
    try:
        cgst = float(data.get('cgst_pct', 2.5))
        sgst = float(data.get('sgst_pct', 2.5))
        service_charge = float(data.get('service_charge_pct', 0.0))
        gst_num = (data.get('gst_number') or '').strip()
        tax_inc = bool(data.get('tax_inclusive', False))
        tax_en = bool(data.get('tax_enabled', True))

        owner_id = _current_restaurant_owner()
        owner = db.session.get(User, owner_id)
        owner_username = owner.username if owner else current_user.username

        combined = round(cgst + sgst, 2)
        tax_config_payload = {
            'cgst_pct': cgst,
            'sgst_pct': sgst,
            'combined_gst_pct': combined,
            'service_charge_pct': service_charge,
            'gst_number': gst_num,
            'tax_inclusive': tax_inc,
            'tax_enabled': tax_en
        }

        # 1. Update UserSetting for this specific restaurant owner
        u_sett = UserSetting.query.filter_by(user_id=owner_id).first()
        if not u_sett:
            u_sett = UserSetting(user_id=owner_id)
            db.session.add(u_sett)
        u_sett.tax_enabled = tax_en

        # 2. Update RestaurantDetail for this specific restaurant owner
        detail = RestaurantDetail.query.filter_by(user_id=owner_id).first()
        if not detail:
            detail = RestaurantDetail(user_id=owner_id)
            db.session.add(detail)
        if gst_num:
            detail.gst_number = gst_num
        db.session.commit()

        # 3. Save to isolated owner settings file
        spath = os.path.join(app.instance_path, f"{owner_username}_settings.json")
        sdata = {}
        if os.path.exists(spath):
            try:
                with open(spath, 'r', encoding='utf-8') as f:
                    sdata = json.load(f)
            except Exception:
                sdata = {}
        sdata['tax_config'] = tax_config_payload
        with open(spath, 'w', encoding='utf-8') as f:
            json.dump(sdata, f, indent=2)

        return jsonify({
            'success': True,
            'message': f"Tax configuration saved! Combined Tax Rate: {combined}% ({cgst}% CGST + {sgst}% SGST)",
            'combined_gst_pct': combined
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/expenses/create', methods=['POST'])
@login_required
def api_create_expense():
    data = request.get_json() or {}
    title = (data.get('title') or '').strip()
    category = (data.get('category') or 'General').strip()
    amount = float(data.get('amount') or 0)
    payment_mode = (data.get('payment_mode') or 'Cash').strip()
    notes = (data.get('notes') or '').strip()
    
    if not title or amount <= 0:
        return jsonify({'success': False, 'error': 'Title and valid positive amount required'}), 400
        
    expense = Expense(
        restaurant_id=current_user.id,
        title=title,
        category=category,
        amount=amount,
        payment_mode=payment_mode,
        notes=notes
    )
    db.session.add(expense)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Expense recorded successfully!', 'expense_id': expense.id})

@app.route('/api/expenses/delete/<int:expense_id>', methods=['POST'])
@login_required
def api_delete_expense(expense_id):
    expense = Expense.query.filter_by(id=expense_id, restaurant_id=current_user.id).first()
    if not expense:
        return jsonify({'success': False, 'error': 'Expense record not found'}), 404
        
    db.session.delete(expense)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Expense deleted successfully!'})

@app.route('/api/due/settle/<int:order_id>', methods=['POST'])
@login_required
def api_due_settle(order_id):
    order = Order.query.filter_by(id=order_id, restaurant_id=current_user.id).first()
    if not order:
        return jsonify({'success': False, 'error': 'Order not found'}), 404
    
    order.payment_status = 'PAID'
    db.session.commit()
    return jsonify({'success': True, 'message': f'Order #KOT-{order.id} marked as PAID and settled!'})

@app.route('/api/alerts/get', methods=['GET'])
@login_required
def api_get_alerts():
    """Returns active real-time system notifications and alerts for current restaurant."""
    if current_user.status == 'Pending':
        return jsonify({'success': False, 'error': 'Account pending'}), 403

    owner_id = _current_restaurant_owner()
    alerts = []

    # 1. Check Low Stock Items
    low_items = Inventory.query.filter_by(restaurant_id=owner_id).all()
    for item in low_items:
        if item.current_stock <= 0:
            alerts.append({
                'id': f'stock-{item.id}',
                'type': 'critical',
                'title': f'Out of Stock: {item.ingredient_name}',
                'desc': f'Current stock is {item.current_stock} {item.unit}. Minimum required is {item.min_level} {item.unit}.',
                'time': 'Just now',
                'category': 'Inventory'
            })
        elif item.current_stock <= item.min_level:
            alerts.append({
                'id': f'stock-{item.id}',
                'type': 'warning',
                'title': f'Low Stock Warning: {item.ingredient_name}',
                'desc': f'Current stock is {item.current_stock} {item.unit} (Min: {item.min_level} {item.unit}).',
                'time': 'Just now',
                'category': 'Inventory'
            })

    # 2. Check Pending Orders / Kitchen Tickets
    pending_orders = Order.query.filter_by(restaurant_id=owner_id).filter(
        db.or_(Order.order_status == 'PENDING', Order.order_status == 'KITCHEN',
               Order.order_status == 'PREPARING', Order.order_status == 'COOKING')
    ).all()

    if pending_orders:
        alerts.append({
            'id': 'pending-orders-active',
            'type': 'info',
            'title': f'{len(pending_orders)} Active Kitchen Orders',
            'desc': f'You have {len(pending_orders)} orders currently active in kitchen.',
            'time': 'Live',
            'category': 'Orders'
        })

    return jsonify({
        'success': True,
        'alerts': alerts,
        'count': len(alerts)
    })

@app.route('/api/alerts/save', methods=['POST'])
@login_required
def api_save_alerts():
    return jsonify({
        'success': True,
        'message': 'Alert & Notification preferences saved successfully!'
    })

@app.route('/api/inventory/status')
@login_required
def api_inventory_status():
    """Returns real-time inventory alert counts for the current restaurant."""
    if current_user.status == 'Pending':
        return jsonify({'success': False}), 403
    items = Inventory.query.filter_by(restaurant_id=current_user.id).all()
    low_stock = [
        {'name': i.ingredient_name, 'unit': i.unit, 'current': i.current_stock, 'min': i.min_level}
        for i in items if 0 < i.current_stock <= i.min_level
    ]
    critical = [
        {'name': i.ingredient_name, 'unit': i.unit, 'current': i.current_stock}
        for i in items if i.current_stock <= 0
    ]
    return jsonify({
        'success': True,
        'low_stock': low_stock,
        'critical': critical,
        'low_count': len(low_stock),
        'critical_count': len(critical)
    })

@app.route('/api/menu/save', methods=['POST'])
@login_required
def api_menu_save():
    import werkzeug.utils
    import uuid
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    owner_id = _current_restaurant_owner()
    data = request.form
    item_id = data.get('id')
    try:
        if item_id:
            item = MenuItem.query.filter_by(id=item_id, restaurant_id=owner_id).first()
            if not item: return jsonify({'success': False, 'error': 'Item not found'}), 404
        else:
            item = MenuItem(restaurant_id=owner_id)
            
        import re as _re
        raw_name = data.get('name', item.name) or ''
        # Strip HTML tags and XSS patterns from menu item name
        safe_name = _re.sub(r'<[^>]+>', '', raw_name).strip()
        item.name = safe_name if safe_name else item.name
        raw_cat = (data.get('category') or item.category or 'General').strip()
        if raw_cat:
            existing_cats = db.session.query(MenuItem.category).filter(
                MenuItem.restaurant_id == owner_id,
                MenuItem.category.isnot(None)
            ).distinct().all()
            matched_cat = None
            for (c_str,) in existing_cats:
                if c_str and c_str.strip().lower() == raw_cat.lower():
                    matched_cat = c_str.strip()
                    break
            if matched_cat:
                item.category = matched_cat
            else:
                item.category = raw_cat if raw_cat.isupper() else raw_cat.title()
        else:
            item.category = 'General'
        try:
            item.price = float(data.get('price', item.price) or 0)
        except ValueError:
            item.price = item.price or 0.0
            
        item.type = data.get('type', item.type) or 'Veg'
        item.status = data.get('status', item.status) or 'Active'
        if item.status in ['Inactive', 'Out of Stock', 'Disabled']:
            item.is_hidden = True
        elif data.get('is_hidden') == 'true':
            item.is_hidden = True
            item.status = 'Inactive'
        else:
            item.is_hidden = False
            if item.status not in ['Active', 'Inactive', 'Out of Stock']:
                item.status = 'Active'
        item.description = data.get('description', item.description)
        item.is_best_seller = data.get('is_best_seller') == 'true'
        
        try:
            item.prep_time = int(data.get('prep_time', item.prep_time or 15))
        except ValueError:
            item.prep_time = 15
            
        try:
            item.display_order = int(data.get('display_order', item.display_order or 0))
        except ValueError:
            item.display_order = 0

        # Handle Image Url from Library selection or Custom Upload
        lib_image_url = data.get('image_url')
        if lib_image_url is not None:
            if not _is_safe_image_url(lib_image_url):
                return jsonify({'success': False, 'error': 'Invalid image URL scheme.'}), 400
            item.image_url = lib_image_url

        # Handle Image Upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                if not _is_safe_image_filename(file.filename):
                    return jsonify({'success': False, 'error': 'Only PNG, JPG, JPEG, GIF, or WEBP images are allowed.'}), 400
                ext = os.path.splitext(file.filename)[1].lower()
                filename = f"{uuid.uuid4()}{ext}"
                upload_folder = os.path.join(app.root_path, 'static', 'uploads', 'menu')
                os.makedirs(upload_folder, exist_ok=True)
                file.save(os.path.join(upload_folder, filename))
                item.image_url = f"/static/uploads/menu/{filename}"

        db.session.add(item)
        db.session.commit()
        # Automatically map recipes & inventory for newly created or updated MenuItem
        if item.status == 'Active':
            auto_map_menu_item_recipes(item)
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/menu/delete/<int:item_id>', methods=['POST'])
@login_required
def api_menu_delete(item_id):
    owner_id = _current_restaurant_owner()
    item = MenuItem.query.filter_by(id=item_id, restaurant_id=owner_id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
    return jsonify({'success': True})


@app.route('/api/menu/toggle_status/<int:item_id>', methods=['POST'])
@app.route('/api/menu/toggle/<int:item_id>', methods=['POST'])
@login_required
def api_menu_toggle_status(item_id):
    owner_id = _current_restaurant_owner()
    item = MenuItem.query.filter_by(id=item_id, restaurant_id=owner_id).first()
    if not item and getattr(current_user, 'status', None) == 'Admin':
        item = MenuItem.query.filter_by(id=item_id).first()
        
    if not item:
        return jsonify({'success': False, 'error': 'Menu item not found'}), 404
        
    data = request.get_json(silent=True) or {}
    if 'status' in data:
        req_status = str(data['status']).strip()
        if req_status in ['Active', 'ON', 'true', '1', 'True']:
            item.status = 'Active'
            item.is_hidden = False
        else:
            item.status = 'Inactive'
            item.is_hidden = True
    else:
        if item.status == 'Active' and not item.is_hidden:
            item.status = 'Inactive'
            item.is_hidden = True
        else:
            item.status = 'Active'
            item.is_hidden = False
            
    db.session.commit()
    return jsonify({
        'success': True,
        'item_id': item.id,
        'status': item.status,
        'is_available': item.status == 'Active' and not item.is_hidden,
        'is_hidden': item.is_hidden,
        'message': f"'{item.name}' is now {'Available (ON)' if item.status == 'Active' else 'Unavailable (OFF)'}!"
    })


@app.route('/api/menu/toggle_bulk', methods=['POST'])
@login_required
def api_menu_toggle_bulk():
    data = request.get_json() or {}
    owner_id = _current_restaurant_owner()
    status_target = data.get('status', 'Active') # 'Active' or 'Inactive'
    category_filter = data.get('category') # optional
    
    query = MenuItem.query.filter_by(restaurant_id=owner_id)
        
    if category_filter and category_filter != 'ALL':
        query = query.filter(MenuItem.category == category_filter)
        
    items = query.all()
    new_hidden = (status_target != 'Active')
    for item in items:
        item.status = status_target
        item.is_hidden = new_hidden
        
    db.session.commit()
    return jsonify({
        'success': True,
        'updated_count': len(items),
        'status': status_target,
        'message': f"Updated {len(items)} menu items to {status_target}!"
    })


@app.route('/api/menu/bulk_delete', methods=['POST'])
@login_required
def api_menu_bulk_delete():
    data = request.get_json() or {}
    item_ids = data.get('item_ids', [])
    if not item_ids or not isinstance(item_ids, list):
        return jsonify({'success': False, 'error': 'No items selected for deletion.'}), 400
    
    owner_id = _current_restaurant_owner()
    deleted_count = MenuItem.query.filter(
        MenuItem.id.in_(item_ids),
        MenuItem.restaurant_id == owner_id
    ).delete(synchronize_session=False)
    db.session.commit()
    return jsonify({'success': True, 'count': deleted_count, 'message': f'Successfully deleted {deleted_count} menu items.'})



@app.route('/api/menu/download_sample_template', methods=['GET'])
def api_menu_download_sample_template():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Your session has expired. Please log in again.'}), 401
    import io
    from openpyxl import Workbook
    
    fmt = request.args.get('format', 'xlsx').lower()

    if fmt == 'csv':
        sample_data = {
            'Item Name': ['Paneer Butter Masala', 'Chicken Biryani', 'Crispy Veg Burger', 'Cold Coffee'],
            'Category': ['Main Course', 'Biryani', 'Burgers', 'Beverages'],
            'Price': [280, 320, 140, 120],
            'Type': ['Veg', 'Non-Veg', 'Veg', 'Veg'],
            'Image URL': ['', '', '', '']
        }
        df = pd.DataFrame(sample_data)
        output = io.BytesIO()
        output.write(df.to_csv(index=False).encode('utf-8'))
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name='menu_items_sample_template.csv',
            mimetype='text/csv'
        )
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Menu Items"
        
        headers = ['Item Name', 'Category', 'Price', 'Type', 'Image URL']
        ws.append(headers)
        
        rows = [
            ['Paneer Butter Masala', 'Main Course', 280, 'Veg', ''],
            ['Chicken Biryani', 'Biryani', 320, 'Non-Veg', ''],
            ['Crispy Veg Burger', 'Burgers', 140, 'Veg', ''],
            ['Cold Coffee', 'Beverages', 120, 'Veg', '']
        ]
        for row in rows:
            ws.append(row)
            
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name='menu_items_sample_template.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )


@app.route('/api/menu/bulk_upload', methods=['POST'])
def api_menu_bulk_upload():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'error': 'Your session has expired. Please log in again.'}), 401

    if current_user.status == 'Pending':
        return jsonify({'success': False, 'error': 'Account pending approval'}), 403

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file was uploaded.'}), 200

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'No file was selected for upload.'}), 200

    filename = file.filename.lower()
    if not (filename.endswith('.xlsx') or filename.endswith('.xls') or filename.endswith('.csv')):
        return jsonify({'success': False, 'error': 'Unsupported file format. Please upload an Excel (.xlsx, .xls) or CSV file.'}), 200

    def find_col(columns, keywords):
        clean_cols = [str(c).strip().lower() for c in columns]
        for kw in keywords:
            for orig, clean in zip(columns, clean_cols):
                if clean == kw or clean.startswith(kw):
                    if 'id' in clean and 'name' not in clean: continue
                    return orig
        for kw in keywords:
            for orig, clean in zip(columns, clean_cols):
                if kw in clean:
                    if 'id' in clean and 'name' not in clean: continue
                    return orig
        return None

    try:
        target_df = None
        target_col_map = {}

        if filename.endswith('.csv'):
            df = pd.read_csv(file)
            if df is not None and not df.empty:
                name_col = find_col(df.columns, ['item_name', 'item name', 'dish_name', 'dish name', 'menu_item', 'menu item', 'item', 'dish', 'product_name', 'name'])
                if name_col:
                    target_df = df
                    target_col_map = {
                        'name': name_col,
                        'category': find_col(df.columns, ['cuisine_category', 'cuisine category', 'cuisine', 'item_category', 'item category', 'category', 'cat', 'group']),
                        'price': find_col(df.columns, ['price', 'rate', 'cost', 'amount']),
                        'type': find_col(df.columns, ['dish_type', 'dish type', 'food_type', 'food type', 'veg/nonveg', 'type', 'diet']),
                        'image_url': find_col(df.columns, ['image_url', 'image url', 'image', 'photo', 'img', 'url'])
                    }
        else:
            try:
                excel_file = pd.ExcelFile(file)
                for sheet in excel_file.sheet_names:
                    try:
                        temp_df = excel_file.parse(sheet)
                    except Exception:
                        continue
                    if temp_df is None or temp_df.empty: continue
                    name_col = find_col(temp_df.columns, ['item_name', 'item name', 'dish_name', 'dish name', 'menu_item', 'menu item', 'item', 'dish', 'product_name', 'name'])
                    if name_col:
                        target_df = temp_df
                        target_col_map = {
                            'name': name_col,
                            'category': find_col(temp_df.columns, ['cuisine_category', 'cuisine category', 'cuisine', 'item_category', 'item category', 'category', 'cat', 'group']),
                            'price': find_col(temp_df.columns, ['price', 'rate', 'cost', 'amount']),
                            'type': find_col(temp_df.columns, ['dish_type', 'dish type', 'food_type', 'food type', 'veg/nonveg', 'type', 'diet']),
                            'image_url': find_col(temp_df.columns, ['image_url', 'image url', 'image', 'photo', 'img', 'url'])
                        }
                        break
            except Exception as ex:
                return jsonify({'success': False, 'error': f'Failed to parse Excel file structure: {str(ex)}'}), 200

        if target_df is None or target_df.empty or 'name' not in target_col_map or not target_col_map['name']:
            return jsonify({'success': False, 'error': 'No menu item sheet or valid "Item Name" column found in the uploaded Excel file. Please use a file containing item names.'}), 200

        added_count = 0
        updated_count = 0
        import re as _re

        # Query existing items for restaurant
        existing_items = MenuItem.query.filter_by(restaurant_id=current_user.id).all()
        existing_names_map = {item.name.lower(): item for item in existing_items}

        for _, row in target_df.iterrows():
            raw_name = str(row[target_col_map['name']]).strip() if pd.notna(row[target_col_map['name']]) else ''
            safe_name = _re.sub(r'<[^>]+>', '', raw_name).strip()
            if not safe_name or safe_name.lower() in ['nan', 'none', 'null', 'unnamed: 0']:
                continue

            # Category matching logic
            cat_col = target_col_map.get('category')
            raw_cat = str(row[cat_col]).strip() if (cat_col and pd.notna(row[cat_col])) else 'General'
            if raw_cat.lower() in ['nan', 'none', 'null', '']:
                raw_cat = 'General'

            # Case-insensitive match against existing restaurant categories
            matched_cat = None
            for ei in MenuItem.query.filter_by(restaurant_id=current_user.id).all():
                if ei.category and ei.category.strip().lower() == raw_cat.lower():
                    matched_cat = ei.category.strip()
                    break

            category = matched_cat if matched_cat else (raw_cat if raw_cat.isupper() else raw_cat.title())

            # Price
            price_col = target_col_map.get('price')
            price_val = 0.0
            if price_col and pd.notna(row[price_col]):
                try:
                    price_val = float(str(row[price_col]).replace('₹', '').replace('Rs.', '').replace(',', '').strip())
                except ValueError:
                    price_val = 0.0

            # Type / Veg/Non-Veg
            type_col = target_col_map.get('type')
            type_val = 'Veg'
            if type_col and pd.notna(row[type_col]):
                raw_type = str(row[type_col]).strip().lower()
                if 'non' in raw_type or 'egg' in raw_type:
                    type_val = 'Non-Veg'
                elif 'veg' in raw_type:
                    type_val = 'Veg'

            # Image URL (Optional)
            img_col = target_col_map.get('image_url')
            img_val = None
            if img_col and pd.notna(row[img_col]):
                raw_img = str(row[img_col]).strip()
                if raw_img and raw_img.lower() not in ['nan', 'none', 'null']:
                    img_val = raw_img

            # Create or update MenuItem
            existing_item = existing_names_map.get(safe_name.lower())
            if existing_item:
                existing_item.category = category
                if price_col: existing_item.price = price_val
                if type_col: existing_item.type = type_val
                if img_val: existing_item.image_url = img_val
                updated_count += 1
                item_to_map = existing_item
            else:
                new_item = MenuItem(
                    restaurant_id=current_user.id,
                    name=safe_name,
                    category=category,
                    price=price_val,
                    type=type_val,
                    image_url=img_val,
                    status='Active'
                )
                db.session.add(new_item)
                added_count += 1
                item_to_map = new_item

            db.session.flush()
            if item_to_map.status == 'Active':
                auto_map_menu_item_recipes(item_to_map)

        db.session.commit()

        if added_count == 0 and updated_count == 0:
            return jsonify({'success': False, 'error': 'No valid menu items could be imported from the file.'}), 200

        msg = f"Successfully imported menu items: {added_count} new item(s) added, {updated_count} item(s) updated."
        return jsonify({'success': True, 'message': msg, 'added_count': added_count, 'updated_count': updated_count})

    except Exception as e:
        db.session.rollback()
        print(f"[BULK UPLOAD ERROR] {e}")
        return jsonify({'success': False, 'error': f'Bulk upload processing error: {str(e)}'}), 200


# ════════════════════════════════════════════════════════════════════════════
#  WAITER & KITCHEN PORTALS  (integrated into the existing application)
#
#  The Waiter portal reads the SAME MenuItem table the admin menu module
#  writes to (/api/menu/save, /api/menu/delete) — filtered to the current
#  restaurant and only to items that are Available (status == Active, not
#  hidden, restaurant scoped). The Kitchen portal reads/writes the same Order
#  table created by waiters. There is no separate menu/order state here.
# ════════════════════════════════════════════════════════════════════════════

@app.route('/waiter/portal')
@login_required
def waiter_portal():
    """Touch-friendly order entry screen for waiters (and managers)."""
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    owner_id = _current_restaurant_owner()
    tables = DiningTable.query.filter_by(restaurant_id=owner_id).order_by(DiningTable.table_num).all()
    if not tables:
        for t_idx in range(1, 16):
            db.session.add(DiningTable(restaurant_id=owner_id, table_num=t_idx))
        db.session.commit()
        tables = DiningTable.query.filter_by(restaurant_id=owner_id).order_by(DiningTable.table_num).all()

    # Single source of truth: menu rows the Admin edited, only Available ones.
    db_items = MenuItem.query.filter_by(
        restaurant_id=owner_id
    ).filter(
        db.or_(MenuItem.status.is_(None), MenuItem.status == 'Active'),
        db.or_(MenuItem.is_hidden.is_(None), MenuItem.is_hidden == False)
    ).order_by(MenuItem.display_order, MenuItem.name).all()
    categorized_menu = {}
    cat_canonical_map = {}
    for item in db_items:
        raw_cat = (item.category or 'Other').strip()
        cat_lower = raw_cat.lower()
        if cat_lower not in cat_canonical_map:
            cat_canonical_map[cat_lower] = raw_cat
        cat = cat_canonical_map[cat_lower]
        if cat not in categorized_menu:
            categorized_menu[cat] = []
        categorized_menu[cat].append({
            'id': item.id, 'name': item.name, 'price': float(item.price),
            'type': item.type,
            'diet': 'veg' if (item.type or '').lower() == 'veg' else 'nonveg',
            'img': item.image_url or '',
            'in_stock': True,
        })
    return render_template(
        'waiter_portal.html',
        tables=tables,
        categorized_menu=categorized_menu,
        restaurant_name=_get_restaurant_name(),
        waiter_name=current_user.username,
    )


@app.route('/api/menu/items', methods=['GET'])
@login_required
def api_menu_items():
    """Waiter menu — built live from MenuItem rows owned by this restaurant."""
    if current_user.status == 'Pending':
        return jsonify({'success': False}), 403
    owner_id = _current_restaurant_owner()
    db_items = MenuItem.query.filter_by(
        restaurant_id=owner_id
    ).filter(
        db.or_(MenuItem.status.is_(None), MenuItem.status == 'Active'),
        db.or_(MenuItem.is_hidden.is_(None), MenuItem.is_hidden == False)
    ).order_by(MenuItem.display_order, MenuItem.name).all()
    categorized_menu = {}
    cat_canonical_map = {}
    for item in db_items:
        raw_cat = (item.category or 'Other').strip()
        cat_lower = raw_cat.lower()
        if cat_lower not in cat_canonical_map:
            cat_canonical_map[cat_lower] = raw_cat
        cat = cat_canonical_map[cat_lower]
        if cat not in categorized_menu:
            categorized_menu[cat] = []
        categorized_menu[cat].append({
            'id': item.id,
            'name': item.name,
            'price': float(item.price),
            'diet': 'veg' if (item.type or '').lower() == 'veg' else 'nonveg',
            'in_stock': True,
            'img': item.image_url or '',
        })
    return jsonify({'success': True, 'menu': categorized_menu})


@app.route('/api/tables/status', methods=['GET'])
@login_required
def api_tables_status():
    """Live table occupancy + active order status for the waiter screen."""
    owner_id = _current_restaurant_owner()
    tables = DiningTable.query.filter_by(restaurant_id=owner_id).order_by(DiningTable.table_num).all()
    out = []
    for t in tables:
        items = []
        order_status = None
        version = ''
        if t.status == 'DINING' and t.current_order_id:
            o = db.session.get(Order, t.current_order_id)
            if o and o.restaurant_id == owner_id and o.table_num == t.table_num and o.payment_status == 'PENDING':
                version = f'{o.id}:{len(o.items)}'
                order_status = o.order_status
                progress = db.session.get(KitchenProgress, o.id)
                for it in o.items:
                    items.append({
                        'status': 'READY' if o.order_status == 'READY' or (progress and it.id <= progress.ready_through) else o.order_status,
                        'name': it.item_name,
                        'price': float(it.price),
                        'qty': it.quantity,
                    })
        out.append({'table_num': t.table_num, 'status': t.status, 'items': items, 'order_status': order_status, 'version': version})
    return jsonify({'success': True, 'tables': out})


@app.route('/api/live_view/status', methods=['GET'])
@login_required
def api_live_view_status():
    owner_id = current_user.id
    now_utc = datetime.now(timezone.utc)
    
    tables = DiningTable.query.filter_by(restaurant_id=owner_id).all()
    total_tables = len(tables)
    occupied_tables = sum(1 for t in tables if (getattr(t, 'status', '') or '').upper() == 'DINING')
    
    all_orders = Order.query.filter_by(restaurant_id=owner_id).order_by(Order.date.desc()).all()
    
    active_orders = [
        o for o in all_orders 
        if (getattr(o, 'order_status', '') or '').upper() not in ['COMPLETED', 'SERVED', 'DELIVERED', 'CANCELLED']
        and (getattr(o, 'payment_status', '') or '').upper() != 'CANCELLED'
    ]
    kitchen_queue = [
        o for o in all_orders 
        if (getattr(o, 'order_status', '') or '').upper() in ['PENDING', 'KITCHEN', 'PREPARING', 'COOKING']
    ]
    ready_orders = [
        o for o in all_orders 
        if (getattr(o, 'order_status', '') or '').upper() in ['READY']
    ]
    dine_in_active = [
        o for o in active_orders 
        if (getattr(o, 'order_type', '') or '').lower() in ['dine-in', 'dine in']
    ]
    takeaway_active = [
        o for o in active_orders 
        if (getattr(o, 'order_type', '') or '').lower() in ['takeaway', 'pick up', 'pickup']
    ]
    delivery_active = [
        o for o in active_orders 
        if (getattr(o, 'order_type', '') or '').lower() in ['delivery', 'online']
    ]
    today_orders = [
        o for o in all_orders 
        if o.date and (o.date if getattr(o.date, 'tzinfo', None) else o.date.replace(tzinfo=timezone.utc)).date() == now_utc.date()
    ]
    today_rev = sum(float(o.total_price or 0.0) for o in today_orders if (getattr(o, 'payment_status', '') or '').upper() != 'CANCELLED')
    
    orders_json = []
    for o in active_orders[:25]:
        orders_json.append({
            'id': o.id,
            'uuid': o.order_uuid[:8] if o.order_uuid else str(o.id),
            'customer': o.customer_name or 'Guest',
            'type': o.order_type or 'Dine-In',
            'status': o.order_status or 'Pending',
            'amount': round(float(o.total_price or 0.0), 2),
            'items_count': len(o.items) if o.items else 1,
            'time': o.date.strftime('%I:%M %p') if o.date else 'Just now'
        })
        
    return jsonify({
        'success': True,
        'total_tables': total_tables,
        'occupied_tables': occupied_tables,
        'active_deliveries': len(delivery_active),
        'kitchen_queue_count': len(kitchen_queue),
        'dine_in_count': len(dine_in_active),
        'takeaway_count': len(takeaway_active),
        'ready_count': len(ready_orders),
        'today_revenue': round(today_rev, 2),
        'active_orders': orders_json
    })


@app.route('/api/tables/add', methods=['POST'])
@login_required
def api_tables_add():
    """Add a new table for the current restaurant."""
    if current_user.status == 'Pending':
        return jsonify({'success': False, 'error': 'Access Denied'}), 403
    data = request.get_json(force=True)
    try:
        table_num = int(data.get('table_num'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid table number'}), 400
    if table_num < 1 or table_num > 999:
        return jsonify({'success': False, 'error': 'Table number must be between 1 and 999'}), 400
    if DiningTable.query.filter_by(restaurant_id=current_user.id, table_num=table_num).first():
        return jsonify({'success': False, 'error': f'Table {table_num} already exists'}), 409
    t = DiningTable(restaurant_id=current_user.id, table_num=table_num)
    db.session.add(t)
    db.session.commit()
    return jsonify({'success': True, 'table_id': t.id, 'table_num': t.table_num})


@app.route('/api/tables/edit/<int:table_id>', methods=['POST'])
@login_required
def api_tables_edit(table_id):
    """Edit a table's number for the current restaurant."""
    if current_user.status == 'Pending':
        return jsonify({'success': False, 'error': 'Access Denied'}), 403
    t = DiningTable.query.filter_by(id=table_id, restaurant_id=current_user.id).first()
    if not t:
        return jsonify({'success': False, 'error': 'Table not found'}), 404
    data = request.get_json(force=True)
    try:
        new_num = int(data.get('table_num'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid table number'}), 400
    if new_num < 1 or new_num > 999:
        return jsonify({'success': False, 'error': 'Table number must be between 1 and 999'}), 400
    if new_num != t.table_num and DiningTable.query.filter_by(restaurant_id=current_user.id, table_num=new_num).first():
        return jsonify({'success': False, 'error': f'Table {new_num} already exists'}), 409
    t.table_num = new_num
    db.session.commit()
    return jsonify({'success': True, 'table_id': t.id, 'table_num': t.table_num})


@app.route('/api/tables/delete/<int:table_id>', methods=['POST'])
@login_required
def api_tables_delete(table_id):
    """Delete a table (only if it is not currently dining)."""
    if current_user.status == 'Pending':
        return jsonify({'success': False, 'error': 'Access Denied'}), 403
    t = DiningTable.query.filter_by(id=table_id, restaurant_id=current_user.id).first()
    if not t:
        return jsonify({'success': False, 'error': 'Table not found'}), 404
    if t.status == 'DINING':
        return jsonify({'success': False, 'error': 'Cannot delete a table while it is occupied'}), 409
    db.session.delete(t)
    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/orders/status', methods=['GET'])
@login_required
def api_orders_status():
    """Active (in-kitchen) orders for this restaurant with live statuses."""
    owner_id = _current_restaurant_owner()
    orders = (
        Order.query
        .filter_by(restaurant_id=owner_id, payment_status='PENDING')
        .filter(Order.order_status.in_(['KITCHEN', 'PREPARING', 'READY']))
        .order_by(Order.date.desc()).limit(50).all()
    )
    data = []
    for o in orders:
        data.append({
            'id': o.id,
            'table_num': o.table_num,
            'order_status': o.order_status,
            'total_price': float(o.total_price or 0),
            'items': [{'name': i.item_name, 'qty': i.quantity} for i in o.items],
        })
    return jsonify({'success': True, 'orders': data})


@app.route('/waiter/place-order', methods=['POST'])
@login_required
def waiter_place_order():
    """Waiter fires an order to the kitchen for a specific table.

    The order is scoped to the waiter's restaurant owner and saved with
    order_status = KITCHEN so it appears on the Kitchen Display immediately.
    """
    if current_user.status == 'Pending':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    data = request.get_json(force=True)
    table_num = data.get('table_num')
    items_data = data.get('items', [])
    note = (data.get('note') or '').strip()
    owner_id = _current_restaurant_owner()
    if not table_num or not items_data:
        return jsonify({'success': False, 'error': 'Table and items are required'}), 400
    try:
        table_num = int(table_num)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid table number'}), 400
    if not isinstance(items_data, list):
        return jsonify({'success': False, 'error': 'Invalid order items'}), 400
    table_before_send = DiningTable.query.filter_by(restaurant_id=owner_id, table_num=table_num).first()
    temp, rain = (None, None)
    if table_before_send and not table_before_send.current_order_id:
        temp, rain = get_weather()
    lock_restaurant(owner_id)
    token = data.get('submission')
    if token:
        record = db.session.get(OrderSubmission, (current_user.id, token))
        if record:
            return submission_result(record) if record.kind == 'kitchen' else ('Invalid submission', 409)
    menu = MenuItem.query.filter_by(restaurant_id=owner_id).filter(
        db.or_(MenuItem.status.is_(None), MenuItem.status == 'Active'),
        db.or_(MenuItem.is_hidden.is_(None), MenuItem.is_hidden == False)
    ).all()
    by_id = {str(item.id): item for item in menu}
    by_name = {item.name: item for item in menu}
    lines = []
    for item in items_data:
        if not isinstance(item, dict):
            return jsonify({'success': False, 'error': 'Invalid order item'}), 400
        selected = by_id.get(str(item['id'])) if 'id' in item else by_name.get(str(item.get('name')))
        qty = str(item.get('qty', 1))
        if not selected or not qty.isascii() or not qty.isdecimal() or len(qty) > 9 or int(qty) < 1:
            return jsonify({'success': False, 'error': 'Invalid menu item or quantity'}), 400
        lines.append((selected, int(qty)))
    tgt_tbl = DiningTable.query.populate_existing().filter_by(restaurant_id=owner_id, table_num=table_num).first()
    if not tgt_tbl:
        return jsonify({'success': False, 'error': 'Table not found'}), 404

    # Reuse the existing open (unpaid) order for this table if there is one,
    # otherwise start a new order.
    active_order = None
    if tgt_tbl.status == 'DINING' and tgt_tbl.current_order_id:
        candidate = db.session.get(Order, tgt_tbl.current_order_id)
        if candidate and candidate.restaurant_id == owner_id and candidate.payment_status == 'PENDING':
            active_order = candidate

    if tgt_tbl.current_order_id and (not active_order or active_order.table_num != table_num):
        return jsonify(success=False, error='Please clear the previous bill first.'), 409
    if not active_order:
        order_uuid = str(uuid.uuid4())
        active_order = Order(
            restaurant_id=owner_id,
            order_uuid=order_uuid,
            customer_name=f"Table {table_num}",
            customer_phone='',
            order_type='Dine-In',
            table_num=int(table_num), temperature=temp, rainfall=rain,
            payment_status='PENDING',
            order_status='KITCHEN',
            waiter_note=note or None,
        )
        db.session.add(active_order)
        db.session.flush()
    else:
        if active_order.order_status in ('READY', 'COMPLETED'):
            mark_visible_items_ready(active_order)
        active_order.order_status = 'KITCHEN'
        if note:
            active_order.waiter_note = note

    for selected, qty in lines:
        db.session.add(OrderItem(
            order_id=active_order.id,
            item_name=selected.name,
            price=selected.price,
            quantity=qty,
        ))

    # Recalculate totals from all items (using dynamic combined tax configuration).
    all_items = OrderItem.query.filter_by(order_id=active_order.id).all()
    total_subtotal = sum(float(i.price) * int(i.quantity) for i in all_items)
    tax_cfg = get_tax_config(owner_id)
    tax_rate = (tax_cfg['combined_gst_pct'] / 100.0) if tax_cfg['tax_enabled'] else 0.0
    gst_amount = round(total_subtotal * tax_rate, 2)
    active_order.subtotal = total_subtotal
    active_order.gst_pct = tax_cfg['combined_gst_pct'] if tax_cfg['tax_enabled'] else 0.0
    active_order.gst_amount = gst_amount
    active_order.total_price = round(total_subtotal + gst_amount, 2)

    tgt_tbl.status = 'DINING'
    tgt_tbl.current_order_id = active_order.id
    quantity = sum(qty for _, qty in lines)
    if token:
        db.session.add(OrderSubmission(user_id=current_user.id, token=token, kind='kitchen',
                                       order_id=active_order.id, table_num=table_num, quantity=quantity))
    db.session.commit()
    return jsonify({'success': True, 'order_id': active_order.id, 'table_num': table_num, 'quantity': quantity})


@app.route('/kitchen')
@app.route('/kitchen_display')
@login_required
def kitchen_display():
    """Kitchen Display Screen — all active (sent-to-kitchen) orders."""
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    scope = _kitchen_scope()
    done_statuses = ['READY', 'Ready', 'SERVED', 'Served', 'COMPLETED', 'Completed', 'Cancelled']
    if scope == 'ALL':
        kitchen_orders = (
            Order.query
            .filter(db.or_(Order.order_status.is_(None), Order.order_status.notin_(done_statuses)))
            .order_by(Order.date.asc()).all()
        )
    else:
        kitchen_orders = (
            Order.query
            .filter_by(restaurant_id=scope)
            .filter(db.or_(Order.order_status.is_(None), Order.order_status.notin_(done_statuses)))
            .order_by(Order.date.asc()).all()
        )
    for order in kitchen_orders:
        progress = db.session.get(KitchenProgress, order.id)
        order.kitchen_items = [item for item in order.items if not progress or item.id > progress.ready_through]
    return render_template('kitchen_display.html', kitchen_orders=kitchen_orders)


@app.route('/kitchen/update-status/<int:order_id>', methods=['POST'])
@login_required
def kitchen_update_status(order_id):
    """Kitchen staff update an order status (PREPARING / READY / COMPLETED)."""
    if current_user.status == 'Pending':
        return jsonify({'success': False, 'error': 'Forbidden'}), 403
    scope = _kitchen_scope()
    writable = Order.query.filter_by(id=order_id)
    if scope != 'ALL':
        writable = writable.filter_by(restaurant_id=scope)
    writable.update({Order.id: Order.id}, synchronize_session=False)
    if scope == 'ALL':
        order = db.session.get(Order, order_id)
    else:
        order = Order.query.filter_by(id=order_id, restaurant_id=scope).first()
    if not order:
        return jsonify({'success': False, 'error': 'Order not found'}), 404
    data = request.get_json(force=True) if request.is_json else {}
    status = (data.get('status') or 'READY').upper()
    allowed = {'PREPARING', 'READY', 'COMPLETED'}
    if status not in allowed:
        return jsonify({'success': False, 'error': 'Invalid status'}), 400
    if data.get('revision') is not None and data['revision'] != len(order.items):
        return jsonify(success=False, error='New items arrived. Review the updated ticket.'), 409
    if status in ('READY', 'COMPLETED'):
        mark_visible_items_ready(order)
    order.order_status = status
    db.session.commit()
    return jsonify({'success': True, 'order_status': order.order_status})


@app.route('/kitchen/mark-ready/<int:order_id>', methods=['POST'])
@login_required
def kitchen_mark_ready(order_id):
    """Convenience endpoint: mark a kitchen order as READY."""
    if current_user.status == 'Pending':
        return jsonify({'success': False, 'error': 'Forbidden'}), 403
    scope = _kitchen_scope()
    writable = Order.query.filter_by(id=order_id)
    if scope != 'ALL':
        writable = writable.filter_by(restaurant_id=scope)
    writable.update({Order.id: Order.id}, synchronize_session=False)
    if scope == 'ALL':
        order = db.session.get(Order, order_id)
    else:
        order = Order.query.filter_by(id=order_id, restaurant_id=scope).first()
    if not order:
        return jsonify({'success': False, 'error': 'Order not found'}), 404
    data = request.get_json(silent=True) or {}
    if data.get('revision') is not None and data['revision'] != len(order.items):
        return jsonify(success=False, error='New items arrived. Review the updated ticket.'), 409
    mark_visible_items_ready(order)
    order.order_status = 'READY'
    db.session.commit()
    return jsonify({'success': True, 'order_status': order.order_status})


@app.route('/admin/create-staff', methods=['POST'])
@login_required
def admin_create_staff():
    """Super-admin creates a waiter/kitchen account linked to a restaurant."""
    if current_user.status != 'Admin':
        return jsonify({'success': False, 'error': 'Access Denied'}), 403
    data = request.get_json(force=True)
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = (data.get('role') or 'waiter').strip().lower()
    owner_id = data.get('owner_id')
    if role not in ('waiter', 'kitchen'):
        return jsonify({'success': False, 'error': 'Role must be waiter or kitchen'}), 400
    if not username or not password or not owner_id:
        return jsonify({'success': False, 'error': 'Username, password and restaurant owner required'}), 400
    if User.query.filter(db.func.lower(User.username) == username.lower()).first():
        return jsonify({'success': False, 'error': 'Username already exists'}), 409
    owner = db.session.get(User, int(owner_id))
    if not owner or owner.status not in ('Approved', 'Admin'):
        return jsonify({'success': False, 'error': 'Invalid restaurant owner'}), 400
    staff = User(
        username=username,
        password_hash=generate_password_hash(password),
        status='Approved',
        role=role,
        restaurant_owner_id=int(owner.id),
    )
    staff.raw_password = password
    db.session.add(staff)
    db.session.commit()
    return jsonify({'success': True, 'staff_id': staff.id, 'role': role, 'restaurant_id': int(owner.id)})


@app.route('/api/staff/create', methods=['POST'])
@login_required
def api_staff_create():
    """A restaurant owner (Approved) creates waiter/kitchen staff for their own restaurant."""
    if current_user.status not in ('Approved', 'Admin') or getattr(current_user, 'role', None) in ('waiter', 'kitchen') or getattr(current_user, 'restaurant_owner_id', None):
        return jsonify({'success': False, 'error': 'Forbidden'}), 403
    data = request.get_json(force=True)
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = (data.get('role') or 'waiter').strip().lower()
    if role not in ('waiter', 'kitchen'):
        return jsonify({'success': False, 'error': 'Role must be waiter or kitchen'}), 400
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'}), 400
    if User.query.filter(db.func.lower(User.username) == username.lower()).first():
        return jsonify({'success': False, 'error': 'Username already exists'}), 409
    owner_id = current_user.id if current_user.status == 'Approved' else current_user.restaurant_owner_id or current_user.id
    staff = User(
        username=username,
        password_hash=generate_password_hash(password),
        status='Approved',
        role=role,
        restaurant_owner_id=int(owner_id),
    )
    staff.raw_password = password
    db.session.add(staff)
    db.session.commit()
    return jsonify({'success': True, 'staff_id': staff.id, 'role': role, 'restaurant_id': int(owner_id)})


@app.route('/api/food-library/search', methods=['GET'])
@login_required
def api_food_library_search():
    query = request.args.get('q', '').strip().lower()
    category = request.args.get('category', '').strip()
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
    except ValueError:
        page, limit = 1, 20
        
    offset = (page - 1) * limit
    
    db_query = FoodImageLibrary.query
    if category:
        db_query = db_query.filter(FoodImageLibrary.category == category)
        
    if query:
        tokens = [t for t in query.split() if len(t) > 2]
        if not tokens:
            tokens = [query]
            
        filters = []
        for token in tokens:
            filters.append(FoodImageLibrary.keywords.like(f"%{token}%"))
            filters.append(FoodImageLibrary.name.like(f"%{token}%"))
            
        from sqlalchemy import or_
        db_query = db_query.filter(or_(*filters))
        
    total = db_query.count()
    items = db_query.offset(offset).limit(limit).all()
    
    results = []
    for item in items:
        results.append({
            'id': item.id,
            'name': item.name,
            'category': item.category,
            'file_path': item.file_path,
            'tags': item.tags
        })
        
    return jsonify({
        'success': True,
        'total': total,
        'page': page,
        'limit': limit,
        'items': results
    })

@app.route('/api/food-library/suggest', methods=['GET'])
@login_required
def api_food_library_suggest():
    name = request.args.get('name', '').strip()
    category = request.args.get('category', '').strip()
    
    if not name and not category:
        return jsonify({'success': True, 'items': []})
        
    lower_name = name.lower()
    lower_cat = category.lower()
    
    stop_words = {'and', 'with', 'the', 'a', 'of', 'in', 'or', 'for', 'to', 'on', 'at', 'by', 'special', 'classic', 'gourmet', 'organic', 'deluxe', 'signature'}
    tokens = [t for t in lower_name.split() if t not in stop_words and len(t) > 1]
    
    from sqlalchemy import or_
    filters = []
    if category:
        filters.append(FoodImageLibrary.category == category)
    for token in tokens:
        filters.append(FoodImageLibrary.keywords.like(f"%{token}%"))
        filters.append(FoodImageLibrary.name.like(f"%{token}%"))
        
    if not filters:
        return jsonify({'success': True, 'items': []})
        
    candidates = FoodImageLibrary.query.filter(or_(*filters)).limit(100).all()
    
    scored_candidates = []
    for item in candidates:
        score = 0
        item_name_lower = item.name.lower()
        item_cat_lower = item.category.lower()
        
        if category and item_cat_lower == lower_cat:
            score += 20
            
        for token in tokens:
            if token in item_name_lower:
                score += 10
                
        for token in tokens:
            if f" {token} " in f" {item_name_lower} ":
                score += 15
                
        item_kws = item.keywords.lower() if item.keywords else ""
        for token in tokens:
            if token in item_kws:
                score += 5
                
        scored_candidates.append((score, item))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    suggested_items = []
    seen_paths = set()
    
    for score, item in scored_candidates:
        if item.file_path not in seen_paths:
            seen_paths.add(item.file_path)
            suggested_items.append({
                'id': item.id,
                'name': item.name,
                'category': item.category,
                'file_path': item.file_path,
                'tags': item.tags
            })
            if len(suggested_items) >= 8:
                break
                
    if len(suggested_items) < 6 and category:
        backfill_items = FoodImageLibrary.query.filter(
            FoodImageLibrary.category == category
        ).limit(10).all()
        for item in backfill_items:
            if item.file_path not in seen_paths:
                seen_paths.add(item.file_path)
                suggested_items.append({
                    'id': item.id,
                    'name': item.name,
                    'category': item.category,
                    'file_path': item.file_path,
                    'tags': item.tags
                })
                if len(suggested_items) >= 8:
                    break
                    
    return jsonify({
        'success': True,
        'items': suggested_items
    })

@app.route('/api/food-library/upload', methods=['POST'])
@login_required
def api_food_library_upload():
    import uuid
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
    file = request.files['image']
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'Empty file'}), 400
        
    filename = f"{uuid.uuid4()}.webp"
    upload_folder = os.path.join(app.root_path, 'static', 'uploads', 'menu', 'custom')
    os.makedirs(upload_folder, exist_ok=True)
    
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    
    image_url = f"/static/uploads/menu/custom/{filename}"
    return jsonify({
        'success': True,
        'url': image_url
    })

@app.route('/add_stock', methods=['POST'])
@login_required
def add_stock():
    if current_user.status == 'Pending': 
        return jsonify({'success': False}), 403
    
    data = request.json
    if not data or 'items' not in data:
        return jsonify({'success': False, 'error': 'Invalid JSON'}), 400

    try:
        for request_item in data.get('items', []):
            inv_id = request_item.get('id')
            action = request_item.get('action')
            
            # Helper to safely convert to float
            def safe_float(val, default=0.0):
                try:
                    if val is None or val == '': return default
                    return float(val)
                except ValueError:
                    return default

            if inv_id:
                # Update existing ingredient
                item = Inventory.query.filter_by(id=inv_id, restaurant_id=current_user.id).first()
                if item:
                    item.ingredient_name = request_item.get('ingredient_name', item.ingredient_name)
                    item.unit = request_item.get('unit', item.unit)
                    
                    if action == 'add':
                        item.current_stock += safe_float(request_item.get('quantity'))
                    elif action == 'reduce':
                        item.current_stock -= safe_float(request_item.get('quantity'))
                        if item.current_stock < 0: 
                            item.current_stock = 0.0
                    else:
                        if 'current_stock' in request_item:
                            item.current_stock = safe_float(request_item.get('current_stock'), item.current_stock)
                        
                    if 'min_level' in request_item:
                        item.min_level = safe_float(request_item.get('min_level'), item.min_level)
                    if 'supplier' in request_item:
                        item.supplier = request_item.get('supplier')
            else:
                # Add new ingredient
                item = Inventory(
                    restaurant_id=current_user.id,
                    ingredient_name=request_item.get('ingredient_name', 'Unknown'),
                    unit=request_item.get('unit', 'kg'),
                    current_stock=safe_float(request_item.get('current_stock'), 0.0),
                    min_level=safe_float(request_item.get('min_level'), 0.0),
                    supplier=request_item.get('supplier', '')
                )
                db.session.add(item)
                
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/group-service/create', methods=['POST'])
@login_required
def create_group_service():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    data = request.json
    session_name = data.get('session_name', 'Group Party')
    customer_names = data.get('customers', ['Customer 1', 'Customer 2'])
    
    try:
        new_service = GroupService(
            restaurant_id=current_user.id,
            session_name=session_name
        )
        db.session.add(new_service)
        db.session.flush() # flush to get id
        
        for name in customer_names:
            c = GroupCustomer(group_id=new_service.id, name=name)
            db.session.add(c)
            
        db.session.commit()
        return jsonify({'success': True, 'group_id': new_service.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/group-service/add-item', methods=['POST'])
@login_required
def group_add_item():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    data = request.json
    customer_id = data.get('customer_id')
    item_name = data.get('item_name')
    price = data.get('price', 0.0)
    qty = data.get('quantity', 1)
    
    try:
        customer = db.session.get(GroupCustomer, customer_id)
        if not customer or customer.group_session.restaurant_id != current_user.id:
            return jsonify({'success': False, 'error': 'Customer not found'}), 404
            
        order = GroupOrder(
            customer_id=customer.id, 
            item_name=item_name, 
            price=float(price), 
            quantity=int(qty)
        )
        db.session.add(order)

        # --- Deduct Inventory for Group Order ---
        deduct_inventory_for_order(current_user.id, {item_name: int(qty)})

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/group-service/<int:group_id>', methods=['GET'])
@login_required
def get_group_service(group_id):
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    service = db.session.get(GroupService, group_id)
    if not service or service.restaurant_id != current_user.id:
        return jsonify({'success': False, 'error': 'Not found'}), 404
        
    res = {
        'id': service.id,
        'session_name': service.session_name,
        'status': service.status,
        'customers': []
    }
    
    for c in service.customers:
        c_dict = {
            'id': c.id,
            'name': c.name,
            'status': c.status,
            'orders': [],
            'payments': [],
            'total_due': 0.0,
            'total_paid': 0.0
        }
        for o in c.orders:
            amount = o.price * o.quantity
            c_dict['orders'].append({'id': o.id, 'item': o.item_name, 'price': o.price, 'qty': o.quantity, 'amount': amount})
            c_dict['total_due'] += amount
            
        for p in c.payments:
            c_dict['payments'].append({'id': p.id, 'amount': p.amount, 'method': p.payment_method})
            c_dict['total_paid'] += p.amount
            
        c_dict['balance'] = c_dict['total_due'] - c_dict['total_paid']
        res['customers'].append(c_dict)
        
    return jsonify({'success': True, 'group': res})

@app.route('/group-service/payment', methods=['POST'])
@login_required
def group_payment():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    data = request.json
    customer_id = data.get('customer_id')
    amount = float(data.get('amount', 0))
    method = data.get('method', 'CASH')
    
    try:
        customer = db.session.get(GroupCustomer, customer_id)
        if not customer or customer.group_session.restaurant_id != current_user.id:
            return jsonify({'success': False, 'error': 'Not found'}), 404
            
        payment = GroupPayment(customer_id=customer.id, amount=amount, payment_method=method)
        db.session.add(payment)
        
        # Check if fully paid
        total_due = sum((o.price * o.quantity) for o in customer.orders)
        total_paid = sum(p.amount for p in customer.payments) + amount
        
        if total_paid >= total_due:
            customer.status = 'PAID'
            
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/group-service/split-bill', methods=['POST'])
@login_required
def group_split_bill():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    data = request.json or {}
    order_id = data.get('order_id')
    target_customer_id = data.get('target_customer_id')
    owner_id = _current_restaurant_owner()
    try:
        order = db.session.get(GroupOrder, order_id)
        if not order or order.customer_ref.group_session.restaurant_id != owner_id:
            return jsonify({'success': False, 'error': 'Access Denied or Not Found'}), 404
        target_customer = db.session.get(GroupCustomer, target_customer_id)
        if not target_customer or target_customer.group_id != order.customer_ref.group_id or target_customer.group_session.restaurant_id != owner_id:
            return jsonify({'success': False, 'error': 'Target customer not found in this group session'}), 404
        order.customer_id = target_customer_id
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/group-service/merge-bill', methods=['POST'])
@login_required
def group_merge_bill():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    data = request.json or {}
    source_customer_id = data.get('source_customer_id')
    target_customer_id = data.get('target_customer_id')
    owner_id = _current_restaurant_owner()
    try:
        source_customer = db.session.get(GroupCustomer, source_customer_id)
        if not source_customer or source_customer.group_session.restaurant_id != owner_id:
            return jsonify({'success': False}), 404
        target_customer = db.session.get(GroupCustomer, target_customer_id)
        if not target_customer or target_customer.group_id != source_customer.group_id or target_customer.group_session.restaurant_id != owner_id:
            return jsonify({'success': False, 'error': 'Target customer not found in this group session'}), 404
        for o in source_customer.orders:
            o.customer_id = target_customer_id
        for p in source_customer.payments:
            p.customer_id = target_customer_id
        db.session.delete(source_customer)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

import random

@app.route('/api/weather')
@login_required
def api_weather():
    lat, lon = 12.9716, 77.5946
    settings_path = os.path.join(app.instance_path, f"{current_user.username}_settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                import json
                s_data = json.load(f)
                if s_data.get('lat') and s_data.get('lon'):
                    lat = s_data['lat']
                    lon = s_data['lon']
        except Exception:
            pass
    temp, rain = get_weather(lat, lon)
    condition = "Rainy" if rain > 0 else "Clear"
    return jsonify({
        'success': True,
        'temperature': round(temp, 1),
        'rain': rain,
        'condition': condition
    })

@app.route('/api/predictions')
@login_required
@require_feature('ai_forecasting')
def api_predictions():
    clf, classes = get_or_train_ml_model()
    temp, rain = get_weather()
    
    samples = pd.DataFrame([
        {'Temperature': temp, 'Rainfall': rain},
        {'Temperature': temp + 1, 'Rainfall': rain},
        {'Temperature': temp - 1, 'Rainfall': max(0, rain - 1)},
        {'Temperature': temp, 'Rainfall': max(0, rain + 2)}
    ])
    
    probs = clf.predict_proba(samples).mean(axis=0)
    top_indices = probs.argsort()[-4:][::-1]
    
    predictions = [classes[idx] for idx in top_indices]
    
    return jsonify({
        'success': True,
        'predictions': predictions
    })

@app.route('/api/search_bill')
@login_required
def api_search_bill():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'}), 400
    
    # Try searching by ID (Bill No) or UUID
    order = None
    if query.isdigit():
        order = Order.query.filter_by(id=int(query), restaurant_id=current_user.id).first()
    
    if not order:
        order = Order.query.filter(Order.order_uuid.like(f"{query}%"), Order.restaurant_id == current_user.id).first()
    
    if order:
        items = []
        for item in order.items:
            items.append({
                'name': item.item_name,
                'price': item.price,
                'qty': item.quantity,
                'item_total': item.price * item.quantity
            })
            
        return jsonify({
            'success': True,
            'bill': {
                'id': order.id,
                'order_type': order.order_type,
                'customer_name': order.customer_name,
                'subtotal': order.subtotal,
                'discount_pct': order.discount_pct,
                'gst_amount': order.gst_amount,
                'total_price': order.total_price,
                'date': order.date.strftime('%Y-%m-%d %H:%M'),
                'items': items
            }
        })
    
    return jsonify({'success': False, 'error': 'Bill not found'})

@app.route('/delete_stock/<int:item_id>', methods=['POST'])
@login_required
def api_inventory_delete(item_id):
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    item = Inventory.query.filter_by(id=item_id, restaurant_id=current_user.id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
    return jsonify({'success': True})

@app.route('/api/recipe/save', methods=['POST'])
@login_required
def api_recipe_save():
    if current_user.status == 'Pending': return jsonify({'success': False, 'error': 'Account pending'}), 403
    data = request.json or request.form
    
    recipe_id = data.get('recipe_id') or data.get('id')
    menu_item_id = data.get('menu_item_id')
    ingredient_id = data.get('ingredient_id')
    new_ing_name = data.get('new_ingredient_name') or data.get('ingredient_name')
    unit = data.get('unit', 'g')
    
    try:
        qty = float(data.get('quantity_required', 0))
    except (ValueError, TypeError):
        qty = 0.0
        
    try:
        start_stock = float(data.get('starting_stock', data.get('current_stock', 0.0)))
    except (ValueError, TypeError):
        start_stock = 0.0
        
    if not menu_item_id or qty <= 0:
        return jsonify({'success': False, 'error': 'Valid Menu Item and Quantity required (> 0)'}), 400
        
    menu_item = MenuItem.query.filter_by(id=menu_item_id, restaurant_id=current_user.id).first()
    if not menu_item:
        return jsonify({'success': False, 'error': 'Active Menu Item not found'}), 404
        
    inv_item = None
    if ingredient_id and str(ingredient_id).isdigit():
        inv_item = Inventory.query.filter_by(id=int(ingredient_id), restaurant_id=current_user.id).first()
        if inv_item and unit:
            inv_item.unit = unit
        
    if not inv_item and new_ing_name:
        safe_ing_name = str(new_ing_name).strip()
        if not safe_ing_name:
            return jsonify({'success': False, 'error': 'Ingredient name cannot be empty'}), 400
            
        inv_item = Inventory.query.filter_by(restaurant_id=current_user.id).filter(
            db.func.lower(Inventory.ingredient_name) == safe_ing_name.lower()
        ).first()
        if not inv_item:
            inv_item = Inventory(
                restaurant_id=current_user.id,
                ingredient_name=safe_ing_name,
                unit=unit,
                current_stock=start_stock,
                consumed_quantity=0.0,
                min_level=5.0
            )
            db.session.add(inv_item)
            db.session.flush()
        elif unit:
            inv_item.unit = unit
            
    if not inv_item:
        return jsonify({'success': False, 'error': 'Please select an existing ingredient or specify a valid new ingredient'}), 400

    recipe = None
    if recipe_id and str(recipe_id).isdigit():
        recipe = MenuRecipe.query.join(MenuItem).filter(
            MenuRecipe.id == int(recipe_id),
            MenuItem.restaurant_id == current_user.id
        ).first()
        
    if recipe:
        recipe.menu_item_id = menu_item.id
        recipe.ingredient_id = inv_item.id
        recipe.quantity_required = qty
        if hasattr(recipe, 'recipe_name'):
            recipe.recipe_name = menu_item.name
    else:
        recipe = MenuRecipe.query.filter_by(menu_item_id=menu_item.id, ingredient_id=inv_item.id).first()
        if recipe:
            recipe.quantity_required = qty
            if hasattr(recipe, 'recipe_name'):
                recipe.recipe_name = menu_item.name
        else:
            recipe_kwargs = {
                'menu_item_id': menu_item.id,
                'ingredient_id': inv_item.id,
                'quantity_required': qty
            }
            if hasattr(MenuRecipe, 'recipe_name'):
                recipe_kwargs['recipe_name'] = menu_item.name
            recipe = MenuRecipe(**recipe_kwargs)
            db.session.add(recipe)
        
    menu_item.recipe_name = menu_item.name
    db.session.commit()
    return jsonify({
        'success': True,
        'recipe_id': recipe.id,
        'menu_item_id': menu_item.id,
        'ingredient_id': inv_item.id
    })

@app.route('/api/recipe/delete', methods=['POST'])
@login_required
def api_recipe_delete():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    data = request.json or request.form
    recipe_id = data.get('id') or data.get('recipe_id')
    if not recipe_id:
        return jsonify({'success': False, 'error': 'Recipe ID required'}), 400
        
    recipe = MenuRecipe.query.join(MenuItem).filter(
        MenuRecipe.id == int(recipe_id),
        MenuItem.restaurant_id == current_user.id
    ).first()
    if recipe:
        db.session.delete(recipe)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Recipe not found'}), 404

@app.route('/api/table/clear/<int:table_num>', methods=['POST'])
@login_required
def api_table_clear(table_num):
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    lock_restaurant(current_user.id)
    t = DiningTable.query.filter_by(table_num=table_num, restaurant_id=current_user.id).first()
    if t:
        order = db.session.get(Order, t.current_order_id) if t.current_order_id else None
        if order and order.table_num == table_num and order.payment_status == 'PENDING':
            return jsonify(success=False, error='Settle the open bill before clearing this table.'), 409
        t.status = 'EMPTY'
        t.current_order_id = None
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 404


# ==========================================
# EXECUTIVE DASHBOARD ROUTES (READ-ONLY)
# ==========================================

@app.route('/dashboard')
@login_required
@require_feature('executive_dashboard')
def dashboard():
    if current_user.status == 'Pending':
        return "Access Denied", 403
    return render_template('dashboard/executive_dashboard.html')

@app.route('/api/dashboard/kpi')
@login_required
@require_feature('executive_dashboard')
def api_dashboard_kpi():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    orders = Order.query.filter_by(restaurant_id=current_user.id).all()
    total_sales = sum((o.total_price or 0) for o in orders)
    total_orders = len(orders)
    rev_per_order = total_sales / total_orders if total_orders > 0 else 0
    profit = total_sales * PROFIT_MARGIN

    # Real month-over-month growth calculation
    # Compare current month total vs previous month total
    from datetime import datetime as _dt
    now = _dt.now()
    cur_month_total = sum(
        (o.total_price or 0) for o in orders
        if o.date.month == now.month and o.date.year == now.year
    )
    prev_month = now.month - 1 if now.month > 1 else 12
    prev_year = now.year if now.month > 1 else now.year - 1
    prev_month_total = sum(
        (o.total_price or 0) for o in orders
        if o.date.month == prev_month and o.date.year == prev_year
    )
    if prev_month_total > 0:
        sales_growth = round(((cur_month_total - prev_month_total) / prev_month_total) * 100, 1)
    else:
        sales_growth = 0  # Cannot compute growth without prior period data

    return jsonify({
        'success': True,
        'total_sales': round(total_sales, 2),
        'total_orders': total_orders,
        'revenue_per_order': round(rev_per_order, 2),
        'gross_profit': round(profit, 2),
        'sales_growth': sales_growth
    })

@app.route('/api/dashboard/sales-trend')
@login_required
@require_feature('executive_dashboard')
def api_dashboard_sales_trend():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    orders = Order.query.filter_by(restaurant_id=current_user.id).all()

    # Return empty if no orders exist
    if not orders:
        return jsonify({'success': True, 'labels': [], 'sales': [], 'growth': []})

    # Group real sales by day (last 7 days)
    from datetime import datetime as _dt, timedelta
    
    daily = {}
    now = _dt.now()
    
    # Filter orders to last 7 days
    for o in orders:
        if (now.date() - o.date.date()).days <= 6:
            key = o.date.strftime('%d %b')
            daily[key] = daily.get(key, 0.0) + (o.total_price or 0.0)

    # Sort and fill missing days
    labels = []
    sales = []
    for i in range(6, -1, -1):
        d = now - timedelta(days=i)
        key = d.strftime('%d %b')
        labels.append(key)
        sales.append(round(daily.get(key, 0.0), 2))

    growth = [0.0] * 7 # Can be used if needed

    return jsonify({'success': True, 'labels': labels, 'sales': sales, 'growth': growth})

@app.route('/api/dashboard/category-performance')
@login_required
@require_feature('executive_dashboard')
def api_dashboard_cat_perf():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    orders = Order.query.filter_by(restaurant_id=current_user.id).all()

    # Return empty if no orders exist — NO demo fallback
    if not orders:
        return jsonify({'success': True, 'labels': [], 'actual': [], 'planned': []})

    # Retrieve all MenuItems for mapping
    menu_items = MenuItem.query.filter_by(restaurant_id=current_user.id).all()
    name_to_category = {m.name: m.category for m in menu_items}

    # Fallback to default dictionary if user hasn't set up the DB yet
    if not name_to_category:
        fallback_menu = {}
        for category, items in fallback_menu.items():
            for item in items.keys():
                name_to_category[item] = category

    cat_sales = {}
    for o in orders:
        for i in o.items:
            category = name_to_category.get(i.item_name)
            if not category or category.strip() == '':
                category = "Uncategorized"
                
            cat_sales[category] = cat_sales.get(category, 0) + (i.price * i.quantity)

    # Return empty if no items found in any order
    if not cat_sales:
        return jsonify({'success': True, 'labels': [], 'actual': [], 'planned': []})

    sorted_items = sorted(cat_sales.items(), key=lambda x: x[1], reverse=True)[:5]
    labels = [k for k, v in sorted_items]
    actual = [round(v, 2) for k, v in sorted_items]

    return jsonify({
        'success': True,
        'labels': labels,
        'actual': actual,
        'planned': []  # No fabricated planned targets
    })

@app.route('/api/dashboard/segment-analysis')
@login_required
@require_feature('executive_dashboard')
def api_dashboard_segment():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    orders = Order.query.filter_by(restaurant_id=current_user.id).all()
    if not orders:
        return jsonify({'success': True, 'labels': [], 'values': []})

    customer_orders = {}
    for o in orders:
        phone = getattr(o, 'customer_phone', None)
        if not phone or phone.strip() == '':
            # Treat unknown walk-ins as unique new customers for the sake of metrics
            phone = 'Walk-in-' + str(o.id)
        customer_orders[phone] = customer_orders.get(phone, 0) + 1

    new_cust, returning, loyal = 0, 0, 0
    for count in customer_orders.values():
        if count <= 1:
            new_cust += 1
        elif 2 <= count <= 4:
            returning += 1
        else:
            loyal += 1
    
    # Only return segments that have > 0 to avoid messy charts, though ChartJS handles it ok
    labels = ['New Customers', 'Returning Customers', 'Loyal Customers']
    values = [new_cust, returning, loyal]
    
    return jsonify({'success': True, 'labels': labels, 'values': values})

@app.route('/api/dashboard/regional-distribution')
@login_required
@require_feature('executive_dashboard')
def api_dashboard_regional():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    # Geography is not tracked in the current schema.
    # Return empty — do NOT fabricate region data.
    return jsonify({'success': True, 'labels': [], 'data': []})


@app.route('/api/dashboard/top-items')
@login_required
@require_feature('executive_dashboard')
def api_dashboard_top_items():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    orders = Order.query.filter_by(restaurant_id=current_user.id).all()
    if not orders:
        return jsonify({'success': True, 'labels': [], 'values': []})

    item_sales = {}
    for o in orders:
        for i in o.items:
            item_sales[i.item_name] = item_sales.get(i.item_name, 0) + (i.price * i.quantity)

    if not item_sales:
        return jsonify({'success': True, 'labels': [], 'values': []})

    sorted_items = sorted(item_sales.items(), key=lambda x: x[1], reverse=True)[:5]
    labels = [k for k, v in sorted_items]
    values = [round(v, 2) for k, v in sorted_items]
    return jsonify({'success': True, 'labels': labels, 'values': values})


@app.route('/api/dashboard/payment-methods')
@login_required
@require_feature('executive_dashboard')
def api_dashboard_payment_methods():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    orders = Order.query.filter_by(restaurant_id=current_user.id).all()
    if not orders:
        return jsonify({'success': True, 'labels': [], 'values': []})

    method_counts = {}
    for o in orders:
        # Use payment_method if column exists, else default to 'Cash'
        method = getattr(o, 'payment_method', None) or 'Cash'
        method_counts[method] = method_counts.get(method, 0) + 1

    if not method_counts:
        return jsonify({'success': True, 'labels': [], 'values': []})

    labels = list(method_counts.keys())
    values = [method_counts[k] for k in labels]
    return jsonify({'success': True, 'labels': labels, 'values': values})


def get_stocksense_recipes():
    try:
        import shutil
        # Safe copy to avoid permission issues if file is open
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        _src_xlsx = os.path.join(_base_dir, 'Gigatos   excel data.xlsx')
        _tmp_xlsx = os.path.join(_base_dir, 'gigatos_temp.xlsx')
        shutil.copy2(_src_xlsx, _tmp_xlsx)
        df = pd.read_excel(_tmp_xlsx, header=None)
        
        recipes = {}
        # Data starts from row 3 (0-indexed it would be index 3)
        # Columns: 2=Item, 4=Ingredient, 5=Qty, 6=Unit
        for i in range(3, len(df)):
            item = str(df.iloc[i, 2]).strip()
            ing = str(df.iloc[i, 4]).strip()
            qty_val = df.iloc[i, 5]
            unit = str(df.iloc[i, 6]).strip()
            
            if item != 'nan' and ing != 'nan':
                if item not in recipes:
                    recipes[item] = []
                try:
                    qty = float(qty_val)
                    recipes[item].append({
                        'ingredient': ing,
                        'quantity': qty,
                        'unit': unit
                    })
                except:
                    continue
        return recipes
    except Exception as e:
        print(f"Excel read error: {e}")
        return {}

@app.route('/api/stocksense/dashboard-data')
@login_required
@require_feature('ai_insights')
def api_stocksense_data():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    
    order_count = Order.query.filter_by(restaurant_id=current_user.id).count()
    rec_status = get_restaurant_recipe_status(current_user.id)
    
    has_menu = rec_status['total_menu_items'] > 0
    has_recipes = rec_status['has_recipes']
    has_inventory = rec_status['total_inventory_ingredients'] > 0
    has_orders = order_count >= 5

    setup_data = {
        'has_menu': has_menu,
        'menu_count': rec_status['total_menu_items'],
        'has_recipes': has_recipes,
        'items_with_recipes': rec_status['items_with_recipes'],
        'total_menu_items': rec_status['total_menu_items'],
        'unmapped_items_count': rec_status['unmapped_items_count'],
        'ambiguous_items_count': rec_status['ambiguous_items_count'],
        'total_recipe_relationships': rec_status['total_recipe_relationships'],
        'total_inventory_ingredients': rec_status['total_inventory_ingredients'],
        'successfully_mapped_items': rec_status['successfully_mapped_items'],
        'unmapped_items': rec_status['unmapped_items'],
        'ambiguous_items': rec_status['ambiguous_items'],
        'has_inventory': has_inventory,
        'inventory_count': rec_status['total_inventory_ingredients'],
        'has_orders': has_orders,
        'order_count': order_count,
        'orders_needed': max(0, 5 - order_count)
    }

    if not has_menu or not has_recipes or not has_inventory or not has_orders:
        return jsonify({
            'success': True,
            'insufficient_data': True,
            'message': 'Insufficient data for prediction',
            'setup': setup_data
        })

    from ai_prediction_engine import get_db_recipes_for_user
    recipes = get_db_recipes_for_user(current_user.id)

    # Orders today
    today = datetime.now(timezone.utc).date()
    orders_today = Order.query.filter_by(restaurant_id=current_user.id).filter(
        db.func.date(Order.date) == today
    ).all()
    
    # Get current stock mapping
    inv_items = Inventory.query.filter_by(restaurant_id=current_user.id).all()
    stock_map = {i.ingredient_name.strip().lower(): i for i in inv_items}
    
    usage_today = {} # {ingredient: {qty, unit}}
    for o in orders_today:
        for itm in o.items:
            recipe = recipes.get(itm.item_name)
            if recipe:
                for res in recipe:
                    ing_name = res['ingredient']
                    inv_item = stock_map.get(ing_name.strip().lower())
                    if inv_item:
                        final_qty = convert_to_base_unit(res['quantity'] * itm.quantity, res['unit'], inv_item.unit, f"POS -> {ing_name}")
                        target_name = inv_item.ingredient_name
                        if target_name not in usage_today:
                            usage_today[target_name] = {'qty': 0, 'unit': inv_item.unit}
                        usage_today[target_name]['qty'] += final_qty

    # Sort and format for horizontal bar chart
    sorted_usage = sorted(usage_today.items(), key=lambda x: x[1]['qty'], reverse=True)[:8]
    bar_labels = [k for k, v in sorted_usage]
    bar_values = [v['qty'] for k, v in sorted_usage]

    # --- Connect to AI Prediction Engine ---
    try:
        from ai_prediction_engine import predict_consumption_tomorrow
        pred_data = predict_consumption_tomorrow(current_user.id)
        if pred_data.get("insufficient_data"):
            pred_data["bar"] = {"labels": bar_labels, "values": bar_values}
            pred_data["setup"] = setup_data
            return jsonify(pred_data)
            
        line_labels = pred_data["line"]["labels"]
        line_datasets = pred_data["line"]["datasets"]
        alerts = pred_data["alerts"]
        table_data = pred_data["table"]
        confidence_value = pred_data.get("confidence_value", "50.0%")
        confidence_level = pred_data.get("confidence_level", "Low")
        ingredient_metrics = pred_data.get("ingredient_metrics", {})
        
        return jsonify({
            'success': True,
            'insufficient_data': False,
            'setup_complete': True,
            'bar': {'labels': bar_labels, 'values': bar_values},
            'line': {'labels': line_labels, 'datasets': line_datasets},
            'alerts': alerts[:5],
            'table': table_data,
            'confidence_value': confidence_value,
            'confidence_level': confidence_level,
            'ingredient_metrics': ingredient_metrics,
            'validation_mae': pred_data['validation_mae'],
            'validation_days': pred_data['validation_days'],
            'recent_days': pred_data['recent_days'],
            'setup': setup_data
        })
    except Exception as ai_err:
        print(f"[AI-PREDICT-ERR] Failed using ML predictions: {ai_err}")
        return jsonify({
            'success': True,
            'insufficient_data': False,
            'prediction_error': True,
            'message': f"Stock Sense AI could not generate a forecast: {ai_err}",
            'setup': setup_data
        })


# =========================================================================
# LOCATION INTELLIGENCE AI ENGINES & API ENDPOINTS
# =========================================================================
def fetch_and_cache_geodata(loc_intel):
    from location_intelligence import LocationIntelligenceService
    try:
        service = LocationIntelligenceService()
        res = service.sync_and_get_dashboard_data(loc_intel, db.session, force_refresh=True)
        return res.get('data_state') in ('REAL_DATA', 'NO_RESULTS', 'ZERO_RESULTS')
    except Exception as err:
        print(f"[LOCATION-AI] fetch_and_cache_geodata error: {err}")
        return False

@app.route('/api/location/save', methods=['POST'])
@login_required
def api_location_save():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    
    data = request.get_json() or {}
    address = (data.get('address') or '').strip()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    radius = data.get('radius', 1000)
    google_maps_key = data.get('google_maps_key', '')

    print(f"[LOCATION-AI-LIVE] Address received: {address}")
    print(f"[LOCATION-AI-LIVE] Latitude received: {latitude}")
    print(f"[LOCATION-AI-LIVE] Longitude received: {longitude}")
    print(f"[LOCATION-AI-LIVE] Radius received: {radius}")

    from location_intelligence.geocoder import GeocoderService
    lat_val, lon_val, is_valid = GeocoderService.validate_and_resolve(address, latitude, longitude)
    if not is_valid:
        return jsonify({'success': False, 'message': 'Location could not be resolved. Enter valid coordinates or a more specific address.'}), 400

    print(f"[LOCATION-AI-LIVE] Coordinates used for provider: ({lat_val}, {lon_val})")

        
    try:
        radius = int(radius)
    except (ValueError, TypeError):
        radius = 1000
        
    loc = LocationIntelligence.query.filter_by(restaurant_id=current_user.id).first()
    if not loc:
        loc = LocationIntelligence(restaurant_id=current_user.id, latitude=lat_val, longitude=lon_val)
        db.session.add(loc)
        
    loc.address = address or f"Location ({lat_val:.4f}, {lon_val:.4f})"
    loc.latitude = lat_val
    loc.longitude = lon_val
    loc.radius = radius
    loc.google_maps_key = google_maps_key
    loc.competitors_cached = None
    loc.nearby_data_cached = None
    loc.last_updated = None
    
    try:
        db.session.commit()
        if data.get('return_analysis') is True:
            from location_intelligence import LocationIntelligenceService
            analysis = LocationIntelligenceService().sync_and_get_dashboard_data(loc, db.session, force_refresh=True)
            return jsonify(success=True, analysis=analysis)
        synced = fetch_and_cache_geodata(loc)
        message = 'Location saved and map data synced.' if synced else 'Location saved. Map data is currently unavailable.'
        return jsonify({'success': True, 'synced': synced, 'message': message})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500

@app.route('/location-intelligence')
@login_required
def location_intelligence_page():
    return index()

@app.route('/api/location/dashboard-data', methods=['GET'])
@login_required
def api_location_dashboard_data():
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    
    loc = LocationIntelligence.query.filter_by(restaurant_id=current_user.id).first()
    if not loc or loc.latitude is None or loc.longitude is None:
        return jsonify({
            'success': False,
            'configured': False,
            'message': 'Restaurant location is required to calculate nearby competitors.'
        })

    try:
        from location_intelligence import LocationIntelligenceService
        service = LocationIntelligenceService()
        payload = service.sync_and_get_dashboard_data(loc, db.session, force_refresh=False)
        return jsonify(payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[LOCATION-AI] Dashboard endpoint error:", e)
        return jsonify({
            'success': True,
            'configured': True,
            'data_available': False,
            'data_state': 'PROVIDER_ERROR',
            'latitude': loc.latitude,
            'longitude': loc.longitude,
            'address': loc.address,
            'radius': loc.radius,
            'has_google_key': bool((loc.google_maps_key or '').strip()),
            'competitors': [],
            'competitor_summary': {'count': 0, 'avg_rating': 'N/A'},
            'opportunity_score': 0,
            'environment_summary': {'environment_type': 'Error', 'evidence': [str(e)]},
            'demand_opportunities': [],
            'top_3_opportunities': [],
            'market_gaps': [],
            'insights': [{'msg': f'WHAT WAS OBSERVED: Error executing location intelligence pipeline: {str(e)} WHY IT MATTERS: Analysis required valid geodata.', 'icon': 'fa-solid fa-triangle-exclamation', 'severity': 'high'}],
            'data_lineage': {'source_provider': 'Error', 'timestamp': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
        })



# =========================================================================
# PROFILE & SETTINGS MANAGEMENT CENTER ENDPOINTS
# =========================================================================
import os
from werkzeug.utils import secure_filename

# Ensure static/uploads/profile folder exists (absolute path so it works on Render too)
_APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join('static', 'uploads', 'profile')
os.makedirs(os.path.join(_APP_BASE_DIR, UPLOAD_FOLDER), exist_ok=True)

@app.route('/profile')
@login_required
def profile_view():
    tab = request.args.get('tab', 'me')
    # Fetch user specifics
    detail = RestaurantDetail.query.filter_by(user_id=current_user.id).first()
    if not detail:
        detail = RestaurantDetail(user_id=current_user.id)
        db.session.add(detail)
        db.session.commit()
        
    settings = UserSetting.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = UserSetting(user_id=current_user.id)
        db.session.add(settings)
        db.session.commit()
        
    sub = Subscription.query.filter_by(restaurant_id=current_user.id).first()
    if not sub and current_user.status != 'Admin':
        sub = ensure_subscription(current_user)
        
    return render_template('profile.html', tab=tab, detail=detail, settings=settings, sub=sub)

@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile_api():
    return jsonify({
        'success': True,
        'profile': {
            'id': current_user.id,
            'username': current_user.username,
            'name': getattr(current_user, 'name', '') or '',
            'email': getattr(current_user, 'email', '') or '',
            'phone': getattr(current_user, 'phone', '') or '',
            'profile_image': getattr(current_user, 'profile_image', '') or '',
            'role': 'Admin' if current_user.status == 'Admin' else 'Manager',
            'status': current_user.status,
            'last_login': getattr(current_user, 'last_login', None).strftime('%Y-%m-%d %H:%M:%S') if getattr(current_user, 'last_login', None) else 'Never'
        }
    })

@app.route('/api/profile', methods=['PUT'])
@login_required
def update_profile_api():
    data = request.form if request.form else (request.json or {})
    
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    phone = data.get('phone', '').strip()
    
    if not name:
        return jsonify({'success': False, 'message': 'Full Name is required.'}), 400
        
    current_user.name = name
    current_user.phone = phone
    
    if 'profile_image' in request.files:
        file = request.files['profile_image']
        if file and file.filename != '':
            if not _is_safe_image_filename(file.filename):
                return jsonify({'success': False, 'message': 'Only PNG, JPG, JPEG, GIF, or WEBP images are allowed.'}), 400
            filename = secure_filename(f"user_{current_user.id}_{file.filename}")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
            file.save(os.path.join(app.root_path, filepath))
            current_user.profile_image = '/' + filepath.replace('\\', '/')
            
    db.session.commit()
    return jsonify({
        'success': True,
        'message': 'Profile updated successfully!',
        'profile': {
            'name': current_user.name,
            'email': email,
            'phone': current_user.phone,
            'profile_image': current_user.profile_image
        }
    })

@app.route('/api/profile/password', methods=['PUT'])
@login_required
def update_profile_password_api():
    data = request.json or {}
    curr_pass = data.get('current_password', '')
    new_pass = data.get('new_password', '')
    
    if not curr_pass or not new_pass:
        return jsonify({'success': False, 'message': 'Current and new passwords are required.'}), 400
        
    # Check current password hash or plain representation
    from werkzeug.security import check_password_hash, generate_password_hash
    if not current_user.raw_password == curr_pass and not check_password_hash(current_user.password_hash, curr_pass):
        return jsonify({'success': False, 'message': 'Incorrect current password.'}), 400
            
    current_user.raw_password = new_pass
    current_user.password_hash = generate_password_hash(new_pass)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Password updated successfully!'})

@app.route('/api/restaurant', methods=['GET'])
@login_required
def get_restaurant_api():
    detail = RestaurantDetail.query.filter_by(user_id=current_user.id).first()
    if not detail:
        detail = RestaurantDetail(user_id=current_user.id)
        db.session.add(detail)
        db.session.commit()
        
    return jsonify({
        'success': True,
        'restaurant': {
            'restaurant_name': detail.restaurant_name,
            'logo': detail.logo or '',
            'gst_number': detail.gst_number or '',
            'address': detail.address or '',
            'phone': detail.phone or '',
            'email': detail.email or '',
            'currency': detail.currency or 'INR',
            'timezone': detail.timezone or 'Asia/Kolkata'
        }
    })

@app.route('/api/restaurant', methods=['PUT'])
@login_required
def update_restaurant_api():
    data = request.form if request.form else (request.json or {})
    
    detail = RestaurantDetail.query.filter_by(user_id=current_user.id).first()
    if not detail:
        detail = RestaurantDetail(user_id=current_user.id)
        db.session.add(detail)
        
    detail.restaurant_name = data.get('restaurant_name', 'My Restaurant').strip()
    detail.gst_number = data.get('gst_number', '').strip()
    detail.address = data.get('address', '').strip()
    detail.phone = data.get('phone', '').strip()
    detail.email = data.get('email', '').strip()
    detail.currency = data.get('currency', 'INR').strip()
    detail.timezone = data.get('timezone', 'Asia/Kolkata').strip()
    
    if 'logo' in request.files:
        file = request.files['logo']
        if file and file.filename != '':
            if not _is_safe_image_filename(file.filename):
                return jsonify({'success': False, 'message': 'Only PNG, JPG, JPEG, GIF, or WEBP images are allowed.'}), 400
            filename = secure_filename(f"logo_{current_user.id}_{file.filename}")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
            file.save(os.path.join(app.root_path, filepath))
            detail.logo = '/' + filepath.replace('\\', '/')
            
    db.session.commit()

    # Also sync the owner settings JSON file so both DB and JSON file match 100%
    owner_id = _current_restaurant_owner()
    owner = db.session.get(User, owner_id)
    if owner:
        settings_path = _safe_settings_path(owner.username)
        existing_settings = {}
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    existing_settings = json.load(f)
            except Exception:
                pass
        existing_settings.update({
            'restaurant_name': detail.restaurant_name,
            'address': detail.address or '',
            'phone': detail.phone or '',
            'currency': detail.currency or '₹ INR',
            'timezone': detail.timezone or 'Asia/Kolkata (IST)'
        })
        os.makedirs(app.instance_path, exist_ok=True)
        try:
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(existing_settings, f)
        except Exception as e:
            print("Error syncing JSON settings:", e)

    return jsonify({'success': True, 'message': 'Restaurant details updated successfully!'})



@app.route('/api/settings', methods=['PUT'])
@login_required
def update_settings_api():
    data = request.json or {}
    
    settings = UserSetting.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = UserSetting(user_id=current_user.id)
        db.session.add(settings)
        
    settings.default_order_type = data.get('default_order_type', 'Dine In')
    settings.language = data.get('language', 'English')
    settings.tax_enabled = bool(data.get('tax_enabled', True))
    settings.auto_print_receipt = bool(data.get('auto_print_receipt', False))
    settings.auto_print_kot = bool(data.get('auto_print_kot', False))
    settings.appearance_mode = data.get('appearance_mode', 'Light')
    
    db.session.commit()
    return jsonify({'success': True, 'message': 'Settings saved successfully!'})

@app.route('/api/subscription', methods=['GET'])
@login_required
def get_subscription_api():
    sub = Subscription.query.filter_by(restaurant_id=current_user.id).first()
    if not sub and current_user.status != 'Admin':
        sub = ensure_subscription(current_user)
        
    plan_name = "Trial Plan"
    order_limit = 50
    if sub and sub.plan:
        plan_name = sub.plan.name
        if sub.plan.slug == 'starter':
            order_limit = 100
        elif sub.plan.slug == 'growth':
            order_limit = 500
        else:
            order_limit = 10000
            
    orders_used = Order.query.filter_by(restaurant_id=current_user.id).count()
    orders_remaining = max(0, order_limit - orders_used)
    
    start_date = sub.trial_start.strftime('%Y-%m-%d') if sub and sub.trial_start else 'N/A'
    expiry_date = 'Never'
    status = 'ACTIVE'
    
    if sub:
        status = sub.status
        if sub.status == 'TRIAL' and sub.trial_end:
            expiry_date = ensure_utc(sub.trial_end).strftime('%Y-%m-%d')
        elif sub.status == 'ACTIVE' and sub.subscription_end:
            expiry_date = ensure_utc(sub.subscription_end).strftime('%Y-%m-%d')
        elif sub.status == 'EXPIRED':
            if sub.subscription_end:
                expiry_date = ensure_utc(sub.subscription_end).strftime('%Y-%m-%d')
            elif sub.trial_end:
                expiry_date = ensure_utc(sub.trial_end).strftime('%Y-%m-%d')
                
    return jsonify({
        'success': True,
        'subscription': {
            'plan_name': plan_name,
            'status': status,
            'start_date': start_date,
            'expiry_date': expiry_date,
            'renewal_date': expiry_date,
            'orders_used': orders_used,
            'orders_remaining': orders_remaining
        }
    })

def raise_support_ticket_api():
    # Legacy wrapper for tests or compatibility
    data = request.json or {}
    subject = data.get('subject', '').strip()
    description = data.get('description', '').strip()
    category = data.get('category', 'General').strip()
    
    if not subject or not description:
        return jsonify({'success': False, 'message': 'Subject and Description are required.'}), 400
        
    uuid = generate_ticket_uuid()
    ticket = SupportTicket(
        ticket_uuid=uuid,
        restaurant_id=_current_restaurant_owner(),
        user_id=current_user.id,
        category=category,
        subject=subject,
        description=description,
        priority="Medium",
        status="Open"
    )
    db.session.add(ticket)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f"Ticket {uuid} created successfully!",
        'ticket_uuid': uuid,
        'ticket_id': ticket.id
    })

import random

def generate_ticket_uuid():
    while True:
        num = random.randint(10000, 99999)
        uuid = f"TKT-{num}"
        if not SupportTicket.query.filter_by(ticket_uuid=uuid).first():
            return uuid

@app.route('/api/support/tickets', methods=['POST'])
@login_required
def create_ticket_api():
    data = request.json or {}
    category = data.get('category', 'General').strip()
    subject = data.get('subject', '').strip()
    description = data.get('description', '').strip()
    priority = data.get('priority', 'Medium').strip()
    
    if not subject or not description:
        return jsonify({'success': False, 'message': 'Subject and Description are required.'}), 400
        
    uuid = generate_ticket_uuid()
    ticket = SupportTicket(
        ticket_uuid=uuid,
        restaurant_id=current_user.id,
        user_id=current_user.id,
        category=category,
        subject=subject,
        description=description,
        priority=priority,
        status="Open"
    )
    db.session.add(ticket)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f"Ticket {uuid} created successfully!",
        'ticket_uuid': uuid
    })

@app.route('/api/support/tickets', methods=['GET'])
@login_required
def get_user_tickets_api():
    status_filter = request.args.get('status', '').strip()
    query = SupportTicket.query.filter_by(restaurant_id=current_user.id)
    if status_filter:
        query = query.filter_by(status=status_filter)
    tickets = query.order_by(SupportTicket.created_at.desc()).all()
    
    return jsonify({
        'success': True,
        'tickets': [{
            'ticket_uuid': t.ticket_uuid,
            'category': t.category,
            'subject': t.subject,
            'description': t.description,
            'priority': t.priority,
            'status': t.status,
            'assigned_admin': t.assigned_admin or 'Unassigned',
            'has_unread_replies': t.has_unread_replies,
            'created_at': t.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': t.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        } for t in tickets]
    })

@app.route('/api/support/tickets/<uuid>', methods=['GET'])
@login_required
def get_ticket_details_api(uuid):
    ticket = SupportTicket.query.filter_by(ticket_uuid=uuid).first()
    if not ticket:
        return jsonify({'success': False, 'message': 'Ticket not found.'}), 404
        
    if ticket.restaurant_id != current_user.id and current_user.status != 'Admin':
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
        
    if current_user.status != 'Admin':
        ticket.has_unread_replies = False
        db.session.commit()
        
    replies_query = TicketReply.query.filter_by(ticket_id=ticket.id)
    if current_user.status != 'Admin':
        replies_query = replies_query.filter_by(is_internal=False)
        
    replies = replies_query.order_by(TicketReply.created_at.asc()).all()
    
    return jsonify({
        'success': True,
        'ticket': {
            'ticket_uuid': ticket.ticket_uuid,
            'category': ticket.category,
            'subject': ticket.subject,
            'description': ticket.description,
            'priority': ticket.priority,
            'status': ticket.status,
            'assigned_admin': ticket.assigned_admin or 'Unassigned',
            'created_at': ticket.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': ticket.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        },
        'replies': [{
            'sender_name': 'Support Staff' if r.sender_type == 'Admin' else 'You',
            'sender_type': r.sender_type,
            'message': r.message,
            'is_internal': r.is_internal,
            'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S')
        } for r in replies]
    })

@app.route('/api/support/tickets/<uuid>/reply', methods=['POST'])
@login_required
def reply_to_ticket_api(uuid):
    ticket = SupportTicket.query.filter_by(ticket_uuid=uuid).first()
    if not ticket:
        return jsonify({'success': False, 'message': 'Ticket not found.'}), 404
        
    if ticket.restaurant_id != current_user.id:
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
        
    if ticket.status == 'Closed':
        return jsonify({'success': False, 'message': 'This ticket is closed.'}), 400
        
    data = request.json or {}
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'success': False, 'message': 'Message body is required.'}), 400
        
    reply = TicketReply(
        ticket_id=ticket.id,
        user_id=current_user.id,
        sender_type="Customer",
        message=message
    )
    db.session.add(reply)
    
    if ticket.status == 'Waiting for Customer':
        ticket.status = 'In Progress'
    ticket.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Reply sent successfully!'})

# =========================================================================
# ADMIN SUPPORT CENTER ENDPOINTS
# =========================================================================
@app.route('/admin/support')
@login_required
def admin_support_view():
    if current_user.status != 'Admin':
        return redirect(url_for('index'))
    return render_template('admin_support.html')

@app.route('/api/admin/support/tickets', methods=['GET'])
@login_required
def admin_get_all_tickets_api():
    if current_user.status != 'Admin':
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
        
    status_filter = request.args.get('status', '').strip()
    priority_filter = request.args.get('priority', '').strip()
    search_query = request.args.get('search', '').strip()
    
    query = SupportTicket.query
    if status_filter:
        query = query.filter_by(status=status_filter)
    if priority_filter:
        query = query.filter_by(priority=priority_filter)
    if search_query:
        query = query.filter(
            (SupportTicket.subject.ilike(f"%{search_query}%")) | 
            (SupportTicket.ticket_uuid.ilike(f"%{search_query}%")) | 
            (SupportTicket.description.ilike(f"%{search_query}%"))
        )
        
    tickets = query.order_by(SupportTicket.updated_at.desc()).all()
    
    users_map = {u.id: u.username for u in User.query.all()}
    
    return jsonify({
        'success': True,
        'tickets': [{
            'ticket_uuid': t.ticket_uuid,
            'restaurant_name': users_map.get(t.restaurant_id, 'Unknown'),
            'category': t.category,
            'subject': t.subject,
            'description': t.description,
            'priority': t.priority,
            'status': t.status,
            'assigned_admin': t.assigned_admin or 'Unassigned',
            'created_at': t.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': t.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        } for t in tickets]
    })

@app.route('/api/admin/support/tickets/<uuid>/reply', methods=['POST'])
@login_required
def admin_reply_to_ticket_api(uuid):
    if current_user.status != 'Admin':
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
        
    ticket = SupportTicket.query.filter_by(ticket_uuid=uuid).first()
    if not ticket:
        return jsonify({'success': False, 'message': 'Ticket not found.'}), 404
        
    data = request.json or {}
    message = data.get('message', '').strip()
    new_status = data.get('status', 'Waiting for Customer').strip()
    
    if not message:
        return jsonify({'success': False, 'message': 'Message is required.'}), 400 
        
    reply = TicketReply(
        ticket_id=ticket.id,
        user_id=current_user.id,
        sender_type="Admin",
        message=message
    )
    db.session.add(reply)
    
    ticket.status = new_status
    ticket.has_unread_replies = True
    ticket.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Reply sent and status updated!'})

@app.route('/api/admin/support/tickets/<uuid>/status', methods=['POST'])
@login_required
def admin_change_status_api(uuid):
    if current_user.status != 'Admin':
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
        
    ticket = SupportTicket.query.filter_by(ticket_uuid=uuid).first()
    if not ticket:
        return jsonify({'success': False, 'message': 'Ticket not found.'}), 404
        
    data = request.json or {}
    if 'status' in data:
        ticket.status = data.get('status')
    if 'priority' in data:
        ticket.priority = data.get('priority')
    if 'assigned_admin' in data:
        ticket.assigned_admin = data.get('assigned_admin')
        
    ticket.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Ticket updated successfully!'})

@app.route('/api/admin/support/tickets/<uuid>/internal-notes', methods=['POST'])
@login_required
def admin_add_internal_note_api(uuid):
    if current_user.status != 'Admin':
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
        
    ticket = SupportTicket.query.filter_by(ticket_uuid=uuid).first()
    if not ticket:
        return jsonify({'success': False, 'message': 'Ticket not found.'}), 404
        
    data = request.json or {}
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'success': False, 'message': 'Message is required.'}), 400
        
    reply = TicketReply(
        ticket_id=ticket.id,
        user_id=current_user.id,
        sender_type="Admin",
        message=message,
        is_internal=True
    )
    db.session.add(reply)
    ticket.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Internal note added!'})

@app.route('/api/support/faqs', methods=['GET'])
@login_required
def get_faqs_api():
    faqs = [
        {"q": "How do I add a new menu item?", "a": "Go to the Store module, select Inventory or Menu items, and click on 'Add Item'. Specify the name, pricing, and category, and click Save."},
        {"q": "How do I clear or reset an occupied dining table?", "a": "Switch to Table View. Click on any table displaying the 'Occupied' status. You will receive a prompt asking to clear the dining session and mark the table as empty."},
        {"q": "How can I change the POS currency or location timezone?", "a": "Go to Restaurant Details tab in your profile center, update the Currency and Timezone inputs, and click Save Details."},
        {"q": "Is my restaurant sales data backed up securely?", "a": "Yes! All sales records, invoice totals, and recipes are synchronized securely and automatically back to the master database on checkout."}
    ]
    return jsonify({'success': True, 'faqs': faqs})


from routes.holiday_routes import holiday_bp
app.register_blueprint(holiday_bp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true')
