from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import requests
import os
from datetime import datetime, timezone
from flask import send_file

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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neodine-super-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///saas_customers.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    raw_password = db.Column(db.String(200))  # Admin visible password
    # 'Pending', 'Approved', 'Admin'
    status = db.Column(db.String(20), default='Pending')
    orders = db.relationship('Order', backref='restaurant', lazy=True)


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
    items = db.relationship('OrderItem', backref='order_ref', lazy=True)


class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    item_name = db.Column(db.String(100))
    price = db.Column(db.Float)

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

class Inventory(db.Model):
    __tablename__ = 'inventory'
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ingredient_name = db.Column(db.String(100), nullable=False)
    unit = db.Column(db.String(20), default='kg') # kg, ml, L, pieces
    current_stock = db.Column(db.Float, default=0.0)
    min_level = db.Column(db.Float, default=5.0)
    supplier = db.Column(db.String(150), nullable=True)
    last_updated = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class MenuRecipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    menu_item_id = db.Column(db.Integer, db.ForeignKey('menu_item.id'), nullable=False)
    ingredient_id = db.Column(db.Integer, db.ForeignKey('inventory.id'), nullable=False)
    quantity_required = db.Column(db.Float, nullable=False)
    
    menu_item = db.relationship('MenuItem', backref=db.backref('recipes', cascade='all, delete-orphan'))
    ingredient = db.relationship('Inventory', backref='used_in_recipes')

class DiningTable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    restaurant_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    table_num = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='EMPTY') # EMPTY or DINING
    current_order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=True)
    
    order = db.relationship('Order', foreign_keys=[current_order_id], lazy=True)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()
    # Auto-seed the default administrator account
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password_hash=generate_password_hash('admin123'),
            status='Admin')
        admin.raw_password = 'admin123'
        db.session.add(admin)
        db.session.commit()

API_KEY = "YOUR_API_KEY_HERE"
CITY = "Bengaluru"
DATABASE_FILE = "restaurant_database.xlsx"
PROFIT_MARGIN = 0.30

# Internal storage for pending orders to display on receipt
PENDING_ORDERS = {}

# Global ML State
GLOBAL_ML_MODEL = None
GLOBAL_ML_CLASSES = None


def get_or_train_ml_model():
    global GLOBAL_ML_MODEL, GLOBAL_ML_CLASSES
    if GLOBAL_ML_MODEL is not None:
        return GLOBAL_ML_MODEL, GLOBAL_ML_CLASSES

    # Extract historical training data from the secure SQL database instead of
    # Excel
    historical_items = db.session.query(
        Order.temperature, Order.rainfall, OrderItem.item_name) .join(
        OrderItem, Order.id == OrderItem.order_id).all()

    # Convert SQL rows to pandas Dataframe just for ML training
    if historical_items:
        df = pd.DataFrame(
            historical_items,
            columns=[
                'Temperature',
                'Rainfall',
                'Item'])
    else:
        df = pd.DataFrame(columns=['Temperature', 'Rainfall', 'Item'])

    # 1. Synthesize baseline data if DB is empty or very small to prevent
    # cold-starts
    if len(df) < 10:
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

    # 2. Train Ensemble Model
    X = train_df[['Temperature', 'Rainfall']]
    y = train_df['Item']

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
    lr = LogisticRegression(max_iter=3000, random_state=42)

    clf = VotingClassifier(
        estimators=[
            ('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')
    clf.fit(X, y)

    GLOBAL_ML_MODEL = clf
    GLOBAL_ML_CLASSES = clf.classes_
    return GLOBAL_ML_MODEL, GLOBAL_ML_CLASSES


def get_weather():
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            "latitude=12.9716&longitude=77.5946&current_weather=true"
        )
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            current = data.get("current_weather", {})
            import random

            # Base temperature + realistic random variation (-2 to +2 degrees)
            base_temp = int(current.get("temperature", 30))
            temperature = base_temp + random.choice([-2, -1, 0, 1, 2])

            # Basic rainfall logic mapping with simulated randomness
            weathercode = current.get("weathercode", 0)
            rainfall = 5 if weathercode >= 50 else 0

            # Occasionally simulate high rain to show ML behavior change
            # dynamically
            if rainfall == 0 and random.random() > 0.8:
                rainfall = random.randint(1, 15)

            return temperature, rainfall
        return 30, 0
    except Exception as e:
        print("Weather fetch error:", e)
        return 30, 0


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
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
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = User.query.filter(
            db.func.lower(User.username) == username.lower()
        ).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.status == 'Admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('index'))
        return render_template('login.html',
                               error="Invalid credentials. Try again.")
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


def update_customer_excel(username, order_data):
    filename = f"{username}_database.xlsx"
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


@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.status != 'Admin':
        return "Access Denied", 403
    pending_users = User.query.filter_by(status='Pending').all()
    approved_users = User.query.filter_by(status='Approved').all()

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
        df_gigatos = pd.read_excel('Gigatos   excel data.xlsx', header=None)

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

    return render_template('admin.html',
                           pending_users=pending_users,
                           approved_users=approved_users,
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
        filename = f"{user.username}_database.xlsx"
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

    # Fetch real SQLite items from DB
    db_items = MenuItem.query.filter_by(restaurant_id=current_user.id).all()
    if db_items:
        categorized_menu = {}
        for i in db_items:
            cat = i.category
            if cat not in categorized_menu:
                categorized_menu[cat] = {}
            categorized_menu[cat][i.name] = i.price
    else:
        # Fallback dictionary if no menu items added yet by user
        categorized_menu = {
            "Burgers": {"Chicken Burger": 100, "Veg Burger": 80},
            "Pizza": {"Cheese Pizza": 150},
            "Soup": {"Mutton Soup": 120, "Chicken Soup": 100},
            "Drinks": {"Pepsi": 50, "Coffee": 40}
        }

    # Fetch live tables
    tables = DiningTable.query.filter_by(restaurant_id=current_user.id).all()
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

        name = request.form.get("name")
        phone = request.form.get("phone")
        foods = request.form.getlist("food")
        try:
            discount_pct = float(request.form.get("discount", 0))
        except ValueError:
            discount_pct = 0.0

        if foods:
            food_items_receipt = []
            subtotal = 0.0

            # Create a UUID for securing the receipt link
            order_id_string = str(uuid.uuid4())

            # Build new SQL database order securely linked to current
            # Restaurant
            new_order = Order(
                restaurant_id=current_user.id,
                order_uuid=order_id_string,
                customer_name=name,
                customer_phone=phone,
                temperature=temperature,
                rainfall=rainfall
            )
            new_order.discount_pct = float(discount_pct)
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
                    for _ in range(qty):
                        order_item = OrderItem(
                            order_id=new_order.id,
                            item_name=item,
                            price=item_price
                        )
                        db.session.add(order_item)
                        
                    # --- Precise Recipe Inventory Deduction Logic ---
                    menu_item = MenuItem.query.filter_by(name=item, restaurant_id=current_user.id).first()
                    if menu_item:
                        recipes = MenuRecipe.query.filter_by(menu_item_id=menu_item.id).all()
                        for recipe in recipes:
                            inv_item = Inventory.query.get(recipe.ingredient_id)
                            if inv_item:
                                inv_item.current_stock -= (recipe.quantity_required * qty)
                    else:
                        # Fallback for dynamic POS items manually typed that don't map to DB
                        exact_inv = Inventory.query.filter_by(ingredient_name=item, restaurant_id=current_user.id).first()
                        if exact_inv: exact_inv.current_stock -= qty

            # Table State Override (Set Table to DINING)
            tbl_num_raw = request.form.get("table_number") # e.g. '4'
            if tbl_num_raw and tbl_num_raw.isdigit():
                tgt_tbl = DiningTable.query.filter_by(table_num=int(tbl_num_raw), restaurant_id=current_user.id).first()
                if tgt_tbl:
                    tgt_tbl.status = 'DINING'
                    tgt_tbl.current_order_id = new_order.id

            # Subtotal -> Apply Discount -> Apply GST (5%) -> Total
            discount_amount = round(
                float(subtotal * (discount_pct / 100.0)), 2)  # type: ignore
            after_discount = subtotal - discount_amount  # type: ignore
            gst_amount = round(float(after_discount * 0.05), 2)  # type: ignore
            final_total = after_discount + gst_amount  # type: ignore

            new_order.subtotal = subtotal
            new_order.gst_pct = 5.0
            new_order.gst_amount = gst_amount
            new_order.total_price = final_total
            db.session.commit()

            # For the immediate frontend receipt view (so customers don't query
            # db for it directly)
            PENDING_ORDERS[order_id_string] = {
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
                'Total Price': final_total
            }
            update_customer_excel(current_user.username, order_record)

            return redirect(url_for('show_receipt', order_id=order_id_string))

    # Calculate restaurant total sales from SQL DB intelligently
    restaurant_orders = Order.query.filter_by(
        restaurant_id=current_user.id).all()
    total_sales = sum((o.total_price or 0) for o in restaurant_orders)
    total_profit = round(total_sales * PROFIT_MARGIN, 2)

    return render_template(
        'index.html',
        menu_data=categorized_menu,
        predicted_tomorrow=predicted_tomorrow,
        predicted_today=predicted_today,
        temperature=temperature,
        rainfall=rainfall,
        total_price=total_price,
        receipt=receipt,
        total_sales=total_sales,
        total_profit=total_profit,
        tables=tables
    )


@app.route("/receipt/<order_id>")
@login_required
def show_receipt(order_id):
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403

    receipt = PENDING_ORDERS.get(order_id)
    if not receipt:
        return "Receipt not found or expired.", 404

    return render_template('receipt.html', receipt=receipt)


@app.route('/download_excel')
@login_required
def download_excel():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
        
@app.route('/<category>/<page>')
@login_required
def render_module(category, page):
    # Only trap module routing if it maps to our templates
    if category not in ['operations', 'finance', 'menu', 'settings', 'reports']:
        return "Route not found.", 404
        
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    
    context = {}
    if category in ['operations', 'reports'] or page in ['orders', 'kots']:
        context['orders'] = Order.query.filter_by(restaurant_id=current_user.id).order_by(Order.date.desc()).all()
        
    if category == 'menu' and page == 'menu':
        context['menu_items'] = MenuItem.query.filter_by(restaurant_id=current_user.id).all()
        context['total_items'] = len(context['menu_items'])
        context['total_categories'] = len(set(i.category for i in context['menu_items']))
        context['hidden_items'] = sum(1 for i in context['menu_items'] if i.is_hidden)
        context['best_sellers'] = sum(1 for i in context['menu_items'] if i.is_best_seller)
    elif category == 'menu' and page == 'inventory':
        context['inventory_items'] = Inventory.query.filter_by(restaurant_id=current_user.id).all()
        print(f"DEBUG IN INVENTORY ROUTE: restaurant_id={current_user.id}, items_found={len(context.get('inventory_items', []))}")
        context['total_items'] = len(context['inventory_items'])
        context['low_stock'] = sum(1 for i in context['inventory_items'] if i.current_stock <= i.min_level and i.current_stock > 0)
        context['critical_stock'] = sum(1 for i in context['inventory_items'] if i.current_stock <= 0)
        context['well_stocked'] = context['total_items'] - context['low_stock'] - context['critical_stock']
    elif category == 'menu' and page == 'recipes':
        context['menu_items'] = MenuItem.query.filter_by(restaurant_id=current_user.id).all()
        context['inventory_items'] = Inventory.query.filter_by(restaurant_id=current_user.id).all()
        
    try:
        return render_template(f'modules/{category}/{page}.html', **context)
    except Exception as e:
        return f"Module in development or missing: {e}", 404

@app.route('/api/menu/save', methods=['POST'])
@login_required
def api_menu_save():
    import werkzeug.utils
    import uuid
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    data = request.form
    item_id = data.get('id')
    try:
        if item_id:
            item = MenuItem.query.filter_by(id=item_id, restaurant_id=current_user.id).first()
            if not item: return jsonify({'success': False}), 404
        else:
            item = MenuItem(restaurant_id=current_user.id)
            db.session.add(item)
            
        item.name = data.get('name', item.name)
        item.category = data.get('category', item.category) or 'General'
        try:
            item.price = float(data.get('price', item.price) or 0)
        except ValueError:
            item.price = item.price or 0.0
            
        item.type = data.get('type', item.type) or 'Veg'
        item.status = data.get('status', item.status) or 'Active'
        item.description = data.get('description', item.description)
        item.is_best_seller = data.get('is_best_seller') == 'true'
        item.is_hidden = data.get('is_hidden') == 'true'
        
        try:
            item.prep_time = int(data.get('prep_time', item.prep_time or 15))
        except ValueError:
            item.prep_time = 15
            
        try:
            item.display_order = int(data.get('display_order', item.display_order or 0))
        except ValueError:
            item.display_order = 0

        # Handle Image Upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                ext = os.path.splitext(file.filename)[1]
                filename = f"{uuid.uuid4()}{ext}"
                upload_folder = os.path.join(app.root_path, 'static', 'uploads', 'menu')
                os.makedirs(upload_folder, exist_ok=True)
                file.save(os.path.join(upload_folder, filename))
                item.image_url = f"/static/uploads/menu/{filename}"

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/menu/delete/<int:item_id>', methods=['POST'])
@login_required
def api_menu_delete(item_id):
    item = MenuItem.query.filter_by(id=item_id, restaurant_id=current_user.id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
    return jsonify({'success': True})

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
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    data = request.json
    menu_item_id = data.get('menu_item_id')
    ingredient_id = data.get('ingredient_id')
    qty = float(data.get('quantity_required', 0))
    
    recipe = MenuRecipe.query.filter_by(menu_item_id=menu_item_id, ingredient_id=ingredient_id).first()
    if recipe:
        recipe.quantity_required = qty
    else:
        recipe = MenuRecipe(menu_item_id=menu_item_id, ingredient_id=ingredient_id, quantity_required=qty)
        db.session.add(recipe)
    
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/recipe/delete', methods=['POST'])
@login_required
def api_recipe_delete():
    data = request.json
    recipe = MenuRecipe.query.get(data.get('id'))
    if recipe:
        db.session.delete(recipe)
        db.session.commit()
    return jsonify({'success': True})

@app.route('/api/table/clear/<int:table_num>', methods=['POST'])
@login_required
def api_table_clear(table_num):
    if current_user.status == 'Pending': return jsonify({'success': False}), 403
    t = DiningTable.query.filter_by(table_num=table_num, restaurant_id=current_user.id).first()
    if t:
        t.status = 'EMPTY'
        t.current_order_id = None
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'success': False}), 404
    filename = f"{current_user.username}_database.xlsx"
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    return "No orders yet to generate a report.", 404


# ─── OPERATIONS MODULE ROUTES ───────────────────────────────────────────────

@app.route('/operations/orders')
@login_required
def op_orders():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    orders = Order.query.filter_by(restaurant_id=current_user.id)\
                        .order_by(Order.date.desc()).limit(50).all()
    return render_template('modules/operations/orders.html', orders=orders)


@app.route('/operations/online-orders')
@login_required
def op_online_orders():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/online_orders.html')


@app.route('/operations/kots')
@login_required
def op_kots():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/kots.html')


@app.route('/operations/due-payment')
@login_required
def op_due_payment():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/due_payment.html')


@app.route('/operations/billing-screen')
@login_required
def op_billing_screen():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/billing_screen.html')


@app.route('/operations/live-view')
@login_required
def op_live_view():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/live_view.html')


@app.route('/operations/bill-kot-print')
@login_required
def op_bill_kot_print():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/bill_kot_print.html')


@app.route('/operations/table')
@login_required
def op_table():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/table.html')


@app.route('/operations/custom-order-status')
@login_required
def op_custom_order_status():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/custom_order_status.html')


@app.route('/operations/delivery-boys')
@login_required
def op_delivery_boys():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/operations/delivery_boys.html')


# ─── FINANCE MODULE ROUTES ───────────────────────────────────────────────────

@app.route('/finance/cash-flow')
@login_required
def fin_cash_flow():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/finance/cash_flow.html')


@app.route('/finance/expense')
@login_required
def fin_expense():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/finance/expense.html')


@app.route('/finance/withdrawal')
@login_required
def fin_withdrawal():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/finance/withdrawal.html')


@app.route('/finance/cash-topup')
@login_required
def fin_cash_topup():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/finance/cash_topup.html')


@app.route('/finance/currency-conversion')
@login_required
def fin_currency_conversion():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/finance/currency_conversion.html')


# ─── MENU & INVENTORY MODULE ROUTES ─────────────────────────────────────────

@app.route('/menu/menu')
@login_required
def menu_menu():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/menu/menu.html')


@app.route('/menu/menu-item-onoff')
@login_required
def menu_item_onoff():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/menu/menu_item_onoff.html')


@app.route('/menu/tax')
@login_required
def menu_tax():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/menu/tax.html')


@app.route('/menu/discount')
@login_required
def menu_discount():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/menu/discount.html')


@app.route('/menu/customers')
@login_required
def menu_customers():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/menu/customers.html')


@app.route('/menu/feedback')
@login_required
def menu_feedback():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/menu/feedback.html')




@app.route('/menu/dual-screen')
@login_required
def menu_dual_screen():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/menu/dual_screen.html')


@app.route('/menu/table-view')
@login_required
def menu_table_view():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/menu/table_view.html')


# ─── SYSTEM SETTINGS MODULE ROUTES ──────────────────────────────────────────

@app.route('/settings/billing-user-profile')
@login_required
def set_billing_user_profile():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/settings/billing_user_profile.html')


@app.route('/settings/manual-sync')
@login_required
def set_manual_sync():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/settings/manual_sync.html')


@app.route('/settings/alerts')
@login_required
def set_alerts():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/settings/alerts.html')


@app.route('/settings/service-renewal')
@login_required
def set_service_renewal():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/settings/service_renewal.html')


@app.route('/settings/help')
@login_required
def set_help():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/settings/help.html')


@app.route('/settings/settings')
@login_required
def set_settings():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/settings/settings.html')


@app.route('/settings/language-profiles')
@login_required
def set_language_profiles():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/settings/language_profiles.html')


# ─── REPORTS / ANALYTICS MODULE ROUTES ──────────────────────────────────────

@app.route('/reports/category-summary')
@login_required
def rep_category_summary():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/reports/category_summary.html')


@app.route('/reports/item-summary')
@login_required
def rep_item_summary():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/reports/item_summary.html')


@app.route('/reports/sales-summary')
@login_required
def rep_sales_summary():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    # Pass real sales data
    restaurant_orders = Order.query.filter_by(
        restaurant_id=current_user.id).all()
    total_sales = sum((o.total_price or 0) for o in restaurant_orders)
    total_orders = len(restaurant_orders)
    return render_template('modules/reports/sales_summary.html',
                           total_sales=total_sales,
                           total_orders=total_orders)


@app.route('/reports/order-summary')
@login_required
def rep_order_summary():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    orders = Order.query.filter_by(restaurant_id=current_user.id)\
                        .order_by(Order.date.desc()).limit(100).all()
    return render_template('modules/reports/order_summary.html', orders=orders)


@app.route('/reports/executive-sales-summary')
@login_required
def rep_executive_sales_summary():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    restaurant_orders = Order.query.filter_by(
        restaurant_id=current_user.id).all()
    total_sales = sum((o.total_price or 0) for o in restaurant_orders)
    total_profit = round(total_sales * PROFIT_MARGIN, 2)
    return render_template('modules/reports/executive_sales_summary.html',
                           total_sales=total_sales,
                           total_profit=total_profit,
                           total_orders=len(restaurant_orders))


@app.route('/reports/tip-summary')
@login_required
def rep_tip_summary():
    if current_user.status == 'Pending':
        return "Access Denied by Firewall", 403
    return render_template('modules/reports/tip_summary.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

