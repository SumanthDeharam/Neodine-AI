import os
import joblib
import random
import threading
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# Fallback-safe ML Imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    pass

XGBRegressor = None
LGBMRegressor = None
CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor
except ImportError:
    pass

try:
    from catboost import CatBoostRegressor
except ImportError:
    pass

# Baseline Models for Benchmark Comparison
class NaiveYesterdayPredictor:
    """Baseline Model 1: Predicts demand equals yesterday's sales (PrevDaySales)."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        if hasattr(X, "columns") and "PrevDaySales" in X.columns:
            return np.array(X["PrevDaySales"].fillna(0))
        return np.zeros(len(X))

class NaiveMovingAvg7Predictor:
    """Baseline Model 2: Predicts demand equals 7-day rolling average (RollingAvgSales)."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        if hasattr(X, "columns") and "RollingAvgSales" in X.columns:
            return np.array(X["RollingAvgSales"].fillna(0))
        return np.zeros(len(X))

class LastRecordedDayPredictor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["LastRecordedSales"].to_numpy()


# Helper: Dynamically resolve db and models to prevent duplicate SQLAlchemy instance issues in circular imports
def get_db_and_models():
    import sys
    try:
        from app import db, Inventory, MenuItem, Order, OrderItem, PredictionHistory
        return db, Inventory, MenuItem, Order, OrderItem, PredictionHistory
    except Exception:
        if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'db'):
            m = sys.modules['__main__']
            return (getattr(m, 'db', None), getattr(m, 'Inventory', None), getattr(m, 'MenuItem', None),
                    getattr(m, 'Order', None), getattr(m, 'OrderItem', None), getattr(m, 'PredictionHistory', None))
        return None, None, None, None, None, None

# Helper: Load recipes from the database table MenuRecipe for active menu items of the restaurant
def get_db_recipes_for_user(restaurant_id):
    db, Inventory, MenuItem, Order, OrderItem, PredictionHistory = get_db_and_models()
    from app import MenuRecipe
    
    recipes = {}
    active_menu = MenuItem.query.filter_by(restaurant_id=restaurant_id, status='Active').all()
    for item in active_menu:
        db_recipes = MenuRecipe.query.filter_by(menu_item_id=item.id).all()
        if db_recipes:
            recipes[item.name] = []
            recipes[item.name.strip()] = recipes[item.name]
            for rcp in db_recipes:
                if rcp.ingredient:
                    recipes[item.name].append({
                        'ingredient': rcp.ingredient.ingredient_name,
                        'quantity': rcp.quantity_required,
                        'unit': rcp.ingredient.unit
                    })
    return recipes

def calculate_menu_recipe_hash(restaurant_id):
    from hashlib import md5
    db, Inventory, MenuItem, Order, OrderItem, PredictionHistory = get_db_and_models()
    from app import MenuRecipe
    
    active_menu = MenuItem.query.filter_by(restaurant_id=restaurant_id, status='Active').order_by(MenuItem.id).all()
    
    hash_str = ""
    for item in active_menu:
        hash_str += f"item:{item.id}:{item.name}:{item.price};"
        
        recipes_list = MenuRecipe.query.filter_by(menu_item_id=item.id).order_by(MenuRecipe.id).all()
        for rcp in recipes_list:
            hash_str += f"recipe:{rcp.ingredient_id}:{rcp.quantity_required};"
            
    orders_count = Order.query.filter_by(restaurant_id=restaurant_id).count()
    hash_str += f"orders:{orders_count};"
    
    return md5(hash_str.encode('utf-8')).hexdigest()

# Feature Engineering
def apply_feature_engineering(df):
    df = df.sort_values(by=["Item", "Date"]).reset_index(drop=True)
    
    for _, days in df.groupby("Item", sort=False):
        sales = days.set_index("Date")["DailySales"]
        df.loc[days.index, "LastRecordedSales"] = sales.shift().fillna(0).to_numpy()
        previous_mean = sales.expanding().mean().shift().fillna(0)
        lags = [sales.reindex(sales.index - pd.Timedelta(days=n)).set_axis(sales.index)
                .fillna(previous_mean) for n in range(1, 8)]
        df.loc[days.index, "PrevDaySales"] = lags[0].to_numpy()
        df.loc[days.index, "PrevWeekSales"] = lags[6].to_numpy()
        df.loc[days.index, "RollingAvgSales"] = pd.concat(lags, axis=1).mean(axis=1).to_numpy()
        df.loc[days.index, "MovingAverage"] = pd.concat(lags[:3], axis=1).mean(axis=1).to_numpy()

    # Label encode Category
    df["CategoryCode"] = df["Category"].astype("category").cat.codes
    # Label encode Item
    df["ItemCode"] = df["Item"].astype("category").cat.codes
    
    df = df.fillna(0)
    return df

# Training pipeline
def train_and_select_model(restaurant_id):
    print(f"[AI-PREDICT] Starting Database-Only ML Model Training for Restaurant {restaurant_id}")
    db, Inventory, MenuItem, Order, OrderItem, PredictionHistory = get_db_and_models()
    
    active_menu = MenuItem.query.filter_by(restaurant_id=restaurant_id, status='Active').all()
    active_item_names = {m.name for m in active_menu}
    
    if not active_menu:
        print("[AI-PREDICT] No active menu items found for training.")
        return None
        
    orders = Order.query.filter_by(restaurant_id=restaurant_id).all()
    if not orders:
        print("[AI-PREDICT] No orders found for training.")
        return None
        
    order_dates = [o.date.date() for o in orders]
    date_range = sorted(set(order_dates))

    # Map order details by date with normalized item name matching
    sales_by_date_item = {}
    active_norm_map = {m.name.strip().lower(): m.name for m in active_menu}
    
    for o in orders:
        dt_val = o.date.date()
        for itm in o.items:
            clean_name = itm.item_name.strip().lower()
            if clean_name in active_norm_map:
                matched_name = active_norm_map[clean_name]
                key = (dt_val, matched_name)
                sales_by_date_item[key] = sales_by_date_item.get(key, 0.0) + itm.quantity
                
    records = []
    for dt in date_range:
        day_of_week = dt.weekday()
        month = dt.month
        is_weekend = 1 if day_of_week in (5, 6) else 0
        
        for item_name in active_item_names:
            sales = sales_by_date_item.get((dt, item_name), 0.0)
            m_item = next(m for m in active_menu if m.name == item_name)
            cuisine = m_item.category or "default"
            
            records.append({
                "Date": pd.to_datetime(dt),
                "Item": item_name.strip(),
                "Category": cuisine.strip(),
                "DailySales": sales,
                "DayOfWeek": day_of_week,
                "Month": month,
                "Weekend": is_weekend,
            })
            
    df = pd.DataFrame(records)
    featured_df = apply_feature_engineering(df)
    
    features = ["ItemCode", "CategoryCode", "DayOfWeek", "Month", "Weekend",
                "PrevDaySales", "PrevWeekSales", "RollingAvgSales", "MovingAverage", "LastRecordedSales"]

    # Sort featured_df chronologically by Date for TimeSeriesSplit walk-forward validation
    featured_df = featured_df.sort_values(by="Date").reset_index(drop=True)
    X_all = featured_df[features].fillna(0).replace([np.inf, -np.inf], 0)
    y_all = featured_df["DailySales"].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Define candidate model architectures (Baselines + Advanced ML Regressors)
    models = {}
    models["Naive_LastRecorded"] = LastRecordedDayPredictor()
    models["Naive_Yesterday"] = NaiveYesterdayPredictor()
    models["Naive_MovingAvg7"] = NaiveMovingAvg7Predictor()
    models["RandomForest"] = RandomForestRegressor(n_estimators=50, random_state=42)
    models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=50, random_state=42)
    
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(n_estimators=50, max_depth=5, random_state=42, verbosity=0)
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(n_estimators=50, max_depth=5, random_state=42, verbose=-1)
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(iterations=50, depth=5, random_state=42, verbose=0)
        
    baseline_maes = {}
    ml_maes = {}
    model_metrics = {}
    trained_models = {}
    model_oof_preds = {}
    
    # Use Date-Grouped TimeSeriesSplit (Walk-Forward Validation) to guarantee future data never leaks into past evaluations
    unique_dates = np.array(sorted(featured_df["Date"].unique()))
    num_unique_dates = len(unique_dates)
    
    # Filter out zero-padded leading dates from out-of-fold validation evaluation
    first_order_date = min(o.date.date() for o in orders)
    actual_order_dates = {o.date.date() for o in orders}
    num_actual_dates = len(actual_order_dates)
    
    if len(orders) < 5 or num_actual_dates < 3 or not sales_by_date_item:
        print(f"[AI-PREDICT] Insufficient sales orders ({len(orders)}/5) for model training.")
        return {
            "success": True,
            "insufficient_data": True,
            "message": "StockSense needs five orders over at least three recorded days and sales matching the active menu.",
            "confidence_level": "Low",
            "confidence_value": "Not calibrated",
            "table": [],
            "alerts": [{"type": "warning", "message": "StockSense needs five orders over at least three recorded days and sales matching the active menu."}],
            "line": {"labels": [], "datasets": []},
            "predicted_consumption": {},
            "ingredient_metrics": {},
            "setup": {
                "has_menu": len(active_menu) > 0,
                "menu_count": len(active_menu),
                "has_recipes": True,
                "items_with_recipes": len(active_menu),
                "total_menu_items": len(active_menu),
                "has_inventory": True,
                "inventory_count": Inventory.query.filter_by(restaurant_id=restaurant_id).count(),
                "has_orders": len(orders) >= 5,
                "order_count": len(orders),
                "orders_needed": max(0, 5 - len(orders))
            }
        }

    for name, model in models.items():
        if num_actual_dates < 14 and not name.startswith("Naive_"):
            continue
        try:
            oof_preds = np.zeros(len(X_all))
            val_indices_visited = set()
            
            n_splits = max(2, min(5, num_unique_dates - 1))
            from sklearn.model_selection import TimeSeriesSplit
            tss = TimeSeriesSplit(n_splits=n_splits)
            
            for train_date_idx, val_date_idx in tss.split(unique_dates):
                train_dates = unique_dates[train_date_idx]
                val_dates = unique_dates[val_date_idx]
                
                train_mask = featured_df["Date"].isin(train_dates)
                val_mask = featured_df["Date"].isin(val_dates)
                val_indices = featured_df.index[val_mask].tolist()
                
                X_tr = featured_df.loc[train_mask, features].fillna(0)
                y_tr = featured_df.loc[train_mask, "DailySales"].fillna(0)
                X_va = featured_df.loc[val_mask, features].fillna(0)
                y_va = featured_df.loc[val_mask, "DailySales"].fillna(0)
                
                if name.startswith("Naive_"):
                    preds = model.predict(X_va)
                else:
                    fold_model = type(model)(**model.get_params())
                    fold_model.fit(X_tr, y_tr)
                    preds = fold_model.predict(X_va)
                    
                oof_preds[val_indices] = preds
                val_indices_visited.update(val_indices)

            val_idx_list = sorted(list(val_indices_visited))
            
            # EXCLUDE zero-padded dates before first_order_date from evaluation metrics
            eval_df = featured_df.iloc[val_idx_list].copy()
            eval_df["OOFPred"] = oof_preds[val_idx_list]
            real_eval_mask = eval_df["Date"] >= pd.to_datetime(first_order_date)
            
            if real_eval_mask.sum() > 0:
                y_eval = eval_df.loc[real_eval_mask, "DailySales"].values
                preds_eval = eval_df.loc[real_eval_mask, "OOFPred"].values
            else:
                y_eval = eval_df["DailySales"].values
                preds_eval = eval_df["OOFPred"].values
            
            mae = mean_absolute_error(y_eval, preds_eval)
            rmse = np.sqrt(mean_squared_error(y_eval, preds_eval))
            mape = np.mean(np.abs((y_eval - preds_eval) / np.where(y_eval == 0, 1, y_eval))) * 100
            
            if np.var(y_eval) > 0 and len(y_eval) > 5:
                r2 = r2_score(y_eval, preds_eval)
            else:
                r2 = 0.0
                
            print(f"[AI-PREDICT] Model {name} (Real-Dates TimeSeriesSplit) - MAE: {round(mae,3)}, RMSE: {round(rmse,3)}, MAPE: {round(mape,2)}%, R2: {round(r2,3)}")
            
            # Fit final model instance on full dataset for production inference
            model.fit(X_all, y_all)
            trained_models[name] = model
            model_metrics[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
            model_oof_preds[name] = (val_idx_list, oof_preds)
            
            if name.startswith("Naive_"):
                baseline_maes[name] = mae
            else:
                ml_maes[name] = mae
        except Exception as e:
            print(f"[AI-PREDICT] Error training {name}: {e}")
            
    # Strict Model Selection Rule:
    # Select ML ONLY if its TimeSeriesSplit MAE improves MAE by at least 5% (0.95 multiplier) over the best baseline model.
    best_baseline_name = min(baseline_maes, key=baseline_maes.get) if baseline_maes else "Naive_MovingAvg7"
    best_baseline_mae = baseline_maes.get(best_baseline_name, float("inf"))
    
    best_ml_name = min(ml_maes, key=ml_maes.get) if ml_maes else None
    best_ml_mae = ml_maes.get(best_ml_name, float("inf"))
    
    if num_actual_dates >= 14 and best_ml_name and best_baseline_mae > 0 and best_ml_mae <= (best_baseline_mae * 0.95):
        best_model_name = best_ml_name
        is_baseline = False
    else:
        best_model_name = best_baseline_name
        is_baseline = True
        
    best_model = trained_models[best_model_name]
    best_metrics = model_metrics[best_model_name]
    lowest_mae = best_metrics["MAE"]
    
    # Calculate Ingredient-Level Validation Metrics (MAE & RMSE per ingredient on real dates)
    recipes = get_db_recipes_for_user(restaurant_id)
    ing_actuals = {}
    ing_preds = {}
    
    val_idx_list, selected_oof_preds = model_oof_preds[best_model_name]
    val_featured_df = featured_df.iloc[val_idx_list].copy()
    val_featured_df["PredSales"] = np.clip(selected_oof_preds[val_idx_list], 0, None)
    real_val_df = val_featured_df[val_featured_df["Date"] >= pd.to_datetime(first_order_date)]
    if real_val_df.empty:
        real_val_df = val_featured_df
    
    from collections import Counter
    for _, row in real_val_df.iterrows():
        itm_name = row["Item"]
        act_s = row["DailySales"]
        prd_s = row["PredSales"]
        
        recipe_list = recipes.get(itm_name) or recipes.get(itm_name.strip())
        if recipe_list:
            for ing_entry in recipe_list:
                ing_name = ing_entry['ingredient'].strip()
                ing_qty = ing_entry['quantity']
                
                if ing_name not in ing_actuals:
                    ing_actuals[ing_name] = Counter()
                    ing_preds[ing_name] = Counter()
                    
                ing_actuals[ing_name][row["Date"]] += act_s * ing_qty
                ing_preds[ing_name][row["Date"]] += prd_s * ing_qty
                
    ingredient_metrics = {}
    print(f"\n[AI-PREDICT] Ingredient-Level Validation (Selected Model: {best_model_name}):")
    for ing_name in ing_actuals:
        acts = np.array(list(ing_actuals[ing_name].values()))
        prds = np.array(list(ing_preds[ing_name].values()))
        ing_mae = mean_absolute_error(acts, prds)
        ing_rmse = np.sqrt(mean_squared_error(acts, prds))
        ingredient_metrics[ing_name] = {"MAE": round(ing_mae, 2), "RMSE": round(ing_rmse, 2)}
        print(f"  -> Ingredient '{ing_name}' | Validation MAE: {round(ing_mae,2)} | Validation RMSE: {round(ing_rmse,2)}")
        
    conf_level = "Low" if num_actual_dates < 14 else "Medium"
    print(f"[AI-PREDICT] Model Selection Decision: '{best_model_name}' selected (Is Baseline: {is_baseline}) | Best ML MAE: {round(best_ml_mae,3)} vs Best Baseline ({best_baseline_name}) MAE: {round(best_baseline_mae,3)}")
            
    os.makedirs("instance", exist_ok=True)
    model_path = f"instance/stocksense_model_{restaurant_id}.joblib"
    
    item_categories = featured_df.groupby("Item")["Category"].first().to_dict()
    category_codes = featured_df.groupby("Category")["CategoryCode"].first().to_dict()
    item_codes = featured_df.groupby("Item")["ItemCode"].first().to_dict()
    
    order_count = len(orders)
    menu_hash = calculate_menu_recipe_hash(restaurant_id)
    
    payload = {
        "forecast_version": 2,
        "validation_days": int(real_val_df["Date"].nunique()),
        "model": best_model,
        "model_name": best_model_name,
        "is_baseline": is_baseline,
        "metrics": best_metrics,
        "ingredient_metrics": ingredient_metrics,
        "confidence_value": "Not calibrated",
        "confidence_level": conf_level,
        "item_categories": item_categories,
        "category_codes": category_codes,
        "item_codes": item_codes,
        "feature_names": features,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_orders_trained": order_count,
        "item_means": featured_df.groupby("Item")["DailySales"].mean().to_dict(),
        "cat_means": featured_df.groupby("Category")["DailySales"].mean().to_dict(),
        "menu_recipe_hash": menu_hash
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(model_path), delete=False) as tmp:
        tmp_path = tmp.name
    try:
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, model_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print(f"[AI-PREDICT] Saved best database-driven model ({best_model_name}) to {model_path} (MAE: {round(lowest_mae, 3)})")
    return payload


# Asynchronous Background Retraining Wrapper
def trigger_background_retrain(restaurant_id):
    def run():
        try:
            from app import app
            with app.app_context():
                train_and_select_model(restaurant_id)
        except Exception as e:
            print(f"[AI-PREDICT] Background retrain error: {e}")
    threading.Thread(target=run, daemon=True).start()

def check_and_trigger_retrain(restaurant_id):
    model_path = f"instance/stocksense_model_{restaurant_id}.joblib"
    if not os.path.exists(model_path):
        print(f"[AI-PREDICT] Model doesn't exist. Triggering initial train.")
        trigger_background_retrain(restaurant_id)
        return
        
    try:
        payload = joblib.load(model_path)
        trained_at_str = payload.get("trained_at")
        num_orders_trained = payload.get("num_orders_trained", 0)
        
        from app import Order
        current_order_count = Order.query.filter_by(restaurant_id=restaurant_id).count()
        order_diff = current_order_count - num_orders_trained
        
        trained_at = datetime.fromisoformat(trained_at_str)
        time_elapsed = datetime.now(timezone.utc) - trained_at
        
        if order_diff >= 100 or time_elapsed >= timedelta(hours=24):
            print(f"[AI-PREDICT] Retrain triggered: order_diff={order_diff}, time_elapsed={time_elapsed}")
            trigger_background_retrain(restaurant_id)
    except Exception as e:
        print(f"[AI-PREDICT] Error checking retraining: {e}")
        trigger_background_retrain(restaurant_id)

# Load the saved model (cold start wrapper)
def get_model_payload(restaurant_id):
    model_path = f"instance/stocksense_model_{restaurant_id}.joblib"
    current_hash = calculate_menu_recipe_hash(restaurant_id)
    
    if not os.path.exists(model_path):
        # Run initial train synchronously if model doesn't exist
        print(f"[AI-PREDICT] Model not found. Training initial model.") 
        return train_and_select_model(restaurant_id)
    try:
        payload = joblib.load(model_path)
        saved_hash = payload.get("menu_recipe_hash")
        if payload.get("forecast_version") != 2 or saved_hash != current_hash:
            print(f"[AI-PREDICT] Recipe hash mismatch ({saved_hash} != {current_hash}). Retraining model.")
            return train_and_select_model(restaurant_id)
        return payload
    except Exception as e:
        print(f"[AI-PREDICT] Error loading model or recipe hash mismatch, retraining: {e}")
        return train_and_select_model(restaurant_id)

# Core prediction logic
def predict_consumption_tomorrow(restaurant_id):
    from app import get_weather, Inventory, PredictionHistory, db, MenuItem, Order
    
    # Check minimum order volume threshold
    orders_count = Order.query.filter_by(restaurant_id=restaurant_id).count()
    if orders_count < 5:
        raise ValueError("Insufficient data for prediction")
        
    # Get active menu items
    active_menu = MenuItem.query.filter_by(restaurant_id=restaurant_id, status='Active').all()
    active_item_names = {m.name.strip() for m in active_menu}
    
    if not active_menu:
        raise ValueError("No active menu items")
        
    payload = get_model_payload(restaurant_id)
    if not payload:
        raise ValueError("Model not trained")
        
    if isinstance(payload, dict) and payload.get("insufficient_data"):
        return payload
        
    model = payload["model"]
    item_codes = payload["item_codes"]
    category_codes = payload["category_codes"]
    item_categories = payload["item_categories"]
    features = payload["feature_names"]
    item_means = payload["item_means"]
    
    # Filter item keys to only retrieve active menu items
    items_to_predict = [item for item in item_codes.keys() if item in active_item_names]
    if not items_to_predict:
        items_to_predict = list(active_item_names)
        
    # We will accumulate ingredient usage for each day: {day_index: {ingredient: qty}}
    daily_ing_consumption = {i: {} for i in range(5)}
    line_labels = []
    
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    recipes = get_db_recipes_for_user(restaurant_id)
    
    inv_items = Inventory.query.filter_by(restaurant_id=restaurant_id).all()
    stock_map = {i.ingredient_name.strip().lower(): i for i in inv_items}
    
    # Track dynamic historical sales per item for multi-step recursive forecasting
    start = tomorrow - timedelta(days=7)
    recent = Order.query.filter(Order.restaurant_id == restaurant_id, Order.date >= start,
                                Order.date < tomorrow).all()
    recorded_days = {order.date.date() for order in recent}
    sales = {}
    for order in recent:
        for line in order.items:
            key = (order.date.date(), line.item_name.strip().casefold())
            sales[key] = sales.get(key, 0) + line.quantity
    last_order = Order.query.filter(Order.restaurant_id == restaurant_id, Order.date < tomorrow).order_by(Order.date.desc()).first()
    last_recorded = {}
    if last_order:
        last_day = last_order.date.date()
        last_orders = Order.query.filter(Order.restaurant_id == restaurant_id, Order.date >= last_day,
                                         Order.date < last_day + timedelta(days=1)).all()
        for order in last_orders:
            for line in order.items:
                name = line.item_name.strip().casefold()
                last_recorded[name] = last_recorded.get(name, 0) + line.quantity
    item_sales_history = {}
    for item in items_to_predict:
        history = []
        for offset in range(7):
            day = start + timedelta(days=offset)
            history.append(sales.get((day, item.casefold()), 0) if day in recorded_days
                           else item_means.get(item, 0))
        item_sales_history[item] = history

    # Predict for each of the 5 days recursively
    for day_idx in range(5):
        target_date = tomorrow + timedelta(days=day_idx)
        date_str = target_date.strftime('%d %b')
        line_labels.append(date_str)
        
        day_of_week = target_date.weekday()
        month = target_date.month
        is_weekend = int(day_of_week >= 5)

        test_records = []
        for item in items_to_predict:
            cat = item_categories.get(item, "default")
            item_code = item_codes.get(item, 0)
            cat_code = category_codes.get(cat, 0)
            
            history = item_sales_history[item]
            prev_day_s = history[-1]
            prev_week_s = history[-7] if len(history) >= 7 else history[0]
            rolling_avg_s = float(np.mean(history[-7:]))
            moving_avg_s = float(np.mean(history[-3:]))
            
            test_records.append({
                "Item": item,
                "ItemCode": item_code,
                "CategoryCode": cat_code,
                "DayOfWeek": day_of_week,
                "Month": month,
                "Weekend": is_weekend,
                "LastRecordedSales": last_recorded.get(item.casefold(), 0) if day_idx == 0 else prev_day_s,
                "PrevDaySales": prev_day_s,
                "PrevWeekSales": prev_week_s,
                "RollingAvgSales": rolling_avg_s,
                "MovingAverage": moving_avg_s,
            })
            
        test_df = pd.DataFrame(test_records)
        test_X = test_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        preds = model.predict(test_X)
        
        # Pure model output without artificial scaling or hardcoded multipliers
        adjusted_preds = np.clip(preds, 0, None)
        test_df["PredictedSales"] = adjusted_preds
        
        # Append today's predictions to history for next day's recursive lag calculation
        for item, pred_val in zip(items_to_predict, adjusted_preds):
            item_sales_history[item].append(float(pred_val))
            
        predicted_menu_demand = test_df.set_index("Item")["PredictedSales"].to_dict()
        
        # Convert to ingredient consumption for this day
        for item_name, predicted_qty in predicted_menu_demand.items():
            if predicted_qty <= 0:
                continue
            recipe_list = recipes.get(item_name) or recipes.get(item_name.strip())
            if recipe_list:
                for ing_entry in recipe_list:
                    ing_name = ing_entry['ingredient']
                    ing_qty = ing_entry['quantity']
                    ing_unit = ing_entry['unit']
                    
                    total_qty = ing_qty * predicted_qty
                    
                    inv_item = stock_map.get(ing_name.strip().lower())
                    if inv_item:
                        from app import convert_to_base_unit
                        converted_qty = convert_to_base_unit(total_qty, ing_unit, inv_item.unit, f"AI-Predict -> {ing_name}")
                        target_name = inv_item.ingredient_name
                        
                        if target_name not in daily_ing_consumption[day_idx]:
                            daily_ing_consumption[day_idx][target_name] = 0.0
                        daily_ing_consumption[day_idx][target_name] += converted_qty
                        
    # Now build response based on Day 0 (Tomorrow) predictions
    alerts = []
    table_data = []
    
    tomorrow_ing_usage = daily_ing_consumption[0]
    model_name = payload["model_name"]
    
    # Clear any old predictions for tomorrow to avoid duplicate history logs
    PredictionHistory.query.filter_by(
        restaurant_id=restaurant_id,
        prediction_date=datetime.combine(tomorrow, datetime.min.time())
    ).delete()
    
    for ing, qty in tomorrow_ing_usage.items():
        qty_round = round(qty, 2)
        inv_item = stock_map.get(ing.strip().lower())
        
        if inv_item:
            current_s = inv_item.current_stock
            min_l = inv_item.min_level
            target_unit = inv_item.unit
            
            if current_s <= 0:
                status = "Out of Stock"       
            elif current_s <= min_l * 0.5:
                status = "Critical"
            elif current_s <= min_l:
                status = "Low"
            else:
                status = "Safe"
        else:
            current_s = 500.0
            status = "Safe"
            target_unit = 'units'
            
        table_data.append({
            'name': ing,
            'on_hand': round(current_s, 2),
            'unit': target_unit,
            'shortfall': round(max(0, qty - current_s), 2),
            'used': f"{qty_round} {target_unit}",
            'est_remain': f"{round(max(0.0, current_s - qty_round), 2)} {target_unit}",
            'status': status
        })
        
        if status in ("Low", "Critical", "Out of Stock"):
            alerts.append({
                'msg': f"{ing} is {status.lower()} (stock: {current_s} {target_unit})",
                'severity': 'high' if status in ("Critical", "Out of Stock") else 'low'
            })
            
        # Log to PredictionHistory
        history_entry = PredictionHistory(
            restaurant_id=restaurant_id,
            prediction_date=datetime.combine(tomorrow, datetime.min.time()),    
            model_used=model_name,
            ingredient_name=ing,
            predicted_qty=qty_round
        )
        db.session.add(history_entry)
        
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"[AI-PREDICT] PredictionHistory commit error: {e}")
        
    # Build Line datasets for top 3 ingredients over 5 days
    total_usage_5days = {}
    for day_idx in range(5):
        for ing, qty in daily_ing_consumption[day_idx].items():
            total_usage_5days[ing] = total_usage_5days.get(ing, 0.0) + qty
            
    top_3_ings = [k for k, v in sorted(total_usage_5days.items(), key=lambda x: x[1], reverse=True)[:3]]
    
    line_datasets = []
    colors = ['#6366f1', '#10b981', '#f59e0b']
    
    for idx, ing in enumerate(top_3_ings):
        data_points = []
        for day_idx in range(5):
            val = daily_ing_consumption[day_idx].get(ing, 0.0)
            data_points.append(round(val, 2))
            
        line_datasets.append({
            'label': ing,
            'data': data_points,
            'borderColor': colors[idx % len(colors)],
            'tension': 0.3
        })
        
    # Multi-Factor Confidence Level & Ingredient Metrics
    confidence_value = payload.get("confidence_value", "50.0%")
    confidence_level = payload.get("confidence_level", "Low")
    ingredient_metrics = payload.get("ingredient_metrics", {})
        
    return {
        "success": True,
        "line": {
            "labels": line_labels,
            "datasets": line_datasets
        },
        "alerts": alerts,
        "table": table_data,
        "predicted_consumption": tomorrow_ing_usage,
        "confidence_value": confidence_value,
        "confidence_level": confidence_level,
        "ingredient_metrics": ingredient_metrics,
        "validation_mae": payload["metrics"]["MAE"],
        "validation_days": payload["validation_days"],
        "recent_days": len(recorded_days),
    }

# ========================================================
# LOCATION INTELLIGENCE AI PREDICTION ENGINE
# ========================================================
def predict_location_intelligence(restaurant_id, lat, lng, radius, nearby_data, competitors_data, provider_status="OK", provider_name=None):
    """
    LOCAL FOOD MARKET OPPORTUNITY ENGINE
    ======================================
    Analyzes real nearby competitors and points of interest (POIs) using the
    unified Location Intelligence module.
    """
    from location_intelligence.models import QueryResult, DataState, CompetitorPOI, EnvironmentPOI
    from location_intelligence.analyzer import LocationMarketAnalyzer

    competitors = []
    for c in (competitors_data or []):
        if isinstance(c, dict):
            competitors.append(CompetitorPOI(
                source_id=c.get('source_id'),
                osm_type=c.get('osm_type', 'node'),
                osm_id=c.get('osm_id'),
                name=c.get('name', 'Unnamed Outlet'),
                category=c.get('category', 'Dining'),
                cuisine=c.get('cuisine'),
                address=c.get('address', 'Local Area'),
                lat=float(c.get('lat', 0) or c.get('latitude', 0)),
                lng=float(c.get('lng', 0) or c.get('longitude', 0)),
                distance=float(c.get('distance', 0)),
                rating=c.get('rating'),
                reviews=int(c.get('reviews', 0) or 0),
                price_level=c.get('price_level'),
                opening_hours=c.get('opening_hours'),
                website=c.get('website'),
                phone=c.get('phone'),
                source=c.get('source', provider_name or 'OpenStreetMap Overpass API')
            ))
        elif isinstance(c, CompetitorPOI):
            competitors.append(c)

    pois = []
    for p in (nearby_data or []):
        if isinstance(p, dict):
            pois.append(EnvironmentPOI(
                type=p.get('type', 'poi'),
                name=p.get('name', 'Unknown POI'),
                vicinity=p.get('vicinity', ''),
                lat=float(p.get('lat', 0) or p.get('latitude', 0)),
                lng=float(p.get('lng', 0) or p.get('lon', 0) or p.get('longitude', 0)),
                distance=float(p.get('distance', 0)),
                tags=p.get('tags', {}),
                source=p.get('source', provider_name or 'OpenStreetMap Overpass API')
            ))
        elif isinstance(p, EnvironmentPOI):
            pois.append(p)

    if competitors:
        default_source = competitors[0].source or 'OpenStreetMap Overpass API'
    elif pois:
        default_source = pois[0].source or 'OpenStreetMap Overpass API'
    else:
        default_source = 'Unavailable'

    p_name = provider_name or default_source

    if provider_status in ("PROVIDER_ERROR", "ERROR"):
        status = DataState.PROVIDER_ERROR
    elif not competitors and not pois:
        status = DataState.ZERO_RESULTS
    else:
        status = DataState.REAL_DATA

    qr = QueryResult(
        status=status,
        competitors=competitors,
        pois=pois,
        raw_count=len(competitors) + len(pois),
        http_status=200 if status != DataState.PROVIDER_ERROR else 500,
        provider_name=p_name
    )

    return LocationMarketAnalyzer.analyze(qr, lat, lng, radius)



