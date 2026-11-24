import os
import sys
import re
import pandas as pd

# Add app directory to sys.path
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

EXCEL_PATH = os.path.join(APP_DIR, "Restaurant_1195_Items_with_Cafe.xlsx")

# Known dish alias mappings for common restaurant menu item variations & test names
KNOWN_ALIASES = {
    'mango lassi': 'classic lassi',
    'test pizza 1': 'margherita pizza',
    'test pizza 2': 'veggie supreme pizza',
    'cheese pizza': 'margherita pizza',
    'veg burger': 'classic veg burger',
    'mutton soup': 'mutton masala',
    'pepsi': 'cold coffee',
}

def find_best_recipe_match(db_name, excel_recipes):
    norm_db = db_name.strip().lower()
    
    # 1. Exact normalized match
    if norm_db in excel_recipes:
        return norm_db
        
    # 2. Known alias match
    if norm_db in KNOWN_ALIASES and KNOWN_ALIASES[norm_db] in excel_recipes:
        return KNOWN_ALIASES[norm_db]
        
    # 3. Cleaned name (remove numbers, "test", "special", "loaded", extra spaces)
    cleaned = re.sub(r'\b(test|special|loaded|item|\d+)\b', '', norm_db, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if cleaned and cleaned in excel_recipes:
        return cleaned
    if cleaned in KNOWN_ALIASES and KNOWN_ALIASES[cleaned] in excel_recipes:
        return KNOWN_ALIASES[cleaned]
        
    # 4. Keyword / Word boundary match if unique
    words = [w for w in norm_db.split() if len(w) > 2 and w not in ('test', 'item', 'special')]
    if words:
        matches = []
        for excel_key in excel_recipes.keys():
            if all(w in excel_key for w in words):
                matches.append(excel_key)
        if len(matches) == 1:
            return matches[0]
            
    return None

def import_recipes_for_user(restaurant_id=None):
    from app import app, db, User, MenuItem, MenuRecipe, Inventory
    
    if not os.path.exists(EXCEL_PATH):
        print(f"[ERROR] Excel file not found at: {EXCEL_PATH}")
        return False
        
    print(f"[IMPORT] Reading Excel file: {EXCEL_PATH}")
    df_ing = pd.read_excel(EXCEL_PATH, sheet_name='Ingredients')
    
    # Required columns check
    req_cols = {'dish_name', 'ingredient_name', 'quantity', 'unit'}
    if not req_cols.issubset(df_ing.columns):
        print(f"[ERROR] Excel sheet 'Ingredients' missing required columns: {req_cols - set(df_ing.columns)}")
        return False
        
    # Group Excel ingredients by exact normalized dish name
    excel_recipes = {}
    for _, row in df_ing.iterrows():
        d_name = str(row['dish_name']).strip()
        ing_name = str(row['ingredient_name']).strip()
        qty = float(row['quantity']) if pd.notnull(row['quantity']) else 0.0
        unit = str(row['unit']).strip() if pd.notnull(row['unit']) else 'g'
        
        norm_d = d_name.lower()
        if norm_d not in excel_recipes:
            excel_recipes[norm_d] = {
                'original_name': d_name,
                'cuisine': str(row.get('cuisine_category', '')).strip(),
                'ingredients': []
            }
        excel_recipes[norm_d]['ingredients'].append({
            'ingredient_name': ing_name,
            'quantity': qty,
            'unit': unit
        })
        
    print(f"[IMPORT] Parsed {len(excel_recipes)} unique recipes from Excel.")
    
    with app.app_context():
        if restaurant_id is not None:
            users = User.query.filter_by(id=restaurant_id).all()
        else:
            users = User.query.filter(User.status == 'Approved').all()
            
        for user in users:
            print(f"\n==================================================")
            print(f"PROCESSING RESTAURANT: {user.username} (ID: {user.id})")
            print(f"==================================================")
            
            menu_items = MenuItem.query.filter_by(restaurant_id=user.id).all()
            if not menu_items:
                print(f"[INFO] User {user.username} has 0 menu items in DB.")
                continue
                
            matched_items = []
            unmatched_items = []
            recipes_created = 0
            inventory_created = 0
            
            for m_item in menu_items:
                matched_key = find_best_recipe_match(m_item.name, excel_recipes)
                
                if matched_key:
                    matched_items.append((m_item.name, excel_recipes[matched_key]['original_name']))
                    recipe_info = excel_recipes[matched_key]
                    
                    for ing in recipe_info['ingredients']:
                        ing_name = ing['ingredient_name']
                        ing_unit = ing['unit']
                        ing_qty = ing['quantity']
                        
                        # Find or create Inventory item with stock = 0.0 (never default fake stock)
                        inv_item = Inventory.query.filter(
                            Inventory.restaurant_id == user.id,
                            db.func.lower(Inventory.ingredient_name) == ing_name.lower()
                        ).first()
                        
                        if not inv_item:
                            inv_item = Inventory(
                                restaurant_id=user.id,
                                ingredient_name=ing_name,
                                unit=ing_unit,
                                current_stock=0.0, # Rule: Stock default is 0.0 (real stock only)
                                min_level=5.0
                            )
                            db.session.add(inv_item)
                            db.session.commit()
                            inventory_created += 1
                            print(f"  [INVENTORY CREATED] '{ing_name}' ({ing_unit}) with stock 0.0")
                            
                        # Create MenuRecipe record if not existing
                        existing_rcp = MenuRecipe.query.filter_by(
                            menu_item_id=m_item.id,
                            ingredient_id=inv_item.id
                        ).first()
                        
                        if not existing_rcp:
                            new_rcp = MenuRecipe(
                                menu_item_id=m_item.id,
                                ingredient_id=inv_item.id,
                                quantity_required=ing_qty
                            )
                            db.session.add(new_rcp)
                            recipes_created += 1
                            print(f"  [RECIPE LINKED] MenuItem '{m_item.name}' -> '{ing_name}': {ing_qty} {ing_unit}")
                        else:
                            # Update quantity if changed
                            existing_rcp.quantity_required = ing_qty
                            
                    db.session.commit()
                else:
                    unmatched_items.append(m_item.name)
                        
            print(f"\n--- Summary for User {user.username} (ID: {user.id}) ---")
            print(f"Total DB Menu Items: {len(menu_items)}")
            print(f"Matched Items: {len(matched_items)}")
            for db_n, ex_n in matched_items:
                print(f"  - MATCHED: '{db_n}' ==> Excel '{ex_n}'")
            print(f"Unmatched Items: {len(unmatched_items)}")
            for db_n in unmatched_items:
                print(f"  - UNMATCHED: '{db_n}' (No matching recipe found in Excel)")
            print(f"New Inventory Ingredients Created (stock=0.0): {inventory_created}")
            print(f"New MenuRecipe Records Linked: {recipes_created}")
            
    return True

if __name__ == '__main__':
    restaurant_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    import_recipes_for_user(restaurant_arg)
