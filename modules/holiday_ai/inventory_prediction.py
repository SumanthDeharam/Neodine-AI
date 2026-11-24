def compare_inventory_requirements(recipe_requirements):
    """
    Compares total calculated recipe requirements against current stock.
    Calculates exact shortages, including a 15% safety buffer margin.
    """
    inventory_comparisons = {}
    
    for ing_id, req in recipe_requirements.items():
        required_qty = req["required_quantity"]
        current_stock = req["current_stock"] or 0.0
        
        # 15% safety buffer to prevent stockouts
        safety_buffer = required_qty * 0.15
        total_target_required = required_qty + safety_buffer
        
        shortage = total_target_required - current_stock
        shortage = max(0.0, shortage)
        
        # Formulate trace string
        comparison_trace = (
            f"Required: {required_qty:.1f} + 15% Buffer ({safety_buffer:.1f}) = {total_target_required:.1f} total target. "
            f"Current Stock: {current_stock:.1f}. Shortage: {shortage:.1f} {req['unit']}"
        )
        
        inventory_comparisons[ing_id] = {
            "ingredient_id": req["ingredient_id"],
            "ingredient_name": req["ingredient_name"],
            "unit": req["unit"],
            "current_stock": current_stock,
            "min_level": req["min_level"],
            "required_quantity": round(required_qty, 2),
            "safety_buffer": round(safety_buffer, 2),
            "total_target_required": round(total_target_required, 2),
            "shortage": round(shortage, 2),
            "comparison_trace": comparison_trace,
            "conversion_logs": req["conversion_logs"]
        }
        
    return inventory_comparisons
