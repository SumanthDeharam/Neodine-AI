def generate_purchase_recommendations(inventory_comparisons):
    """
    Generates a prioritized list of raw ingredient purchase recommendations.
    Sets High, Medium, or Low urgency priorities and traces calculation costs.
    """
    recommendations = []
    
    # Static unit price mappings for estimating costs in rupees
    UNIT_PRICES = {
        "kg": 120.0,
        "g": 0.15,
        "l": 95.0,
        "ml": 0.12,
        "pcs": 15.0,
        "units": 20.0,
        "bottles": 35.0,
        "can": 30.0
    }
    
    for ing_id, comp in inventory_comparisons.items():
        shortage = comp["shortage"]
        current_stock = comp["current_stock"]
        required_qty = comp["required_quantity"]
        total_target = comp["total_target_required"]
        unit = (comp["unit"] or "units").lower().strip()
        
        # Calculate priority
        if shortage > 0:
            if current_stock < required_qty:
                priority = "High"
                explanation = f"Critical shortage: Current stock ({current_stock:.1f}) is below core recipe needs ({required_qty:.1f})."
            else:
                priority = "Medium"
                explanation = f"Encroaching safety buffer: Stock ({current_stock:.1f}) is enough for baseline recipes but fails the 15% safety buffer target ({total_target:.1f})."
        else:
            priority = "Low"
            explanation = f"Sufficient stock: Current stock ({current_stock:.1f}) covers the total target including safety buffer ({total_target:.1f})."
            
        # Estimate Cost
        unit_price = UNIT_PRICES.get(unit, 20.0)
        estimated_cost = shortage * unit_price
        
        recommendations.append({
            "ingredient_id": comp["ingredient_id"],
            "ingredient_name": comp["ingredient_name"],
            "unit": comp["unit"],
            "current_stock": current_stock,
            "required_quantity": required_qty,
            "total_target_required": total_target,
            "suggested_purchase": round(shortage, 2),
            "estimated_cost": int(estimated_cost),
            "priority": priority,
            "explanation": explanation,
            "unit_price": unit_price
        })
        
    # Sort recommendations by suggested purchase quantity (descending) and priority (High first)
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recommendations.sort(key=lambda x: (priority_order[x["priority"]], -x["suggested_purchase"]))
    
    return recommendations
