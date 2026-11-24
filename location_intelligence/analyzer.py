import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from location_intelligence.models import (
    QueryResult, DataState, CompetitorPOI, EnvironmentPOI
)

logger = logging.getLogger("location_intelligence")

class LocationMarketAnalyzer:
    """
    Location Intelligence Analysis Engine.
    Computes competitor metrics, category distributions, environment classification,
    opportunity scores (0-100), market gaps, and actionable insights strictly from real input data.
    """

    FOOD_CATEGORIES = [
        'Biryani & Indian',
        'South Indian',
        'Fast Food',
        'Burgers',
        'Pizza',
        'Cafe & Beverages',
        'Chinese & Asian',
        'Bakery & Desserts',
        'Healthy & Salads',
        'Casual / Fine Dining',
        'Other'
    ]

    CATEGORY_POI_PROFILES = {
        'Fast Food': {'pois': ['college', 'transit', 'office', 'mall'], 'env_pref': 'Student/Transit'},
        'Cafe & Beverages': {'pois': ['college', 'office', 'mall'], 'env_pref': 'Student/Corporate'},
        'Biryani & Indian': {'pois': ['office', 'residential', 'transit', 'hospital'], 'env_pref': 'Corporate/Residential'},
        'South Indian': {'pois': ['transit', 'office', 'residential'], 'env_pref': 'Transit/Corporate'},
        'Pizza': {'pois': ['college', 'residential', 'mall'], 'env_pref': 'Student/Residential'},
        'Burgers': {'pois': ['college', 'mall', 'transit'], 'env_pref': 'Student/Shopping'},
        'Chinese & Asian': {'pois': ['college', 'office', 'residential'], 'env_pref': 'Student/Corporate'},
        'Bakery & Desserts': {'pois': ['mall', 'college', 'office'], 'env_pref': 'Shopping/Student'},
        'Healthy & Salads': {'pois': ['office', 'college', 'mall'], 'env_pref': 'Corporate/Student'},
        'Casual / Fine Dining': {'pois': ['mall', 'hotel', 'office'], 'env_pref': 'Shopping/Hospitality'}
    }

    @classmethod
    def analyze(cls, query_result: QueryResult, lat: float, lng: float, radius: float) -> Dict[str, Any]:
        """
        Processes QueryResult and generates complete dashboard dataset adhering strictly to UI contract.
        """
        radius = float(radius or 1000)
        radius_km = round(radius / 1000.0, 1)

        # Handle non-REAL_DATA states
        if query_result.status != DataState.REAL_DATA:
            return cls._build_non_real_data_payload(query_result, lat, lng, radius)

        competitors = query_result.competitors
        pois = query_result.pois
        comp_count = len(competitors)
        poi_count = len(pois)

        # 1. Competitor Distance Metrics
        comp_1km = sum(1 for c in competitors if c.distance <= 1000)
        comp_2km = sum(1 for c in competitors if c.distance <= 2000)
        distances = [c.distance for c in competitors]
        avg_dist = round(sum(distances) / len(distances), 1) if distances else 0.0
        min_dist = round(min(distances), 1) if distances else 0.0
        max_dist = round(max(distances), 1) if distances else 0.0

        # Ratings & Price: OSM does NOT provide standard rating or price data. MUST NOT FABRICATE.
        avg_rating_str = "N/A"
        total_reviews = 0

        # 2. Category Distribution
        category_map = {cat: 0 for cat in cls.FOOD_CATEGORIES}
        cuisine_counts = {}

        for c in competitors:
            raw_cat = (c.category or '').lower()
            c_name = (c.name or '').lower()
            c_cuisine = (c.cuisine or '').lower()

            if c.cuisine:
                cuisine_counts[c.cuisine] = cuisine_counts.get(c.cuisine, 0) + 1

            if any(k in raw_cat or k in c_name or k in c_cuisine for k in ['biryani', 'indian', 'thali', 'curry', 'dhaba']):
                category_map['Biryani & Indian'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['south indian', 'dosa', 'idli', 'tiffin', 'sambar']):
                category_map['South Indian'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['burger', 'mac', 'king']):
                category_map['Burgers'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['pizza', 'pizzeria', 'domino']):
                category_map['Pizza'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['cafe', 'coffee', 'tea', 'beverage', 'juice', 'shake', 'boba']):
                category_map['Cafe & Beverages'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['chinese', 'asian', 'noodle', 'momos', 'ramen', 'thai']):
                category_map['Chinese & Asian'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['bakery', 'cake', 'pastry', 'dessert', 'ice cream', 'waffle', 'sweet']):
                category_map['Bakery & Desserts'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['salad', 'healthy', 'bowl', 'organic', 'vegan', 'wrap']):
                category_map['Healthy & Salads'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['fast food', 'snack', 'kebab', 'street food', 'roll', 'kfc', 'subway']):
                category_map['Fast Food'] += 1
            elif any(k in raw_cat or k in c_name or k in c_cuisine for k in ['restaurant', 'dining', 'bistro', 'bar', 'pub', 'grill']):
                category_map['Casual / Fine Dining'] += 1
            else:
                category_map['Other'] += 1

        active_category_counts = {k: v for k, v in category_map.items() if v > 0}

        import time
        t_an_start = time.perf_counter()

        # 3. Environment POI Signal Analysis (Distance-Weighted Proximity Model)
        raw_poi_counts = {
            'campus': 0,
            'school': 0,
            'office': 0,
            'commercial': 0,
            'residential': 0,
            'transit': 0,
            'medical': 0,
            'hospitality': 0,
            'industrial': 0
        }

        poi_weighted_scores = {
            'campus': 0.0,
            'office': 0.0,
            'commercial': 0.0,
            'residential': 0.0,
            'transit': 0.0,
            'medical': 0.0,
            'hospitality': 0.0,
            'industrial': 0.0
        }

        t_env_0 = time.perf_counter()

        def get_dist_weight(dist_m: float) -> float:
            if dist_m <= 300.0:
                return 1.0
            elif dist_m <= 800.0:
                return 0.7
            elif dist_m <= 1500.0:
                return 0.4
            else:
                return 0.2

        for p in pois:
            t_type = (p.type or '').lower()
            tags = p.tags or {}
            amenity = (tags.get('amenity') or '').lower()
            shop = (tags.get('shop') or '').lower()
            office = (tags.get('office') or '').lower()
            tourism = (tags.get('tourism') or '').lower()
            building = (tags.get('building') or '').lower()
            landuse = (tags.get('landuse') or '').lower()
            highway = (tags.get('highway') or '').lower()
            railway = (tags.get('railway') or '').lower()

            w = get_dist_weight(p.distance)

            if amenity in ('college', 'university') or building in ('university', 'college') or t_type in ('college', 'university'):
                raw_poi_counts['campus'] += 1
                poi_weighted_scores['campus'] += 3.5 * w
            elif amenity == 'school' or building == 'school' or t_type == 'school':
                raw_poi_counts['school'] += 1
                poi_weighted_scores['campus'] += 1.2 * w

            if office or amenity == 'office' or building in ('office', 'commercial') or shop == 'office' or t_type in ('office', 'commercial'):
                raw_poi_counts['office'] += 1
                poi_weighted_scores['office'] += 2.5 * w

            if shop in ('mall', 'supermarket', 'department_store', 'clothes', 'electronics', 'convenience', 'boutique', 'center', 'retail') or amenity in ('marketplace', 'bank') or t_type in ('shopping_mall', 'shop', 'retail'):
                raw_poi_counts['commercial'] += 1
                poi_weighted_scores['commercial'] += 2.2 * w

            if building in ('residential', 'apartments', 'house', 'detached') or landuse == 'residential' or t_type in ('residential', 'apartments'):
                raw_poi_counts['residential'] += 1
                poi_weighted_scores['residential'] += 2.2 * w

            if amenity in ('bus_station', 'bus_stop', 'railway_station', 'subway_entrance') or highway == 'bus_stop' or railway in ('station', 'subway_entrance') or t_type in ('transit_station', 'bus_station', 'subway_station'):
                raw_poi_counts['transit'] += 1
                poi_weighted_scores['transit'] += 2.5 * w

            if amenity in ('hospital', 'clinic', 'doctors', 'pharmacy') or building == 'hospital' or t_type in ('hospital', 'clinic'):
                raw_poi_counts['medical'] += 1
                poi_weighted_scores['medical'] += 3.0 * w

            if tourism in ('hotel', 'hostel', 'guest_house', 'motel') or building in ('hotel', 'hostel') or t_type in ('lodging', 'hotel'):
                raw_poi_counts['hospitality'] += 1
                poi_weighted_scores['hospitality'] += 2.5 * w

            if landuse == 'industrial' or building == 'industrial' or t_type == 'industrial':
                raw_poi_counts['industrial'] += 1
                poi_weighted_scores['industrial'] += 2.5 * w

        # Food business density adds commercial footfall weight
        if comp_count > 0:
            avg_comp_w = sum(get_dist_weight(c.distance) for c in competitors) / float(comp_count)
            poi_weighted_scores['commercial'] += min(15.0, comp_count * 0.25 * avg_comp_w)

        sorted_scores = sorted(poi_weighted_scores.items(), key=lambda x: x[1], reverse=True)
        top1_cat, top1_score = sorted_scores[0]
        top2_cat, top2_score = sorted_scores[1]
        total_score_sum = sum(poi_weighted_scores.values())

        category_display_labels = {
            'campus': 'Student / Campus Area',
            'office': 'Corporate / IT Hub',
            'commercial': 'Shopping & Commercial Hub',
            'residential': 'Residential Zone',
            'transit': 'Transit & Footfall Hub',
            'medical': 'Medical & Healthcare District',
            'hospitality': 'Hospitality & Tourism Hub',
            'industrial': 'Industrial Zone'
        }

        mixed_combo_labels = {
            ('office', 'commercial'): 'Commercial / Office Area',
            ('commercial', 'office'): 'Commercial / Office Area',
            ('commercial', 'residential'): 'Mixed Commercial & Residential',
            ('residential', 'commercial'): 'Mixed Commercial & Residential',
            ('office', 'residential'): 'Corporate & Residential Zone',
            ('residential', 'office'): 'Corporate & Residential Zone',
            ('transit', 'commercial'): 'Transit & Commercial Hub',
            ('commercial', 'transit'): 'Transit & Commercial Hub',
            ('campus', 'commercial'): 'Student & Commercial Zone',
            ('commercial', 'campus'): 'Student & Commercial Zone',
            ('campus', 'office'): 'Campus & Corporate Zone',
            ('office', 'campus'): 'Campus & Corporate Zone',
            ('hospitality', 'commercial'): 'Hospitality & Commercial Zone',
            ('commercial', 'hospitality'): 'Hospitality & Commercial Zone',
            ('medical', 'residential'): 'Healthcare & Residential Zone',
            ('residential', 'medical'): 'Healthcare & Residential Zone',
            ('office', 'transit'): 'Corporate & Transit Hub',
            ('transit', 'office'): 'Corporate & Transit Hub'
        }

        # Dominant Classification Decision Logic:
        # Minimum evidence threshold required (total_score_sum >= 2.5)
        if total_score_sum < 2.5 or (poi_count == 0 and comp_count == 0):
            dominant_env = 'Mixed / Insufficient Evidence'
            confidence_pct = 50.0
        else:
            # Check for mixed environment if second category is at least 50% as strong as top category and score >= 2.5
            is_mixed = (top2_score >= 0.50 * top1_score) and (top2_score >= 2.5)
            if is_mixed and (top1_cat, top2_cat) in mixed_combo_labels:
                dominant_env = mixed_combo_labels[(top1_cat, top2_cat)]
            elif is_mixed:
                dominant_env = f"{category_display_labels[top1_cat]} & {category_display_labels[top2_cat]}"
            else:
                dominant_env = category_display_labels.get(top1_cat, 'Mixed Commercial & Residential')

            confidence_pct = round(min(95.0, max(50.0, (top1_score / max(1.0, total_score_sum)) * 100.0)), 1)

        env_reasons = []
        if raw_poi_counts['commercial'] > 0:
            env_reasons.append(f"{raw_poi_counts['commercial']} commercial & retail POIs detected.")
        if raw_poi_counts['office'] > 0:
            env_reasons.append(f"{raw_poi_counts['office']} corporate offices & workplaces detected.")
        if comp_count > 0:
            qualifier = "mapped" if query_result.snapshot_date else "active"
            env_reasons.append(f"{comp_count} {qualifier} food & dining businesses in area.")
        if raw_poi_counts['transit'] > 0:
            env_reasons.append(f"{raw_poi_counts['transit']} transit & transport access points.")
        if raw_poi_counts['residential'] > 0:
            env_reasons.append(f"{raw_poi_counts['residential']} residential complexes / buildings nearby.")
        if raw_poi_counts['hospitality'] > 0:
            env_reasons.append(f"{raw_poi_counts['hospitality']} hotels & lodging facilities detected.")
        if raw_poi_counts['medical'] > 0:
            env_reasons.append(f"{raw_poi_counts['medical']} medical & healthcare facilities detected.")
        if raw_poi_counts['campus'] > 0 or raw_poi_counts['school'] > 0:
            tot_ed = raw_poi_counts['campus'] + raw_poi_counts['school']
            env_reasons.append(f"{tot_ed} educational institution(s) / campus POIs.")
        if raw_poi_counts['industrial'] > 0:
            env_reasons.append(f"{raw_poi_counts['industrial']} industrial facilities detected.")

        close_pois = sum(1 for p in pois if p.distance <= 1000)
        env_reasons.append(f"Strongest signals within {radius_km} km radius ({close_pois} POIs within 1km).")

        if not env_reasons:
            env_reasons.append("Standard local area footfall signals.")

        env_ms = round((time.perf_counter() - t_env_0) * 1000.0, 1)

        # 4. Normalized Market Opportunity Scoring Model (0-100 Scale)
        t_mkt_0 = time.perf_counter()
        demand_opportunities = []
        max_expected_pois = max(4.0, float(poi_count))

        # Map poi_counts dictionary for opportunity profile lookups
        poi_counts_lookup = {
            'college': raw_poi_counts['campus'] + raw_poi_counts['school'],
            'office': raw_poi_counts['office'],
            'transit': raw_poi_counts['transit'],
            'mall': raw_poi_counts['commercial'],
            'hospital': raw_poi_counts['medical'],
            'hotel': raw_poi_counts['hospitality'],
            'residential': raw_poi_counts['residential']
        }

        for cat_name, profile in cls.CATEGORY_POI_PROFILES.items():
            c_count = category_map.get(cat_name, 0)
            relevant_pois = sum(poi_counts_lookup.get(p_type, 0) for p_type in profile['pois'])
            s_poi = min(100.0, (relevant_pois / max(1.0, max_expected_pois * 0.5)) * 100.0)

            category_share = (c_count / float(max(1, comp_count))) if comp_count > 0 else 0.0
            if c_count == 0:
                s_sat = 95.0
            else:
                s_sat = max(10.0, 100.0 - (category_share * 160.0) - (c_count * 8.0))

            s_fit = 85.0 if any(k in dominant_env for k in profile['env_pref'].split('/')) else 50.0

            opportunity_score = int(round(0.45 * s_poi + 0.40 * s_sat + 0.15 * s_fit))
            opportunity_score = min(98, max(10, opportunity_score))

            sat_level = 'NONE' if c_count == 0 else ('LOW' if category_share <= 0.15 and c_count <= 2 else ('MODERATE' if category_share <= 0.30 and c_count <= 5 else 'HIGH'))
            opp_level = 'HIGH' if opportunity_score >= 72 else ('MEDIUM' if opportunity_score >= 48 else 'LOW')

            reasons = []
            if relevant_pois > 0:
                poi_breakdown_parts = []
                if poi_counts_lookup.get('college', 0) > 0 and 'college' in profile['pois']:
                    poi_breakdown_parts.append(f"{poi_counts_lookup['college']} educational spots")
                if poi_counts_lookup.get('office', 0) > 0 and 'office' in profile['pois']:
                    poi_breakdown_parts.append(f"{poi_counts_lookup['office']} offices")
                if poi_counts_lookup.get('transit', 0) > 0 and 'transit' in profile['pois']:
                    poi_breakdown_parts.append(f"{poi_counts_lookup['transit']} transit stations")
                if poi_counts_lookup.get('mall', 0) > 0 and 'mall' in profile['pois']:
                    poi_breakdown_parts.append(f"{poi_counts_lookup['mall']} commercial/retail spots")

                breakdown_str = (", ".join(poi_breakdown_parts)) if poi_breakdown_parts else f"{relevant_pois} POIs"
                reasons.append(f"{relevant_pois} relevant POIs detected ({breakdown_str}) in surrounding area.")
            else:
                reasons.append("Standard local area footfall signals.")

            if c_count == 0:
                reasons.append(f"0 direct {cat_name.lower()} competitors detected (0% category share).")
            else:
                reasons.append(f"{c_count} competitor(s) detected ({round(category_share * 100, 1)}% of market).")

            reasons.append(f"Category market saturation is {sat_level.lower()}.")

            demand_opportunities.append({
                'category': cat_name,
                'opportunity_score': opportunity_score,
                'opportunity_level': opp_level,
                'competitor_count': c_count,
                'relevant_poi_count': relevant_pois,
                'saturation_level': sat_level,
                'category_share_pct': round(category_share * 100.0, 1),
                'supporting_reasons': reasons[:3]
            })

        demand_opportunities = sorted(demand_opportunities, key=lambda x: x['opportunity_score'], reverse=True)
        top_3_opportunities = demand_opportunities[:3]

        # 5. Market Gap Detection
        mean_opp_score = sum(d['opportunity_score'] for d in demand_opportunities) / float(len(demand_opportunities))
        avg_cat_comps = float(comp_count) / float(len(demand_opportunities))

        market_gaps = []
        for dop in demand_opportunities:
            is_high_opp = dop['opportunity_score'] >= max(58, mean_opp_score)
            is_low_comp = dop['competitor_count'] <= max(1, int(round(avg_cat_comps)))
            is_low_sat = dop['saturation_level'] in ('NONE', 'LOW')

            if is_high_opp and is_low_comp and is_low_sat:
                why_text = f"WHAT WAS OBSERVED: Strong relevant POI density ({dop['relevant_poi_count']} POIs) with below-average competition ({dop['competitor_count']} competitor(s) vs market avg {round(avg_cat_comps, 1)}). WHY IT MATTERS: Potential to capture unmet local dining demand in {dop['category']}."
                market_gaps.append({
                    'category': dop['category'],
                    'opportunity_score': dop['opportunity_score'],
                    'competitor_count': dop['competitor_count'],
                    'relevant_poi_count': dop['relevant_poi_count'],
                    'competition_level': dop['saturation_level'],
                    'category_share_pct': dop['category_share_pct'],
                    'why_underserved': why_text,
                    'key_drivers': dop['supporting_reasons']
                })

        # 6. Actionable Insights
        insights = []
        if raw_poi_counts['campus'] > 0 or raw_poi_counts['school'] > 0:
            tot_ed = raw_poi_counts['campus'] + raw_poi_counts['school']
            insights.append({
                'msg': f"WHAT WAS OBSERVED: {tot_ed} educational spots detected within {radius_km} km. WHY IT MATTERS: Student areas create strong daily demand for budget fast-food, snacks, and beverages.",
                'icon': 'fa-solid fa-graduation-cap',
                'severity': 'info'
            })
        if raw_poi_counts['office'] > 0:
            insights.append({
                'msg': f"WHAT WAS OBSERVED: {raw_poi_counts['office']} commercial workplaces detected. WHY IT MATTERS: Office areas generate high weekday lunch peak orders and beverage breaks.",
                'icon': 'fa-solid fa-briefcase',
                'severity': 'info'
            })
        if raw_poi_counts['transit'] > 0:
            insights.append({
                'msg': f"WHAT WAS OBSERVED: {raw_poi_counts['transit']} transport stations nearby. WHY IT MATTERS: High commuter density drives grab-and-go food and quick-service purchases.",
                'icon': 'fa-solid fa-train-subway',
                'severity': 'info'
            })
        if comp_count > 0:
            top_cat = max(category_map.items(), key=lambda x: x[1]) if category_map else ('Dining', 0)
            if top_cat[1] > 0:
                top_pct = round((top_cat[1] / float(comp_count)) * 100.0, 1)
                insights.append({
                    'msg': f"WHAT WAS OBSERVED: {top_cat[0]} accounts for {top_cat[1]} places ({top_pct}% of market). WHY IT MATTERS: Indicates high category competition in the immediate radius.",
                    'icon': 'fa-solid fa-chart-pie',
                    'severity': 'info'
                })
        if market_gaps:
            gap_names = ", ".join([g['category'] for g in market_gaps[:2]])
            insights.append({
                'msg': f"WHAT WAS OBSERVED: [{gap_names}] displays high POI opportunity score with low relative competition. WHY IT MATTERS: High potential to capture unmet local dining demand.",
                'icon': 'fa-solid fa-bullseye',
                'severity': 'high'
            })

        overall_opp_score = demand_opportunities[0]['opportunity_score'] if demand_opportunities else 60
        market_ms = round((time.perf_counter() - t_mkt_0) * 1000.0, 1)
        total_an_ms = round((time.perf_counter() - t_an_start) * 1000.0, 1)

        perf_metrics = {
            'env_ms': env_ms,
            'market_ms': market_ms,
            'analyzer_total_ms': total_an_ms,
            'parsing_ms': query_result.perf_metrics.get('parsing_ms', 0.0) if hasattr(query_result, 'perf_metrics') else 0.0,
            'distance_ms': query_result.perf_metrics.get('distance_ms', 0.0) if hasattr(query_result, 'perf_metrics') else 0.0,
            'overpass_ms': query_result.perf_metrics.get('overpass_ms', 0.0) if hasattr(query_result, 'perf_metrics') else 0.0
        }

        return {
            'success': True,
            'configured': True,
            'data_available': True,
            'data_state': DataState.REAL_DATA.value,
            'opportunity_score': overall_opp_score,
            'perf_metrics': perf_metrics,
            'environment_summary': {
                'environment_type': dominant_env,
                'confidence': confidence_pct,
                'supporting_pois': poi_counts_lookup,
                'evidence': env_reasons
            },
            'competitor_summary': {
                'count': comp_count,
                'within_1km': comp_1km,
                'within_2km': comp_2km,
                'within_radius': comp_count,
                'avg_distance': avg_dist,
                'min_distance': min_dist,
                'max_distance': max_dist,
                'avg_rating': avg_rating_str,
                'total_reviews': total_reviews,
                'category_counts': active_category_counts,
                'cuisine_counts': cuisine_counts
            },
            'competitors': [c.to_dict() for c in competitors],
            'demand_opportunities': demand_opportunities,
            'top_3_opportunities': top_3_opportunities,
            'market_gaps': market_gaps,
            'insights': insights,
            'data_lineage': {
                'source_provider': query_result.provider_name or 'OpenStreetMap Overpass API',
                'competitors_source': query_result.provider_name or 'OpenStreetMap Overpass API',
                'poi_source': query_result.provider_name or 'OpenStreetMap Overpass API',
                'latitude': lat,
                'longitude': lng,
                'search_radius': f"{radius_km} km",
                'competitors_found': comp_count,
                'pois_found': poi_count,
                'environment_signals': poi_counts_lookup,
                'data_state': DataState.REAL_DATA.value,
                'timestamp': query_result.snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                'category_counts': active_category_counts
            }
        }

    @classmethod
    def _build_non_real_data_payload(cls, query_result: QueryResult, lat: float, lng: float, radius: float) -> Dict[str, Any]:
        """
        Builds standardized response structure for non-REAL_DATA states (NO_RESULTS, ZERO_RESULTS, PROVIDER_ERROR, GEOCODING_ERROR, INVALID_LOCATION).
        """
        radius = float(radius or 1000)
        radius_km = round(radius / 1000.0, 1)
        state_str = query_result.status.value

        if query_result.status in (DataState.NO_RESULTS, DataState.ZERO_RESULTS):
            msg = f"No nearby food competitors or relevant POIs were found within {radius_km} km."
            env_type = "Zero Results Area"
        elif query_result.status == DataState.PROVIDER_ERROR:
            msg = query_result.error_message or "Real market data is currently unavailable. Please configure a valid Google Places API Key."
            env_type = "Provider Failure"
        elif query_result.status == DataState.GEOCODING_ERROR:
            msg = query_result.error_message or "Location address could not be resolved."
            env_type = "Geocoding Error"
        else:  # INVALID_LOCATION
            msg = query_result.error_message or "Invalid latitude or longitude coordinates provided."
            env_type = "Invalid Location"

        source_name = query_result.provider_name or ('Unavailable' if query_result.status == DataState.PROVIDER_ERROR else 'OpenStreetMap Overpass API')

        return {
            'success': True,
            'configured': True,
            'data_available': False,
            'data_state': state_str,
            'message': msg,
            'opportunity_score': 0,
            'environment_summary': {
                'environment_type': env_type,
                'confidence': 0.0,
                'supporting_pois': {},
                'evidence': [msg]
            },
            'competitor_summary': {
                'count': 0,
                'within_1km': 0,
                'within_2km': 0,
                'within_radius': 0,
                'avg_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'avg_rating': 'N/A',
                'total_reviews': 0,
                'category_counts': {}
            },
            'competitors': [],
            'demand_opportunities': [],
            'top_3_opportunities': [],
            'market_gaps': [],
            'insights': [{
                'msg': f"WHAT WAS OBSERVED: {msg} WHY IT MATTERS: Analysis requires external market signals.",
                'icon': 'fa-solid fa-triangle-exclamation',
                'severity': 'warning' if query_result.status == DataState.PROVIDER_ERROR else 'info'
            }],
            'data_lineage': {
                'source_provider': source_name,
                'competitors_source': source_name,
                'poi_source': source_name,
                'latitude': lat,
                'longitude': lng,
                'search_radius': f"{radius_km} km",
                'competitors_found': 0,
                'pois_found': 0,
                'data_state': state_str,
                'timestamp': query_result.snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                'category_counts': {}
            }
        }

