def calculate_risk(aqi, duration_hours, user_category):
    
    # AQI weight mapping
    if aqi <= 50:
        weight = 10
    elif aqi <= 100:
        weight = 20
    elif aqi <= 200:
        weight = 40
    elif aqi <= 300:
        weight = 70
    else:
        weight = 100

    # Vulnerability multiplier
    vulnerability = {
        "adult": 1.0,
        "child": 1.8,
        "elderly": 2.2,
        "asthma": 2.8,
        "outdoor_worker": 1.4,
        "pregnant_woman": 2.2
    }
    multiplier = vulnerability.get(user_category, 1.0)  # Default to adult if unknown

    # Calculate risk score
    score = weight * duration_hours * multiplier

    # Determine level, color, recommendation
    if score <= 30:
        level = "Low"
        color = "#00FF00"
        recommendation = "Outdoor activity is generally acceptable. Keep checking forecast."
    elif score <= 60:
        level = "Moderate"
        color = "#FFFF00"
        recommendation = "Limit prolonged outdoor exposure. Sensitive users should be careful."
    elif score <= 90:
        level = "High"
        color = "#FFA500"
        recommendation = "Wear N95/KN95 mask outdoors. Reduce travel time. Use indoor purifier."
    else:
        level = "Severe"
        color = "#FF0000"
        recommendation = "Avoid outdoor activity. Shift plans indoors. Alerts for children, elderly, respiratory sensitivity."

    return {
        "score": round(score, 2),
        "level": level,
        "color": color,
        "recommendation": recommendation
    }