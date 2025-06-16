def calculate_grades(avg, hr, run_time, is_lhh, pop_time=None):
    def get_hit_grade(avg):
        scale = [(0.315, 80), (0.295, 70), (0.275, 60), (0.265, 55),
                 (0.255, 50), (0.245, 45), (0.235, 40), (0.215, 30), (0, 20)]
        for threshold, grade in scale:
            if avg >= threshold:
                return grade
        return 20

    def get_power_grade(hr):
        scale = [(40, 80), (34, 70), (28, 60), (23, 55),
                 (19, 50), (14, 45), (10, 40), (5, 30), (0, 20)]
        for threshold, grade in scale:
            if hr >= threshold:
                return grade
        return 20

    def get_speed_grade(time, is_lhh):
        thresholds = [3.90, 4.00, 4.05, 4.10, 4.15,
                      4.20, 4.25, 4.30, 4.40, 4.50]
        grades = [80, 70, 65, 60, 55, 50, 45, 40, 30, 20]
        for t, g in zip(thresholds, grades):
            if time <= t:
                return g
        return 20

    def get_arm_grade(pop_time):
        if pop_time is None:
            return 50
        thresholds = [1.90, 1.94, 1.99, 2.04, 2.09, 2.14]
        grades = [80, 70, 60, 50, 40, 30]
        for t, g in zip(thresholds, grades):
            if pop_time <= t:
                return g
        return 20

    return {
        'HIT': get_hit_grade(avg),
        'POWER': get_power_grade(hr),
        'SPEED': get_speed_grade(run_time, is_lhh),
        'ARM': get_arm_grade(pop_time)
    }

def grade_description(score):
    desc = {
        80: "Elite",
        70: "Plus-Plus",
        60: "Plus",
        55: "Above-Average",
        50: "Major League Average",
        45: "Fringe-Average",
        40: "Below-Average",
        30: "Poor",
        20: "Very Poor"
    }
    return desc.get(score, "Unknown")

def generate_final_summary(name, grades):
    comps = {
        (70, 55, 55): [("Luis Arraez", 3.5), ("Steven Kwan", 4.1), ("Ichiro-lite", 5.0)]
    }
    comp_key = (grades['HIT'], grades['POWER'], grades['SPEED'])
    players = comps.get(comp_key, [])
    comp_names = ', '.join(p[0] for p in players)
    avg_war = round(sum(p[1] for p in players)/len(players), 1) if players else 'N/A'

    summary = f"""
📋 Scouting Report: {name}

- HIT ({grades['HIT']}): {grade_description(grades['HIT'])}
- POWER ({grades['POWER']}): {grade_description(grades['POWER'])}
- SPEED ({grades['SPEED']}): {grade_description(grades['SPEED'])}
- ARM ({grades['ARM']}): {grade_description(grades['ARM'])}

🎯 Final Take:
{name} projects as a high-contact hitter with average to above-average power and speed. Suitable for top-of-the-order roles. Defense and arm are stable, though not standout tools.

🧑‍💻 Comparable Players:
{comp_names}
Avg WAR of Comps: {avg_war}
"""
    return summary.strip()