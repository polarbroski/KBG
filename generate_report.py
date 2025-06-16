from metrics_to_2080 import calculate_grades, generate_final_summary

def generate_scouting_report(player_info):
    grades = calculate_grades(
        avg=player_info['avg'],
        hr=player_info['hr'],
        run_time=player_info['run_time'],
        is_lhh=player_info['is_lhh'],
        pop_time=player_info['pop_time']
    )

    summary = generate_final_summary(player_info['name'], grades)
    return summary