# ============================================================
# config.py — Central configuration for the betting model
# ============================================================

FEATURES = [
    # --- Goals rolling ---
    "home_goals_last5", "away_goals_last5",
    "home_goals_last3", "away_goals_last3",
    "home_goals_ewm",   "away_goals_ewm",
    # --- Defense rolling ---
    "home_conceded_last5", "away_conceded_last5",
    "home_conceded_last3", "away_conceded_last3",
    # --- Home/away splits ---
    "home_goals_at_home_last5", "home_conceded_at_home_last5",
    "away_goals_away_last5",    "away_conceded_away_last5",
    # --- Shots & possession ---
    "home_shots_last5", "away_shots_last5",
    "home_sot_last5",   "away_sot_last5",
    "home_pos_last5",   "away_pos_last5",
    # --- Efficiency ---
    "home_shot_accuracy", "away_shot_accuracy",
    "home_conversion_last5", "away_conversion_last5",
    # --- Form & momentum ---
    "home_form_last5",  "away_form_last5",  "form_diff",
    # --- H2H ---
    "h2h_avg_goals",    "h2h_over25_rate",
    # --- Attack composite ---
    "home_attack_score","away_attack_score","combined_attack",
    # --- ELO ---
    "home_elo", "away_elo", "elo_diff", "elo_home_win_prob", "elo_sum",
    # --- Poisson xG ---
    "exp_home_goals",   "exp_away_goals",
    # --- Rest days (NEW) ---
    "home_rest_days",   "away_rest_days",   "rest_diff",
    "home_fatigued",    "away_fatigued",
    # --- Table position (NEW) ---
    "home_table_pos",   "away_table_pos",   "table_pos_diff",
    "home_table_pts",   "away_table_pts",
    "home_table_gd",    "away_table_gd",
]

TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.85

MIN_EDGE        = 0.05
MAX_MODEL_PROB  = 0.80
MIN_KELLY       = 0.0

KELLY_FRACTION  = 0.25
STARTING_UNITS  = 100

STOP_LOSS_PCT       = 0.20
MAX_DAILY_EXPOSURE  = 0.10