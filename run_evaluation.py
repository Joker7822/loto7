
# run_evaluation.py
# -*- coding: utf-8 -*-
from evaluation import evaluate_predictions_with_bonus, simulate_random_baseline, summarize_random_baseline, summarize_comparison

PRED_FILE = 'loto7_predictions.csv'
RESULTS_FILE = 'loto7.csv'

model_detail = evaluate_predictions_with_bonus(PRED_FILE, RESULTS_FILE)
n_per_draw = 5
random_detail = simulate_random_baseline(RESULTS_FILE, n_per_draw=n_per_draw, trials=1000)
random_best = summarize_random_baseline(random_detail)
random_best.to_csv('random_best.csv', index=False, encoding='utf-8-sig')
summarize_comparison(model_detail, random_best, output_prefix='comparison')
print('Done.')
