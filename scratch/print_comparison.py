import json

with open('runs/reports/phase6e/phase6e_diagnostic_metrics.json') as f:
    d = json.load(f)

leg = d['legacy_features_evaluation']['models']
hard = d['hardened_features_evaluation']['models']

models = ['historical_v2_clean', 'model_a_frozen', 'model_b_unfrozen']

print('| Model | Feature Pipeline | Threshold | Accuracy | Bal Acc | Macro-F1 | LOW Rec | LOW Prec | HIGH Rec | HIGH Prec |')
print('| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')

for m in models:
    name = leg[m]['name']
    l_50 = leg[m]['metrics_tau50']
    h_50 = hard[m]['metrics_tau50']
    cal_t = leg[m]['calibrated_threshold']
    l_cal = leg[m]['metrics_calibrated']
    h_cal = hard[m]['metrics_calibrated']

    for p_name, res_50, res_cal in [('Legacy (Ch23=0)', l_50, l_cal), ('Hardened (Ch23 Active)', h_50, h_cal)]:
        print(f"| {name} | {p_name} | 0.50 | {res_50['accuracy']*100:.2f}% | {res_50['balanced_accuracy']*100:.2f}% | {res_50['macro_f1']*100:.2f}% | {res_50['class_low']['recall']*100:.2f}% | {res_50['class_low']['precision']*100:.2f}% | {res_50['class_high']['recall']*100:.2f}% | {res_50['class_high']['precision']*100:.2f}% |")
        print(f"| | {p_name} | {cal_t:.2f} (cal) | {res_cal['accuracy']*100:.2f}% | {res_cal['balanced_accuracy']*100:.2f}% | {res_cal['macro_f1']*100:.2f}% | {res_cal['class_low']['recall']*100:.2f}% | {res_cal['class_low']['precision']*100:.2f}% | {res_cal['class_high']['recall']*100:.2f}% | {res_cal['class_high']['precision']*100:.2f}% |")
