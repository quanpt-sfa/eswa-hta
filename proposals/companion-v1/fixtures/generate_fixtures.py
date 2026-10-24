"""
Generate golden_cases.csv and golden_expectations.csv with verified mathematical derivations.
"""

import csv
import math
from pathlib import Path
from scipy import stats

fixtures_dir = Path(__file__).resolve().parent

cases_header = [
    "test_id",
    "description",
    "pat_pre",
    "pat_post",
    "total_assets_pre",
    "k",
    "dist_family",
    "dist_param1",
    "dist_param2",
    "c",
    "pi_zero"
]

expectations_header = [
    "test_id",
    "eligible_label",
    "eligible_model",
    "is_zero_adj",
    "y_k",
    "expected_prob",
    "lower_delta_cut",
    "upper_delta_cut",
    "tolerance"
]

cases_data = [
    # Label and threshold cases
    ("T-L01", "Below threshold adjustment", "100", "104", "1000", "0.05", "", "", "", "", ""),
    ("T-L02", "Inclusive threshold adjustment (positive)", "100", "105", "1000", "0.05", "", "", "", "", ""),
    ("T-L03", "Negative adjustment at threshold", "100", "95", "1000", "0.05", "", "", "", "", ""),
    ("T-L04", "Zero pre-audit PAT (ineligible)", "0", "10", "1000", "0.05", "", "", "", "", ""),
    ("T-L05", "Sign change (ineligible)", "100", "-100", "1000", "0.05", "", "", "", "", ""),
    ("T-L06", "Zero post-audit PAT (eligible)", "100", "0", "1000", "0.05", "", "", "", "", ""),
    ("T-L07", "Negative pre-audit PAT with negative adjustment at threshold", "-100", "-105", "1000", "0.05", "", "", "", "", ""),
    ("T-L08", "Negative pre-audit PAT with positive adjustment at threshold", "-100", "-95", "1000", "0.05", "", "", "", "", ""),
    ("T-L09", "Negative pre-audit PAT below threshold", "-100", "-96", "1000", "0.05", "", "", "", "", ""),
    
    # Zero adjustment and near-zero tests
    ("T-Z01", "Exact zero adjustment", "100", "100", "1000", "0.05", "", "", "", "", ""),
    ("T-Z02", "Small adjustment below materiality threshold (corrected)", "100", "100.0000000001", "1000", "0.05", "", "", "", "", ""),
    ("T-Z03", "Near-zero earnings with genuine threshold exceedance", "1e-9", "1.1e-9", "1000", "0.05", "", "", "", "", ""),
    
    # Asset eligibility
    ("T-M01", "Zero assets (eligible_label True, eligible_model False)", "100", "105", "0", "0.05", "", "", "", "", ""),
    ("T-M02", "Negative assets (eligible_label True, eligible_model False)", "100", "105", "-500", "0.05", "", "", "", "", ""),
    
    # Distribution and exceedance probabilities
    ("T-C01", "Standard normal exceedance at c=1.0", "", "", "", "", "normal", "0", "1", "1.0", "0.0"),
    ("T-C02", "Laplace exceedance at c=1.0", "", "", "", "", "laplace", "0", "1", "1.0", "0.0"),
    ("T-H01", "Hurdle normal mixture (pi_zero=0.4, c=1.0)", "", "", "", "", "normal", "0", "1", "1.0", "0.4"),
    
    # Asymmetric post-denominator cuts
    ("T-P01", "Asymmetric post-denominator cut (positive PAT_pre)", "100", "", "1000", "0.05", "", "", "", "", ""),
    ("T-P02", "Asymmetric post-denominator cut (negative PAT_pre)", "-100", "", "1000", "0.05", "", "", "", "", ""),
    
    # Nested monotonic probabilities
    ("T-N01", "Monotonic nested exceedance probabilities (Normal)", "", "", "", "", "normal", "0", "1", "0.05", "0.0")
]

# Calculate exact probabilities
norm_c1 = 2 * (1 - stats.norm.cdf(1.0)) # 0.31731050786291415
laplace_c1 = math.exp(-1.0) # 0.36787944117144233
hurdle_h01 = (1 - 0.4) * norm_c1 # 0.19038630471774849
norm_c005 = 2 * (1 - stats.norm.cdf(0.05)) # 0.9601136279172403

# Calculate post-cuts
# For PAT_pre = 100, k = 0.05:
# Lower: -100 * 0.05 / (1 + 0.05) = -4.761904761904762
# Upper: +100 * 0.05 / (1 - 0.05) = +5.263157894736842
# For PAT_pre = -100, k = 0.05:
# Lower: -100 * 0.05 / (1 - 0.05) = -5.263157894736842
# Upper: +100 * 0.05 / (1 + 0.05) = +4.761904761904762

expectations_data = [
    # test_id, eligible_label, eligible_model, is_zero_adj, y_k, expected_prob, lower_delta_cut, upper_delta_cut, tolerance
    ("T-L01", "True", "True", "False", "0", "", "", "", "0.0"),
    ("T-L02", "True", "True", "False", "1", "", "", "", "0.0"),
    ("T-L03", "True", "True", "False", "1", "", "", "", "0.0"),
    ("T-L04", "False", "False", "", "", "", "", "", "0.0"),
    ("T-L05", "False", "False", "", "", "", "", "", "0.0"),
    ("T-L06", "True", "True", "False", "1", "", "", "", "0.0"),
    ("T-L07", "True", "True", "False", "1", "", "", "", "0.0"),
    ("T-L08", "True", "True", "False", "1", "", "", "", "0.0"),
    ("T-L09", "True", "True", "False", "0", "", "", "", "0.0"),
    
    ("T-Z01", "True", "True", "True", "0", "", "", "", "0.0"),
    ("T-Z02", "True", "True", "False", "0", "", "", "", "0.0"),
    ("T-Z03", "True", "True", "False", "1", "", "", "", "0.0"),
    
    ("T-M01", "True", "False", "False", "1", "", "", "", "0.0"),
    ("T-M02", "True", "False", "False", "1", "", "", "", "0.0"),
    
    ("T-C01", "", "", "", "", f"{norm_c1:.17f}", "", "", "1e-10"),
    ("T-C02", "", "", "", "", f"{laplace_c1:.17f}", "", "", "1e-10"),
    ("T-H01", "", "", "", "", f"{hurdle_h01:.17f}", "", "", "1e-10"),
    
    ("T-P01", "", "", "", "", "", "-4.761904761904762", "5.263157894736842", "1e-8"),
    ("T-P02", "", "", "", "", "", "-5.263157894736842", "4.761904761904762", "1e-8"),
    
    ("T-N01", "", "", "", "", f"{norm_c005:.17f}", "", "", "1e-10")
]

def write_csv(filepath, header, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            assert len(r) == len(header), f"Row length mismatch for {r[0]}: {len(r)} != {len(header)}"
            writer.writerow(r)

if __name__ == "__main__":
    cases_path = fixtures_dir / "golden_cases.csv"
    exp_path = fixtures_dir / "golden_expectations.csv"
    write_csv(cases_path, cases_header, cases_data)
    write_csv(exp_path, expectations_header, expectations_data)
    print(f"Generated {cases_path} with {len(cases_data)} rows.")
    print(f"Generated {exp_path} with {len(expectations_data)} rows.")
