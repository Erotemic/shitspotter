"""
SeeAlso:
    python ~/code/shitspotter/papers/neurips-2025/scripts/estimate_training_resources.py
    ~/code/shitspotter/papers/neurips-2025/scripts/build_v2_result_table.py
"""


def devcheck_main_stuff():
    import sys
    import ubelt as ub
    sys.path.append(ub.Path('~/code/shitspotter/papers/neurips-2025/scripts').expand())
    from build_v2_result_table import load_aggregators
    from build_v2_result_table import process_aggregators
    from build_v2_result_table import report_resources
    aggregators_rows = load_aggregators()
    summary_df = process_aggregators(aggregators_rows)
    report_resources(aggregators_rows)


def main():
    # Fill in this with best estimates for final table
    data_table = {

    }
