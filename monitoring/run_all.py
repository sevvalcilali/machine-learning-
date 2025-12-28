# monitoring/run_all.py

from monitoring.plots import run_all_plots
from monitoring.calc_accuracy import main as accuracy_main
from monitoring.advanced_monitoring import main as advanced_main

def main():
    # 1) ROC / PR / Threshold / Probability distribution (basic evaluation)
    run_all_plots()

    # 2) Accuracy çıktısı
    accuracy_main()

    # 3) Advanced monitoring (label-aware dist + over-time + PSI + alerts)
    advanced_main()

    print("✅ All monitoring outputs (basic + advanced) generated under /reports")

if __name__ == "__main__":
    main()
