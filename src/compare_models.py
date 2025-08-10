import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def compare_and_plot(models_info, X_test, y_test, save_path="reports/figures/roc_comparison.png"):
    plt.figure(figsize=(8,6))
    for name, model in models_info.items():
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC comparison")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
