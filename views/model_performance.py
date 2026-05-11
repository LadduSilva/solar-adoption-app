import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def show():

    @st.cache_data
    def load_data():
        results_df    = pd.read_csv("Research/model_results.csv")
        return results_df

    @st.cache_resource
    def load_artifacts():
        model         = joblib.load("Research/lr_model.pkl")
        scaler        = joblib.load("Research/scaler.pkl")
        model_columns = joblib.load("Research/model_columns.pkl")
        return model, scaler, model_columns

    results_df    = load_data()
    model, scaler, model_columns = load_artifacts()

    st.title("📊 Model Performance Comparison")
    st.markdown("Comparison of Logistic Regression, Random Forest, and XGBoost models.")
    st.divider()

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("📋 Model Metrics")
    st.dataframe(results_df.style.highlight_max(
        subset=["Test Accuracy","F1 Score","ROC-AUC"], color="#d4edda"),
        use_container_width=True)

    st.divider()

    # ── Load extra artifacts ──────────────────────────────────────────────────
    @st.cache_resource
    def load_models_and_test():
        from xgboost import XGBClassifier
        lr  = joblib.load("Research/lr_model.pkl")
        rf  = joblib.load("Research/rf_model.pkl")
        xgb = joblib.load("Research/xgb_model.pkl")
        X_test_lr = joblib.load("Research/X_test_lr.pkl")
        X_test    = joblib.load("Research/X_test.pkl")
        y_test    = joblib.load("Research/y_test.pkl")
        return lr, rf, xgb, X_test_lr, X_test, y_test

    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
        lr, rf, xgb, X_test_lr, X_test, y_test = load_models_and_test()

        models = {
            "Logistic Regression": (lr,  X_test_lr),
            "Random Forest":       (rf,  X_test),
            "XGBoost":             (xgb, X_test),
        }

        # ── Confusion Matrices ────────────────────────────────────────────────
        st.subheader("🔢 Confusion Matrices")
        cm_cols = st.columns(3)

        for i, (name, (m, X)) in enumerate(models.items()):
            with cm_cols[i]:
                st.markdown(f"**{name}**")
                cm  = confusion_matrix(y_test, m.predict(X))
                fig, ax = plt.subplots(figsize=(3.5, 3))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                             display_labels=["Not Adopt","Adopt"])
                disp.plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(name, fontsize=9)
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        st.divider()

        # ── ROC Curves ────────────────────────────────────────────────────────
        st.subheader("📈 ROC Curves")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors  = ["#3498db", "#2ecc71", "#e74c3c"]

        for (name, (m, X)), color in zip(models.items(), colors):
            y_prob        = m.predict_proba(X)[:, 1]
            fpr, tpr, _   = roc_curve(y_test, y_prob)
            roc_auc       = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=color, lw=2)

        ax.plot([0,1], [0,1], linestyle="--", color="gray", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    except Exception as e:
        st.warning(f"⚠️ Could not load model files for charts: {e}")