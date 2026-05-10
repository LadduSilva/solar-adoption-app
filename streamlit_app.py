
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solar Adoption Predictor", page_icon="☀️", layout="wide")

@st.cache_resource
def load_artifacts():
    model         = joblib.load("Research/lr_model.pkl")
    scaler        = joblib.load("Research/scaler.pkl")
    model_columns = joblib.load("Research/model_columns.pkl")
    return model, scaler, model_columns

@st.cache_data
def load_data():
    results_df    = pd.read_csv("Research/model_results.csv")
    importance_df = pd.read_csv("Research/grouped_feature_importance.csv")
    return results_df, importance_df

model, scaler, model_columns = load_artifacts()
results_df, importance_df    = load_data()
top_importance_df            = importance_df.head(10)

def predict_solar_adoption(area, socio_class, education, occupation,
    monthly_expenditure, household_size, house_ownership, built_year,
    type_of_house, floor_area, cooking_fuel, water_heating_method):

    input_df = pd.DataFrame([{
        "monthly_expenditure":           monthly_expenditure,
        "household_size":                household_size,
        "floor_area":                    floor_area,
        "electricity_provider_csc_area": area,
        "socio_economic_class":          socio_class,
        "education":                     education,
        "occupation":                    occupation,
        "house_ownership":               house_ownership,
        "built_year_of_the_house":       built_year,
        "type_of_house":                 type_of_house,
        "cooking_fuel":                  cooking_fuel,
        "water_heating_method":          water_heating_method
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    scale_cols = ["monthly_expenditure", "floor_area", "household_size"]
    input_encoded[scale_cols] = scaler.transform(input_encoded[scale_cols])

    prediction  = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    label = "✅ Likely Solar Adopter" if prediction == 1 else "❌ Not Likely to Adopt Solar"
    return label, probability

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("☀️ Solar Adoption")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard",
    "🔍 Prediction",
    "📊 Model Performance",
    "🔑 Influential Features"
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🏠 Solar Adoption Dashboard")
    st.markdown("Overview of solar energy adoption across households in the dataset.")
    st.divider()

    # Load original dataset
    @st.cache_data
    def load_raw_data():
        return pd.read_excel("solar_adoption_data_set.xlsx")

    raw_df = load_raw_data()
    total       = len(raw_df)
    adopters    = (raw_df["solar_generation"] == "Yes").sum()
    non_adopters = total - adopters
    adoption_rate = adopters / total * 100

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total Households",   f"{total:,}")
    m2.metric("✅ Solar Adopters",      f"{adopters:,}")
    m3.metric("❌ Non Adopters",        f"{non_adopters:,}")
    m4.metric("☀️ Adoption Rate",       f"{adoption_rate:.1f}%")

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Adoption by Socio-Economic Class")
        sec_data = raw_df.groupby("socio_economic_class")["solar_generation"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).round(1).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(sec_data.index, sec_data.values, color=["#2ecc71","#3498db","#f39c12","#e74c3c","#9b59b6"])
        ax.set_ylabel("Adoption Rate (%)")
        ax.set_title("Solar Adoption Rate by Socio-Economic Class")
        for bar, val in zip(bars, sec_data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val}%", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Adopters vs Non-Adopters")
        _, inner, _ = st.columns([0.5, 2, 0.5])
        with inner:
            fig, ax = plt.subplots(figsize=(3.5, 3))
            ax.pie(
                [adopters, non_adopters],
                labels=["Solar Adopters", "Non Adopters"],
                autopct="%1.1f%%",
                colors=["#f4a261", "#2ecc71"],
                startangle=90,
                textprops={"fontsize": 8},
                radius=0.9
            )
            ax.set_title("Overall Solar Adoption Distribution", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close()

    st.divider()

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Top 10 Areas by Adoption Rate")
        area_data = raw_df.groupby("electricity_provider_csc_area")["solar_generation"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).round(1).sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(area_data.index, area_data.values, color="#3498db")
        ax.invert_yaxis()
        ax.set_xlabel("Adoption Rate (%)")
        ax.set_title("Top 10 CSC Areas by Adoption Rate")
        for i, val in enumerate(area_data.values):
            ax.text(val + 0.3, i, f"{val}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c4:
        st.subheader("Adoption by House Type")
        house_data = raw_df.groupby("type_of_house")["solar_generation"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).round(1).sort_values(ascending=False)
        house_data = house_data[house_data > 0]  # only show house types with adoption

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(house_data.index, house_data.values, color="#f4a261")
        ax.invert_yaxis()
        ax.set_xlabel("Adoption Rate (%)")
        ax.set_title("Solar Adoption Rate by House Type")
        for i, val in enumerate(house_data.values):
            ax.text(val + 0.3, i, f"{val}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Key insights ──────────────────────────────────────────────────────────
    st.subheader("💡 Key Insights")
    i1, i2, i3 = st.columns(3)

    with i1:
        st.info("📈 **SEC A households** have the highest adoption rate at **30.4%**, nearly 3x higher than SEC B (10.4%)")
    with i2:
        st.info("🏠 **Double and multi-floor houses** are far more likely to adopt solar (20–28%) vs single floor houses (3.9%)")
    with i3:
        st.info("📍 **Nugegoda** leads all CSC areas with a **31.2%** adoption rate, followed by Boralasgamuwa (24.4%)")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prediction":
    st.title("🔍 Solar Adoption Prediction")
    st.markdown("Enter household details or upload an Excel file to predict solar adoption.")
    st.divider()

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # ── Single Prediction ─────────────────────────────────────────────────────
    with tab1:
        area_options = [
            "GALLE","BORALASGAMUWA","KOLONNAWA","AMBALANGODA","MAHARAGAMA",
            "MORATUWA SOUTH","SEEDUWA","JA-ELA","ALUTHGAMA","HIKKADUWA",
            "KANDANA","WATTALA","NEGOMBO","KALUTHARA","PANADURA","DALUGAMA",
            "PITA-KOTTE","MORATUWA NORTH","MAHARA","KESELWATTA","PAYAGALA",
            "NUGEGODA","KOTIKAWATTA","OTHER"
        ]
        area_selected = st.selectbox("Electricity Provider CSC Area", area_options)
        if area_selected == "OTHER":
            area_custom = st.text_input("Please specify your CSC Area")
            area = area_custom.strip().upper() if area_custom.strip() else "OTHER"
        else:
            area = area_selected

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                socio_class = st.selectbox("Socio Economic Class", ["SEC A","SEC B","SEC C","SEC D","SEC E"])
                education = st.selectbox("Education", [
                    "Graduate / Post-Grads / Degree level professional qualification",
                    "Diploma with O/L or A/L (Non graduate)",
                    "Other professional certificates with O/L or A/L / Part qualification (Non graduate)",
                    "O/L or A/L pending / Passed","Schooling up to Grade 6 - 9",
                    "Primary Education","Illiterate"])
                occupation = st.selectbox("Occupation", [
                    "Skilled Worker","Unskilled Worker","Middle and Senior executive",
                    "Manager / Professional","Clerk / Salesman grades",
                    "Junior executive / Executive",
                    "Small Businessman / Self employed (Non professional)",
                    "1-9 Employed","10+ Employed","Supervisor grades",
                    "Agricultural labourer / Worker",
                    "Self employed (Professional) - No employees","Boutique owner",
                    "Tenant cultivator","Farmer owning - Less than 1/2 Acre",
                    "Farmer owning - 1/2 - 1 Acre","Farmer owning - 1 - 2 Acre",
                    "Farmer owning - 2 - 5 Acre",
                    "Farmer owning - Over 5 acres / Landed proprietor"])
                monthly_expenditure = st.number_input("Monthly Expenditure (LKR)", min_value=0.0, step=100.0)
                household_size      = st.number_input("Household Size", min_value=1, step=1)

            with col2:
                house_ownership = st.selectbox("House Ownership", [
                    "Yes, I or a household member owns it.",
                    "No, I am living on rent and the rent is paid by me or a household member.",
                    "No, I am living on rent and the rent is paid by the employer.",
                    "No, I or any household member does not own or rent this household. We occupy this household without any payment of rent."])
                built_year = st.selectbox("Built Year of the House", [
                    "Before 1980","1980-1989","1990-1999","2000-2009",
                    "2010-2019","In 2020 or After 2020","Don't know"])
                type_of_house = st.selectbox("Type of House", [
                    "Single House - Single Floor","Single House - Double Floor",
                    "Single House - More than 2 floors","Flat",
                    "Attached house / Annex","Twin Houses","Slum / Shanty",
                    "Line room/row house","Condominium/ Luxury apartments","Other"])
                floor_area   = st.number_input("Floor Area (sq ft)", min_value=0.0, step=10.0)
                cooking_fuel = st.selectbox("Cooking Fuel", [
                    "Gas","Electricity","Firewood","Solar",
                    "Kerosene","sawdust_or_paddy_husk","Other"])
                water_heating_method = st.selectbox("Water Heating Method", [
                    "Electricity (directly from the national grid)",
                    "Electricity (generated from solar energy system)",
                    "Gas","Firewood","Kerosene","Saw dust/paddy husk.",
                    "Coconut shells/charcoal","Other","Unknown"])

            submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

        if submitted:
            label, probability = predict_solar_adoption(
                area, socio_class, education, occupation,
                monthly_expenditure, household_size,
                house_ownership, built_year, type_of_house,
                floor_area, cooking_fuel, water_heating_method)

            # Confidence level
            if probability >= 0.75:
                confidence       = "🟢 High Confidence"
                confidence_color = "success"
            elif probability >= 0.45:
                confidence       = "🟡 Medium Confidence"
                confidence_color = "warning"
            else:
                confidence       = "🔴 Low Confidence"
                confidence_color = "error"

            st.divider()
            c1, c2, c3 = st.columns(3)

            with c1:
                if "✅" in label:
                    st.success(label)
                else:
                    st.error(label)

            with c2:
                st.metric("Probability of Solar Adoption", f"{probability*100:.2f}%")
                st.progress(float(probability))

            with c3:
                if confidence_color == "success":
                    st.success(confidence)
                elif confidence_color == "warning":
                    st.warning(confidence)
                else:
                    st.error(confidence)
                st.caption("🟢 High: ≥75%  |  🟡 Medium: 45–75%  |  🔴 Low: <45%")

    # ── Batch Prediction ──────────────────────────────────────────────────────
    with tab2:
        st.markdown("Upload an Excel file with household data to predict in bulk.")
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx","xls"])

        if uploaded_file:
            batch_df = pd.read_excel(uploaded_file)
            st.markdown("**Preview of uploaded data:**")
            st.dataframe(batch_df.head(), use_container_width=True)

            required_cols = [
                "monthly_expenditure","household_size","floor_area",
                "electricity_provider_csc_area","socio_economic_class",
                "education","occupation","house_ownership",
                "built_year_of_the_house","type_of_house",
                "cooking_fuel","water_heating_method"
            ]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                if st.button("🔍 Run Batch Predictions"):
                    with st.spinner("Running predictions..."):
                        input_encoded = pd.get_dummies(batch_df[required_cols])
                        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
                        scale_cols    = ["monthly_expenditure","floor_area","household_size"]
                        input_encoded[scale_cols] = scaler.transform(input_encoded[scale_cols])

                        predictions   = model.predict(input_encoded)
                        probabilities = model.predict_proba(input_encoded)[:, 1]

                        result_df = batch_df.copy()
                        result_df["Prediction"]    = ["✅ Likely Adopter" if p == 1 else "❌ Not Likely" for p in predictions]
                        result_df["Probability %"] = (probabilities * 100).round(2)

                        cols      = ["Prediction","Probability %"] + [c for c in result_df.columns if c not in ["Prediction","Probability %"]]
                        result_df = result_df[cols]
                        result_df = result_df.sort_values("Probability %", ascending=False).reset_index(drop=True)

                    total    = len(result_df)
                    adopters = (predictions == 1).sum()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Records",   total)
                    m2.metric("Likely Adopters", adopters)
                    m3.metric("Adoption Rate",   f"{adopters/total*100:.1f}%")

                    st.success(f"✅ Done! {total} predictions completed.")
                    st.dataframe(result_df, use_container_width=True)

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv,
                        file_name="solar_adoption_predictions.csv",
                        mime="text/csv"
                    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Influential Features
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔑 Influential Features":
    st.title("🔑 Most Influential Features")
    st.markdown("Top 10 features that most influence solar adoption prediction.")
    st.divider()

    # ── Feature importance chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_importance_df["Original_Feature"], top_importance_df["Importance"], color="#f4a261")
    ax.invert_yaxis()
    ax.set_title("Top 10 Most Influential Features")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Feature descriptions ──────────────────────────────────────────────────
    st.subheader("📖 Feature Descriptions")
    st.markdown("What each feature means and why it matters for solar adoption.")

    feature_descriptions = {
        "monthly_expenditure": {
            "icon": "💰",
            "label": "Monthly Expenditure",
            "desc": "Total monthly household spending in LKR. Higher spending households tend to have more disposable income, making solar installation more financially accessible.",
        },
        "floor_area": {
            "icon": "📐",
            "label": "Floor Area",
            "desc": "Total floor area of the house in square feet. Larger homes typically have more roof space available for solar panel installation.",
        },
        "household_size": {
            "icon": "👨‍👩‍👧‍👦",
            "label": "Household Size",
            "desc": "Number of people living in the household. Larger households consume more electricity, increasing the financial benefit of switching to solar energy.",
        },
        "socio_economic_class": {
            "icon": "📊",
            "label": "Socio-Economic Class",
            "desc": "Classified from SEC A (highest) to SEC E (lowest) based on income and lifestyle. SEC A households show a 30% adoption rate vs only 0.8% for SEC E.",
        },
        "electricity_provider_csc_area": {
            "icon": "📍",
            "label": "CSC Area",
            "desc": "The Ceylon Electricity Board Customer Service Centre area the household belongs to. Urban areas like Nugegoda and Boralasgamuwa show significantly higher adoption rates.",
        },
        "education": {
            "icon": "🎓",
            "label": "Education Level",
            "desc": "Highest educational qualification of the household head. More educated individuals tend to be more aware of solar energy benefits and long-term cost savings.",
        },
        "occupation": {
            "icon": "💼",
            "label": "Occupation",
            "desc": "Type of employment of the household head. Professionals and managers typically have higher income stability, making long-term solar investments more feasible.",
        },
        "house_ownership": {
            "icon": "🏠",
            "label": "House Ownership",
            "desc": "Whether the household owns or rents the property. Homeowners are far more likely to invest in solar panels since they benefit directly from long-term savings.",
        },
        "built_year_of_the_house": {
            "icon": "🏗️",
            "label": "Built Year of House",
            "desc": "The decade in which the house was constructed. Newer houses are more likely to be built with solar-compatible infrastructure and modern electrical systems.",
        },
        "type_of_house": {
            "icon": "🏡",
            "label": "Type of House",
            "desc": "The structural type of the dwelling. Double and multi-floor houses show 20–28% adoption rates compared to only 3.9% for single floor houses, likely due to larger roof areas.",
        },
        "cooking_fuel": {
            "icon": "🍳",
            "label": "Cooking Fuel",
            "desc": "Primary fuel used for cooking. Households already using modern energy sources like electricity or gas are more likely to adopt solar energy.",
        },
        "water_heating_method": {
            "icon": "🚿",
            "label": "Water Heating Method",
            "desc": "Method used to heat water. Households using solar water heating already show openness to renewable energy, making full solar adoption more likely.",
        },
    }

    # Display 2 cards per row
    features_list = list(feature_descriptions.items())
    for i in range(0, len(features_list), 2):
        col1, col2 = st.columns(2)
        for col, (feature, info) in zip([col1, col2], features_list[i:i+2]):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {info['icon']} {info['label']}")
                    st.caption(f"`{feature}`")
                    st.markdown(info["desc"])

    st.divider()
    st.subheader("📋 Full Feature Importance Table")
    st.dataframe(importance_df, use_container_width=True)
