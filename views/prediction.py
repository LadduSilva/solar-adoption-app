import streamlit as st
import pandas as pd
import joblib

def show():
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

        probability = model.predict_proba(input_encoded)[0][1]
        prediction  = 1 if probability >= threshold else 0
        label = "✅ Likely Solar Adopter" if prediction == 1 else "❌ Not Likely to Adopt Solar"
        return label, probability

    @st.cache_resource
    def load_artifacts():
        model         = joblib.load("Research/lr_model.pkl")
        scaler        = joblib.load("Research/scaler.pkl")
        model_columns = joblib.load("Research/model_columns.pkl")
        threshold     = joblib.load("Research/lr_threshold.pkl")
        return model, scaler, model_columns, threshold

    model, scaler, model_columns, threshold = load_artifacts()

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
                socio_class = st.selectbox("Socio Economic Class", ["SEC A","SEC B","SEC C","SEC D","SEC E"], index=2)
                education = st.selectbox("Education", [
                    "Graduate / Post-Grads / Degree level professional qualification",
                    "Diploma with O/L or A/L (Non graduate)",
                    "Other professional certificates with O/L or A/L / Part qualification (Non graduate)",
                    "O/L or A/L pending / Passed","Schooling up to Grade 6 - 9",
                    "Primary Education","Illiterate"], index=3)
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
                    "Farmer owning - Over 5 acres / Landed proprietor"], index=0)
                monthly_expenditure = st.number_input("Monthly Expenditure (LKR)", min_value=0.0, value=50000.0, step=100.0)
                household_size      = st.number_input("Household Size", min_value=1, value=4, step=1)

            with col2:
                house_ownership = st.selectbox("House Ownership", [
                    "Yes, I or a household member owns it.",
                    "No, I am living on rent and the rent is paid by me or a household member.",
                    "No, I am living on rent and the rent is paid by the employer.",
                    "No, I or any household member does not own or rent this household. We occupy this household without any payment of rent."
                ], index=0)
                built_year = st.selectbox("Built Year of the House", [
                    "Before 1980","1980-1989","1990-1999","2000-2009",
                    "2010-2019","In 2020 or After 2020","Don't know"], index=3)
                type_of_house = st.selectbox("Type of House", [
                    "Single House - Single Floor","Single House - Double Floor",
                    "Single House - More than 2 floors","Flat",
                    "Attached house / Annex","Twin Houses","Slum / Shanty",
                    "Line room/row house","Condominium/ Luxury apartments","Other"], index=0)
                floor_area = st.number_input("Floor Area (sq ft)", min_value=0.0, value=1200.0, step=10.0)
                cooking_fuel = st.selectbox("Cooking Fuel", [
                    "Gas","Electricity","Firewood","Solar",
                    "Kerosene","sawdust_or_paddy_husk","Other"], index=0)
                water_heating_method = st.selectbox("Water Heating Method", [
                    "Electricity (directly from the national grid)",
                    "Electricity (generated from solar energy system)",
                    "Gas","Firewood","Kerosene","Saw dust/paddy husk.",
                    "Coconut shells/charcoal","Other","Unknown"], index=0)

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

                        probabilities = model.predict_proba(input_encoded)[:, 1]
                        predictions   = (probabilities >= threshold).astype(int)

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
