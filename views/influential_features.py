import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show():

    @st.cache_data
    def load_data():
        importance_df = pd.read_csv("Research/grouped_feature_importance.csv")
        return importance_df

    importance_df       = load_data()
    top_importance_df   = importance_df.head(10)

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