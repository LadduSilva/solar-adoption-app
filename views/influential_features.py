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
        "type_of_house": {
            "icon": "🏡",
            "label": "Type of House",
            "desc": "The most influential feature. The structural type of the dwelling strongly predicts solar adoption. Double and multi-floor houses show 20–28% adoption rates compared to only 3.9% for single floor houses, likely due to larger roof areas and higher income levels of residents.",
        },
        "occupation": {
            "icon": "💼",
            "label": "Occupation",
            "desc": "The second most influential feature. Type of employment determines income stability and financial capacity. Professionals and managers typically have the financial means to invest in solar panels and benefit from long-term savings.",
        },
        "socio_economic_class": {
            "icon": "📊",
            "label": "Socio-Economic Class",
            "desc": "Classified from SEC A (highest) to SEC E (lowest). SEC A households show a 30.4% adoption rate vs only 0.8% for SEC E — a 38x difference — making this one of the strongest indicators of solar adoption likelihood.",
        },
        "floor_area": {
            "icon": "📐",
            "label": "Floor Area",
            "desc": "Total floor area of the house in square feet. Larger homes typically have more roof space available for solar panel installation and tend to consume more electricity, making solar energy more cost-effective.",
        },
        "education": {
            "icon": "🎓",
            "label": "Education Level",
            "desc": "Highest educational qualification of the household head. More educated individuals are more aware of solar energy benefits, long-term cost savings, and available government incentives for renewable energy adoption.",
        },
        "electricity_provider_csc_area": {
            "icon": "📍",
            "label": "CSC Area",
            "desc": "The Ceylon Electricity Board Customer Service Centre area. Urban areas like Nugegoda (31.2%) and Boralasgamuwa (24.4%) show significantly higher adoption rates compared to rural areas, reflecting income and infrastructure differences.",
        },
        "water_heating_method": {
            "icon": "🚿",
            "label": "Water Heating Method",
            "desc": "Method used to heat water. Households already using solar water heating show strong openness to renewable energy, making full solar adoption significantly more likely. It reflects both financial capacity and environmental awareness.",
        },
        "built_year_of_the_house": {
            "icon": "🏗️",
            "label": "Built Year of House",
            "desc": "The decade in which the house was constructed. Newer houses are more likely to be built with solar-compatible electrical infrastructure. Older houses may require costly upgrades before solar installation is feasible.",
        },
        "monthly_expenditure": {
            "icon": "💰",
            "label": "Monthly Expenditure",
            "desc": "Total monthly household spending in LKR. Higher spending reflects greater disposable income, making the upfront cost of solar installation more financially accessible. It also indicates higher electricity consumption, increasing the ROI of solar energy.",
        },
        "household_size": {
            "icon": "👨‍👩‍👧‍👦",
            "label": "Household Size",
            "desc": "Number of people living in the household. Larger households consume more electricity, increasing the financial benefit of switching to solar. However, larger families may also have competing financial priorities.",
        },
        "cooking_fuel": {
            "icon": "🍳",
            "label": "Cooking Fuel",
            "desc": "Primary fuel used for cooking. Households already using modern energy sources like electricity or gas show greater openness to adopting solar energy compared to those relying on traditional fuels like firewood.",
        },
        "house_ownership": {
            "icon": "🏠",
            "label": "House Ownership",
            "desc": "The least influential feature but still relevant. Homeowners are more likely to invest in solar panels since they directly benefit from long-term savings. Renters have little incentive as they don't own the property.",
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