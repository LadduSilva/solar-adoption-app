import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show():
    @st.cache_data
    def load_raw_data():
        return pd.read_excel("solar_adoption_data_set.xlsx")

    raw_df        = load_raw_data()
    total         = len(raw_df)
    adopters      = (raw_df["solar_generation"] == "Yes").sum()
    non_adopters  = total - adopters
    adoption_rate = adopters / total * 100

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
