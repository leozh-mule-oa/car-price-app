import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="🚗 Car Price Predictor", layout="wide", page_icon="🚘")
st.title("🚗 Car Price Prediction App")
st.write("A simple interactive demo using Streamlit + Scikit-learn with real data")

# -------------------------------
# 2. LOAD AND IMPROVE DATA
# -------------------------------
@st.cache_data
def load_data(n=200):
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv"
    data = pd.read_csv(url)
    data = data.dropna(subset=['Price', 'Horsepower'])
    np.random.seed(42)
    data['year'] = np.random.randint(2005, 2023, len(data))
    data['mileage'] = (2023 - data['year']) * np.random.randint(8000, 15000, len(data))
    data['mileage'] = np.clip(data['mileage'], 5000, 250000)
    data['horsepower'] = data['Horsepower']
    data['price'] = data['Price'] * 1000
    return data[['year', 'mileage', 'horsepower', 'price']]

n = 210
df = load_data(n)
st.subheader("📊 Loaded Data")
st.dataframe(df.head(n), use_container_width=True)

# -------------------------------
# 3. VISUALIZATION WITH FILTERS
# -------------------------------
st.subheader("📈 Data Visualization with Filters")
st.sidebar.header("🔍 Filters")

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Select Year Range", year_min, year_max, (year_min, year_max))

hp_min, hp_max = int(df["horsepower"].min()), int(df["horsepower"].max())
hp_range = st.sidebar.slider("Select Horsepower Range", hp_min, hp_max, (hp_min, hp_max))

mileage_min, mileage_max = int(df["mileage"].min()), int(df["mileage"].max())
mileage_range = st.sidebar.slider("Select Mileage Range", mileage_min, mileage_max, (mileage_min, mileage_max), step=1000)

filtered_df = df[(df["year"].between(*year_range)) & (df["horsepower"].between(*hp_range)) & (df["mileage"].between(*mileage_range))]
st.write(f"Showing {len(filtered_df)} cars that match your filters.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Horsepower vs Price**")
    fig, ax = plt.subplots()
    ax.scatter(filtered_df["horsepower"], filtered_df["price"], alpha=0.6, color="#1e90ff")
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Price ($)")
    ax.set_title("Filtered by Year, Horsepower & Mileage")
    st.pyplot(fig, use_container_width=True)
with col2:
    st.markdown("**Mileage vs Price**")
    fig, ax = plt.subplots()
    ax.scatter(filtered_df["mileage"], filtered_df["price"], alpha=0.6, color="#ffa502")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price ($)")
    ax.set_title("Filtered by Year, Horsepower & Mileage")
    st.pyplot(fig, use_container_width=True)

# -------------------------------
# Correlation Heatmap (much smaller)
# -------------------------------
st.markdown("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize=(4.5, 3))  # slightly larger than before
sns.heatmap(
    filtered_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    ax=ax,
    annot_kws={"size": 7},  # slightly bigger font
    cbar_kws={"shrink": 0.2}
)
fig.tight_layout(pad=0.5)
st.pyplot(fig, use_container_width=False)


# -------------------------------
# 4. MODEL SELECTION & TRAINING (styled)
# -------------------------------
st.markdown("""
<div style='background-color:#f0f8ff; padding:15px; border-radius:10px;'>
<h3 style='color:#1e90ff; text-align:center;'>⚙️ Select a Model to Train</h3>
</div>
""", unsafe_allow_html=True)

model_name = st.radio("", ("Linear Regression", "Random Forest Regressor"), index=0, horizontal=True)

if model_name == "Random Forest Regressor":
    from sklearn.ensemble import RandomForestRegressor
    n_estimators = st.slider("Number of Trees (n_estimators):", 50, 500, 100, 50)
    max_depth = st.slider("Max Depth:", 2, 20, 6)

X = df[['year', 'mileage', 'horsepower']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_name == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

model.fit(X_train, y_train)
preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

st.subheader("🧮 Model Performance")
st.success(f"Root Mean Squared Error (RMSE): **${rmse:,.2f}**")

# -------------------------------
# 4.5 VISUALIZE MODEL PERFORMANCE
# -------------------------------
st.subheader("📉 Model Diagnostics")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Predicted vs Actual Prices**")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_test, preds, alpha=0.6, color="#2ed573")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title(f"{model_name} Fit")
    st.pyplot(fig, use_container_width=True)
with col2:
    st.markdown("**Residuals Distribution**")
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(residuals, bins=20, kde=True, color="#1e90ff", ax=ax)
    ax.set_xlabel("Prediction Error (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Histogram")
    st.pyplot(fig, use_container_width=True)

st.markdown("""
✅ **Interpretation:**
- Scatter points near the red line → good fit.
- Residuals centered around 0 → unbiased.
- Skewed residuals → model under/overestimates some cars.
""")

# -------------------------------
# 5. USER INPUT & PREDICTION
# -------------------------------
st.subheader("🎯 Try Your Own Inputs")
col1, col2, col3 = st.columns(3)
with col1:
    year = st.slider("Year", year_min, year_max, int(df["year"].median()))
with col2:
    mileage = st.slider("Mileage", mileage_min, mileage_max, int(df["mileage"].median()), step=1000)
with col3:
    horsepower = st.slider("Horsepower", hp_min, hp_max, int(df["horsepower"].median()))

input_data = pd.DataFrame([[year, mileage, horsepower]], columns=["year", "mileage", "horsepower"])
prediction = model.predict(input_data)[0]
st.success(f"💰 Estimated Car Price: **${prediction:,.2f}**")

# -------------------------------
# 6. FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Seaborn and Scikit-learn with real car data.")