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
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Car Price Prediction App")
st.write("A simple interactive demo using Streamlit + Scikit-learn")

# -------------------------------
# 2. LOAD SAMPLE DATA
# -------------------------------
@st.cache_data
def load_data(n=200):
    np.random.seed(42)
    data = pd.DataFrame({
        'year': np.random.randint(2000, 2023, n),
        'mileage': np.random.randint(5000, 200000, n),
        'horsepower': np.random.randint(80, 400, n)
    })
    # Synthetic target (car price)
    data['price'] = (
        30000
        - (2025 - data['year']) * 1000
        - data['mileage'] * 0.05
        + data['horsepower'] * 100
        + np.random.normal(0, 2000, n)
    )
    return data

n = 220
df = load_data(n)
st.subheader("📊 Sample Data")
st.dataframe(df.head(n))

# -------------------------------
# 3. INTERACTIVE VISUALIZATION
# -------------------------------
st.subheader("📈 Data Visualization with Filters")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Year slider
year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# Horsepower slider
hp_min, hp_max = int(df["horsepower"].min()), int(df["horsepower"].max())
hp_range = st.sidebar.slider(
    "Select Horsepower Range",
    min_value=hp_min,
    max_value=hp_max,
    value=(hp_min, hp_max)
)

# Mileage slider
mileage_min, mileage_max = int(df["mileage"].min()), int(df["mileage"].max())
mileage_range = st.sidebar.slider(
    "Select Mileage Range",
    min_value=mileage_min,
    max_value=mileage_max,
    value=(mileage_min, mileage_max),
    step=1000
)

# Apply all filters
filtered_df = df[
    (df["year"].between(*year_range)) &
    (df["horsepower"].between(*hp_range)) &
    (df["mileage"].between(*mileage_range))
]

st.write(f"Showing {len(filtered_df)} cars that match your filters.")

# Two side-by-side scatter plots
col1, col2 = st.columns(2)

with col1:
    st.write("**Horsepower vs Price**")
    fig, ax = plt.subplots()
    ax.scatter(filtered_df["horsepower"], filtered_df["price"], alpha=0.6)
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Price ($)")
    ax.set_title("Filtered by Year, Horsepower & Mileage")
    st.pyplot(fig)

with col2:
    st.write("**Mileage vs Price**")
    fig, ax = plt.subplots()
    ax.scatter(filtered_df["mileage"], filtered_df["price"], alpha=0.6, color="orange")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price ($)")
    ax.set_title("Filtered by Year, Horsepower & Mileage")
    st.pyplot(fig)

# Correlation heatmap
st.write("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize=(4, 3))
fig.tight_layout()
sns.heatmap(filtered_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 6},  cbar_kws={"shrink": 0.4})
st.pyplot(fig)

# -------------------------------
# 4. TRAIN MODEL
# -------------------------------
X = df[['year', 'mileage', 'horsepower']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

st.subheader("🧮 Model Performance")
st.write(f"Mean Squared Error: **{mse:,.2f}**")

# -------------------------------
# 5. USER INPUT & PREDICTION
# -------------------------------
st.subheader("🎯 Try Your Own Inputs")

col1, col2, col3 = st.columns(3)
with col1:
    year = st.slider("Year", 2000, 2023, 2015)
with col2:
    mileage = st.slider("Mileage", 5000, 200000, 50000, step=1000)
with col3:
    horsepower = st.slider("Horsepower", 80, 400, 150)

input_data = pd.DataFrame([[year, mileage, horsepower]], columns=["year", "mileage", "horsepower"])
prediction = model.predict(input_data)[0]

st.success(f"💰 Estimated Car Price: **${prediction:,.2f}**")

# -------------------------------
# 6. FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Seaborn and Scikit-learn.")
