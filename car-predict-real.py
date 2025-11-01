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
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
        color: #222;
    }
    h1, h2, h3, h4 {
        color: #0a3d62;
    }
    .stButton>button {
        background-color: #0a3d62;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1e3799;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

st.write("A simple interactive demo using Streamlit + Scikit-learn with real data")

# -------------------------------
# 2. LOAD REAL DATA
# -------------------------------
@st.cache_data
def load_data(n=200):
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv"
    data = pd.read_csv(url)
    
    # Drop rows with missing Price or Horsepower
    data = data.dropna(subset=['Price', 'Horsepower'])
    
    # Simulate 'year' and 'mileage' since the dataset doesn't have them
    np.random.seed(42)
    data['year'] = np.random.randint(2000, 2023, len(data))
    data['mileage'] = np.random.randint(5000, 200000, len(data))
    
    # Rename columns to match your model
    data['horsepower'] = data['Horsepower']
    data['price'] = data['Price'] * 1000  # adjust units if needed
    
    return data[['year', 'mileage', 'horsepower', 'price']]

n = 210
df = load_data(n)
st.subheader("📊 Loaded Data")
st.dataframe(df.head(n), use_container_width=True)

# -------------------------------
# 3. INTERACTIVE VISUALIZATION
# -------------------------------
st.subheader("📈 Data Visualization with Filters")

st.sidebar.header("🔍 Filters")

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Select Year Range", year_min, year_max, (year_min, year_max))

hp_min, hp_max = int(df["horsepower"].min()), int(df["horsepower"].max())
hp_range = st.sidebar.slider("Select Horsepower Range", hp_min, hp_max, (hp_min, hp_max))

mileage_min, mileage_max = int(df["mileage"].min()), int(df["mileage"].max())
mileage_range = st.sidebar.slider("Select Mileage Range", mileage_min, mileage_max, (mileage_min, mileage_max), step=1000)

filtered_df = df[
    (df["year"].between(*year_range)) &
    (df["horsepower"].between(*hp_range)) &
    (df["mileage"].between(*mileage_range))
]

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

st.markdown("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(filtered_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 6}, cbar_kws={"shrink": 0.4})
st.pyplot(fig, use_container_width=True)

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
st.success(f"Mean Squared Error: **{mse:,.2f}**")

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
