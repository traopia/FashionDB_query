import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Streamlit Environment Test")

st.header("Basic Streamlit Test")
st.write("✅ Streamlit is working!")

# Test pandas
st.header("Pandas Test")
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
st.dataframe(df.head())

# Test matplotlib
st.header("Matplotlib Test")
fig, ax = plt.subplots()
ax.scatter(df['x'], df['y'])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Random Scatter Plot')
st.pyplot(fig)

# Test seaborn
st.header("Seaborn Test")
fig, ax = plt.subplots()
sns.histplot(df['x'], ax=ax)
ax.set_title('Histogram of X values')
st.pyplot(fig)

# Test plotly
st.header("Plotly Test")
fig = px.scatter(df, x='x', y='y', title='Plotly Scatter Plot')
st.plotly_chart(fig)

# Test scikit-learn
st.header("Scikit-learn Test")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[['x']], df['y'])
st.write(f"Linear regression coefficient: {model.coef_[0]:.4f}")

st.success("🎉 All packages are working correctly!") 