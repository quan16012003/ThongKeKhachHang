import streamlit as st
import pandas as pd

# Đọc dữ liệu thống kê
df = pd.read_csv('outputs/statistics.csv')

# Hiển thị biểu đồ
st.title("Thống kê số lượng khách vào/ra cửa hàng")
st.line_chart(data=df.set_index('Time')['Person Count'], use_container_width=True)

# Hiển thị dữ liệu dạng bảng
st.dataframe(df)
