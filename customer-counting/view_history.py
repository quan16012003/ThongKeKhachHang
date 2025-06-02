import streamlit as st
import pandas as pd
import os

def view_statistics():
    st.title("Lịch sử ra/vào")
    try:
        history = pd.read_csv('outputs/statistics.csv')
        st.dataframe(history)

        total_entry = history['Entry Count'].sum()
        total_exit = history['Exit Count'].sum()
        st.write(f"**Tổng số người vào:** {total_entry}")
        st.write(f"**Tổng số người ra:** {total_exit}")
    except FileNotFoundError:
        st.error("Không tìm thấy file lịch sử. Hãy chạy ghi dữ liệu trước!")

if __name__ == "__main__":
    view_statistics()
