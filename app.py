import streamlit as st

st.set_page_config(page_title="Test App")

st.title("✅ This app is working!")
st.write("If you're seeing this message, Streamlit Cloud is working.")







# import streamlit as st

# st.set_page_config(
#     page_title="Super Store Dashboard",
#     page_icon="🛒",
#     layout="wide"
# )

# # Main Title
# st.markdown("<h1 style='color:#00FFFF;'>🛒 Super Store Sales Dashboard</h1>", unsafe_allow_html=True)
# st.markdown("Explore visualizations and predict sales using our ML model. Use the sidebar to navigate.")

# # Info Box
# st.info("🔍 **Explore**: Visualize sales trends and customer behavior. 🧠 **Predict**: Estimate future sales using machine learning.")

# # KPI Cards
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric(label="Total Orders", value="7,542")
# with col2:
#     st.metric(label="Total Revenue", value="₹12.6M")
# with col3:
#     st.metric(label="Active Customers", value="3,210")

# # Footer
# st.markdown("---")
# st.markdown("🔧 Built with Streamlit  \n👨‍💻 Developed by **Surya Rao**  \n🕒 Updated: June 2025")
