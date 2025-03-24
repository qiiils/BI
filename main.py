import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="Supermarket Sales Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add title and description
st.title("🛒 Supermarket Sales Dashboard")
st.markdown("""
This dashboard visualizes supermarket sales data to identify patterns, trends, and insights for business intelligence.
Please upload your CSV file to start the analysis.
""")

# Function to process data
def process_data(df):
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract month and year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Year'] = df['Date'].dt.year
    
    # Extract time components
    # Check if Time column exists and has the correct format
    if 'Time' in df.columns:
        try:
            df['Hour'] = pd.to_datetime(df['Time']).dt.hour
            
            # Create time of day category
            time_conditions = [
                (df['Hour'] >= 6) & (df['Hour'] < 12),
                (df['Hour'] >= 12) & (df['Hour'] < 18),
                (df['Hour'] >= 18) & (df['Hour'] < 24)
            ]
            time_categories = ['Morning', 'Afternoon', 'Evening']
            df['Time_of_Day'] = np.select(time_conditions, time_categories, default='Night')
        except:
            st.warning("Time column format not recognized. Time-based analysis will be limited.")
            df['Hour'] = 12  # Default value
            df['Time_of_Day'] = 'Unknown'
    
    return df

# File upload section
uploaded_file = st.file_uploader("Upload your supermarket sales CSV file", type=["csv"])

# Check if file is uploaded
if uploaded_file is not None:
    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded and processed!")
        
        # Show raw data sample in an expander
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head())
            st.text(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
            st.text(f"Columns: {', '.join(df.columns)}")
        
        # Process the data
        df = process_data(df)
        
        # Main sections using tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Detailed Analysis", "📈 Trends", "🗂 Data Warehouse"])

        with tab1:
            st.header("Sales Overview")
            
            # Check required columns for KPIs
            required_cols = {
                'Total': 'total sales', 
                'Rating': 'average rating', 
                'gross margin percentage': 'gross margin'
            }
            
            missing_cols = [col for col, desc in required_cols.items() if col not in df.columns]
            
            if missing_cols:
                st.warning(f"Missing columns for some KPIs: {', '.join(missing_cols)}. Some visualizations may be limited.")
            
            # Top KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'Total' in df.columns:
                    total_sales = df['Total'].sum()
                    st.metric("Total Sales", f"${total_sales:,.2f}")
                else:
                    st.metric("Total Sales", "N/A")
                    
            with col2:
                if 'Rating' in df.columns:
                    avg_rating = df['Rating'].mean()
                    st.metric("Average Rating", f"{avg_rating:.2f}/10")
                else:
                    st.metric("Average Rating", "N/A")
                    
            with col3:
                total_customers = df.shape[0]
                st.metric("Total Customers", f"{total_customers:,}")
                
            with col4:
                if 'gross margin percentage' in df.columns:
                    gross_margin_pct = df['gross margin percentage'].mean()
                    st.metric("Avg. Gross Margin", f"{gross_margin_pct:.2f}%")
                else:
                    st.metric("Avg. Gross Margin", "N/A")
            
            # Check for product line column
            if 'Product line' in df.columns and 'Total' in df.columns:
                # Sales by Category and Branch
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sales by Product Line")
                    product_sales = df.groupby('Product line')['Total'].sum().reset_index()
                    fig = px.bar(
                        product_sales, 
                        x='Product line', 
                        y='Total',
                        title='Total Sales by Product Line',
                        color='Product line',
                        labels={'Total': 'Total Sales ($)', 'Product line': 'Product Category'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Branch' in df.columns and 'City' in df.columns:
                        st.subheader("Sales by Branch")
                        branch_sales = df.groupby(['Branch', 'City'])['Total'].sum().reset_index()
                        fig = px.pie(
                            branch_sales, 
                            values='Total', 
                            names='City',
                            title='Sales Distribution by Branch',
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif 'Branch' in df.columns:
                        st.subheader("Sales by Branch")
                        branch_sales = df.groupby('Branch')['Total'].sum().reset_index()
                        fig = px.pie(
                            branch_sales, 
                            values='Total', 
                            names='Branch',
                            title='Sales Distribution by Branch',
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Branch information not available in the dataset.")
            else:
                st.warning("Product line or Total column missing. Some visualizations cannot be displayed.")
            
            # Customer Demographics
            if 'Gender' in df.columns and 'Customer type' in df.columns and 'Total' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sales by Gender")
                    gender_sales = df.groupby('Gender')['Total'].sum().reset_index()
                    fig = px.pie(
                        gender_sales, 
                        values='Total', 
                        names='Gender',
                        title='Sales by Gender',
                        color='Gender',
                        color_discrete_map={'Male': '#1F77B4', 'Female': '#FF7F0E'},
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Sales by Customer Type")
                    customer_sales = df.groupby('Customer type')['Total'].sum().reset_index()
                    fig = px.pie(
                        customer_sales, 
                        values='Total', 
                        names='Customer type',
                        title='Sales by Customer Type',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Gender or Customer type columns missing. Customer demographics cannot be displayed.")
            
            # Payment Method Distribution
            if 'Payment' in df.columns:
                st.subheader("Payment Method Distribution")
                payment_count = df['Payment'].value_counts().reset_index()
                payment_count.columns = ['Payment Method', 'Count']
                
                fig = px.bar(
                    payment_count, 
                    x='Payment Method', 
                    y='Count',
                    title='Transactions by Payment Method',
                    color='Payment Method'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Payment column missing. Payment method distribution cannot be displayed.")

        with tab2:
            st.header("Detailed Analysis")
            
            # Filter section
            filters = []
            filter_cols = []
            
            if 'Branch' in df.columns:
                filter_cols.append('Branch')
                branch_filter = st.multiselect(
                    "Select Branch",
                    options=df['Branch'].unique(),
                    default=df['Branch'].unique()
                )
                if branch_filter:
                    filters.append(df['Branch'].isin(branch_filter))
            
            if 'Product line' in df.columns:
                filter_cols.append('Product line')
                product_filter = st.multiselect(
                    "Select Product Line",
                    options=df['Product line'].unique(),
                    default=df['Product line'].unique()
                )
                if product_filter:
                    filters.append(df['Product line'].isin(product_filter))
            
            if 'Customer type' in df.columns:
                filter_cols.append('Customer type')
                customer_filter = st.multiselect(
                    "Select Customer Type",
                    options=df['Customer type'].unique(),
                    default=df['Customer type'].unique()
                )
                if customer_filter:
                    filters.append(df['Customer type'].isin(customer_filter))
            
            # Apply filters
            if filters:
                filtered_df = df[np.logical_and.reduce(filters)]
            else:
                filtered_df = df
            
            # Show filter summary
            if filter_cols:
                st.info(f"Data filtered by: {', '.join(filter_cols)}")
                st.text(f"Showing {filtered_df.shape[0]} out of {df.shape[0]} records")
            
            # Show filtered statistics
            st.subheader("Filtered Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'Total' in filtered_df.columns:
                    filtered_sales = filtered_df['Total'].sum()
                    st.metric("Total Sales", f"${filtered_sales:,.2f}")
                else:
                    st.metric("Total Sales", "N/A")
                    
            with col2:
                if 'Rating' in filtered_df.columns:
                    filtered_avg_rating = filtered_df['Rating'].mean()
                    st.metric("Average Rating", f"{filtered_avg_rating:.2f}/10")
                else:
                    st.metric("Average Rating", "N/A")
                    
            with col3:
                filtered_customers = filtered_df.shape[0]
                st.metric("Total Customers", f"{filtered_customers:,}")
                
            with col4:
                if 'gross margin percentage' in filtered_df.columns:
                    filtered_margin = filtered_df['gross margin percentage'].mean()
                    st.metric("Avg. Gross Margin", f"{filtered_margin:.2f}%")
                else:
                    st.metric("Avg. Gross Margin", "N/A")
            
            # Correlation heatmap
            st.subheader("Correlation Analysis")
            
            # Select only numeric columns
            numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
            
            if numeric_df.shape[1] > 1:  # Need at least 2 numeric columns for correlation
                # Remove irrelevant columns for correlation
                if 'Invoice ID' in numeric_df.columns:
                    numeric_df = numeric_df.drop(['Invoice ID'], axis=1)
                
                # Calculate correlation matrix
                corr = numeric_df.corr()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                plt.title('Correlation Matrix of Numeric Variables')
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric columns for correlation analysis.")
            
            # Product popularity vs profitability
            if all(col in filtered_df.columns for col in ['Product line', 'Quantity', 'gross income', 'Rating']):
                st.subheader("Product Line: Popularity vs. Profitability")
                
                product_analysis = filtered_df.groupby('Product line').agg({
                    'Quantity': 'sum',
                    'gross income': 'sum',
                    'Rating': 'mean'
                }).reset_index()
                
                fig = px.scatter(
                    product_analysis,
                    x='Quantity',
                    y='gross income',
                    size='gross income',
                    color='Rating',
                    hover_name='Product line',
                    text='Product line',
                    title='Product Line: Quantity Sold vs. Gross Income',
                    labels={'Quantity': 'Total Quantity Sold', 'gross income': 'Gross Income ($)', 'Rating': 'Average Rating'}
                )
                
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Required columns for product analysis not available.")
            
            # Time of day analysis
            if 'Time_of_Day' in filtered_df.columns and 'Total' in filtered_df.columns:
                st.subheader("Sales by Time of Day")
                
                time_analysis = filtered_df.groupby('Time_of_Day')['Total'].agg(['sum', 'count']).reset_index()
                time_analysis.columns = ['Time of Day', 'Total Sales', 'Transaction Count']
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(x=time_analysis['Time of Day'], y=time_analysis['Total Sales'], name="Total Sales"),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=time_analysis['Time of Day'], y=time_analysis['Transaction Count'], name="Transaction Count", mode='lines+markers'),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title_text="Sales and Transaction Count by Time of Day",
                    xaxis_title="Time of Day"
                )
                
                fig.update_yaxes(title_text="Total Sales ($)", secondary_y=False)
                fig.update_yaxes(title_text="Transaction Count", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Time of Day analysis not available due to missing columns.")

        with tab3:
            st.header("Sales Trends")
            
            # Daily sales trend
            if 'Date' in filtered_df.columns and 'Total' in filtered_df.columns:
                st.subheader("Daily Sales Trend")
                
                daily_sales = filtered_df.groupby('Date')['Total'].sum().reset_index()
                
                fig = px.line(
                    daily_sales,
                    x='Date',
                    y='Total',
                    title='Daily Sales Over Time',
                    labels={'Total': 'Total Sales ($)', 'Date': 'Date'}
                )
                
                fig.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Daily sales trend cannot be displayed due to missing columns.")
            
            # Monthly trends
            if all(col in filtered_df.columns for col in ['Year', 'Month', 'Month_Name', 'Total']):
                st.subheader("Monthly Sales and Customer Analysis")
                
                monthly_data = filtered_df.groupby(['Year', 'Month', 'Month_Name']).agg({
                    'Total': 'sum',
                    'Invoice ID': 'count',
                    'Rating': 'mean' if 'Rating' in filtered_df.columns else 'count'
                }).reset_index()
                
                monthly_data['YearMonth'] = monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str).str.zfill(2)
                monthly_data = monthly_data.sort_values('YearMonth')
                
                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add traces
                fig.add_trace(
                    go.Bar(x=monthly_data['Month_Name'], y=monthly_data['Total'], name="Total Sales"),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=monthly_data['Month_Name'], y=monthly_data['Invoice ID'], name="Transaction Count", mode='lines+markers'),
                    secondary_y=True
                )
                
                if 'Rating' in filtered_df.columns:
                    fig.add_trace(
                        go.Scatter(x=monthly_data['Month_Name'], y=monthly_data['Rating'], name="Average Rating", mode='lines+markers', line=dict(dash='dash')),
                        secondary_y=True
                    )
                
                # Set titles
                fig.update_layout(
                    title_text="Monthly Sales, Transactions, and Ratings",
                    xaxis_title="Month"
                )
                
                fig.update_yaxes(title_text="Total Sales ($)", secondary_y=False)
                fig.update_yaxes(title_text="Count / Rating", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Monthly sales analysis cannot be displayed due to missing columns.")
            
            # Product line trends
            if all(col in filtered_df.columns for col in ['Month_Name', 'Product line', 'Total']):
                st.subheader("Product Line Trends")
                
                product_monthly = filtered_df.groupby(['Month_Name', 'Product line'])['Total'].sum().reset_index()
                
                fig = px.line(
                    product_monthly,
                    x='Month_Name',
                    y='Total',
                    color='Product line',
                    title='Monthly Sales by Product Line',
                    labels={'Total': 'Total Sales ($)', 'Month_Name': 'Month', 'Product line': 'Product Category'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Product line trends cannot be displayed due to missing columns.")
            
            # Customer type trends
            if all(col in filtered_df.columns for col in ['Month_Name', 'Customer type', 'Total']):
                st.subheader("Customer Type Trends")
                
                customer_monthly = filtered_df.groupby(['Month_Name', 'Customer type'])['Total'].sum().reset_index()
                
                fig = px.line(
                    customer_monthly,
                    x='Month_Name',
                    y='Total',
                    color='Customer type',
                    title='Monthly Sales by Customer Type',
                    labels={'Total': 'Total Sales ($)', 'Month_Name': 'Month', 'Customer type': 'Customer Type'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Customer type trends cannot be displayed due to missing columns.")

        with tab4:
            st.header("Data Warehouse Design")
            
            st.subheader("ETL Process")
            st.markdown("""
            ### Extract, Transform, Load (ETL) Process
            
            *Extract:*
            - Data is sourced from POS systems across different branches
            - Additional customer demographic data is collected through loyalty programs
            - Transaction details are captured including time, product, and payment method
            
            *Transform:*
            - Date and time formatted into appropriate datetime formats
            - Categorization of transactions by time of day
            - Customer segmentation based on purchase behavior
            - Product hierarchical classification
            - Currency normalization across branches
            
            *Load:*
            - Data loaded into a star schema data warehouse
            - Regular incremental loads scheduled daily
            - Full refresh performed monthly
            """)
            
            st.subheader("Data Warehouse Schema")
            
            # Create a simple star schema diagram
            schema_code = """
            graph TD
                F["Fact_Sales"] --> D1["Dim_Date"]
                F --> D2["Dim_Product"]
                F --> D3["Dim_Customer"]
                F --> D4["Dim_Branch"]
                F --> D5["Dim_Payment"]
                
                F[Fact_Sales<br>sales_id<br>date_id<br>product_id<br>customer_id<br>branch_id<br>payment_id<br>quantity<br>unit_price<br>tax<br>total<br>cogs<br>gross_income<br>rating]
                
                D1[Dim_Date<br>date_id<br>date<br>day<br>month<br>year<br>quarter<br>day_of_week<br>is_weekend<br>time<br>hour<br>time_of_day]
                
                D2[Dim_Product<br>product_id<br>product_line<br>unit_price<br>cost<br>gross_margin_pct<br>category<br>subcategory]
                
                D3[Dim_Customer<br>customer_id<br>customer_type<br>gender<br>loyalty_status]
                
                D4[Dim_Branch<br>branch_id<br>branch<br>city<br>location]
                
                D5[Dim_Payment<br>payment_id<br>payment_method]
            """
            
            st.markdown("### Star Schema Design")
            st.graphviz_chart(schema_code)
            
            st.markdown("### Dimensions and Fact Table Details")
            
            st.info("""
            *Fact_Sales:* Contains all sales transactions with foreign keys to all dimension tables and measures like quantity, price, etc.
            
            *Dim_Date:* Time dimension with various time hierarchies and classifications.
            
            *Dim_Product:* Product hierarchy with categorization and pricing information.
            
            *Dim_Customer:* Customer details and segmentation.
            
            *Dim_Branch:* Store location and geographical hierarchy.
            
            *Dim_Payment:* Payment method details.
            """)
            
            st.subheader("Dashboard Design Considerations")
            st.markdown("""
            Based on the data warehouse schema, the following visualizations and analyses have been implemented:
            
            1. *Sales Overview:* High-level KPIs and distribution across key dimensions
            2. *Detailed Analysis:* Correlation analysis and cross-dimensional relationships
            3. *Trends:* Time-based analysis across various dimensions
            
            Additional dashboards that could be developed:
            
            1. *Customer Segmentation Dashboard:* Deep dive into customer behavior and loyalty
            2. *Inventory Management Dashboard:* Product performance and stock optimization
            3. *Branch Performance Dashboard:* Comparative analysis across locations
            4. *Forecasting Dashboard:* Sales predictions based on historical patterns
            """)

        # Sidebar with additional information
        st.sidebar.header("About This Dashboard")
        st.sidebar.info("""
        This dashboard visualizes supermarket sales data to provide insights for business intelligence.
        The analysis follows the format requirements for:
        - Data Analysis
        - Information Needs Identification
        - ETL Design
        - Data Warehouse Schema
        - Dashboard Visualization
        """)

        # Footer
        st.markdown("---")
        st.markdown("© 2025 Supermarket Sales Analysis Dashboard")
        
    except Exception as e:
        st.error(f"Error processing the CSV file: {e}")
        st.info("Please make sure your CSV file has the necessary columns for supermarket sales analysis (e.g., 'Date', 'Total', 'Product line', etc.)")
else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload your supermarket sales CSV file to begin the analysis.")
    
    # Example of expected columns
    st.subheader("Expected CSV Format")
    st.markdown("""
    Your CSV file should preferably include these columns for full functionality:
    - Invoice ID
    - Branch
    - City
    - Customer type
    - Gender
    - Product line
    - Unit price
    - Quantity
    - Tax 5%
    - Total
    - Date
    - Time
    - Payment
    - cogs
    - gross margin percentage
    - gross income
    - Rating
    
    The dashboard will adapt to your data even if some columns are missing, but functionality may be limited.
    """)
    
    # Show sample data structure
    sample_data = {
        'Invoice ID': ['750-67-8428', '226-31-3081'],
        'Branch': ['A', 'C'],
        'City': ['Yangon', 'Naypyitaw'],
        'Customer type': ['Member', 'Normal'],
        'Gender': ['Female', 'Male'],
        'Product line': ['Health and beauty', 'Electronic accessories'],
        'Unit price': [74.69, 15.28],
        'Quantity': [7, 5],
        'Tax 5%': [26.1415, 3.82],
        'Total': [548.9715, 80.22],
        'Date': ['1/5/2019', '3/8/2019'],
        'Time': ['13:08', '10:29'],
        'Payment': ['Ewallet', 'Cash'],
        'cogs': [522.83, 76.4],
        'gross margin percentage': [4.761904762, 4.761904762],
        'gross income': [26.1415, 3.82],
        'Rating': [9.1, 6.2]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)
