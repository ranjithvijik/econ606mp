
def render_custom_analysis_workbench(
    macro_data: pd.DataFrame, 
    tariff_data: pd.DataFrame, 
    treasury_data: pd.DataFrame, 
    debt_data: pd.DataFrame,
    yield_data: pd.DataFrame
):
    """Render Custom Data Analysis Workbench."""
    st.markdown("### 🛠️ Custom Data Workbench")
    
    st.markdown("""
    <div class="info-box">
    <strong>Data Explorer:</strong> Select datasets, filter time ranges, and perform custom correlation 
    analysis. Export your findings for external research.
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Dataset Selection
    st.markdown("#### 1. Select Data Source")
    dataset_options = {
        "Macroeconomic Indicators": macro_data,
        "Tariff Data": tariff_data,
        "Treasury Holdings": treasury_data,
        "Federal Debt": debt_data,
        "Yield Suppression": yield_data
    }
    
    selected_dataset_name = st.selectbox("Choose Dataset:", list(dataset_options.keys()))
    df = dataset_options[selected_dataset_name].copy()
    
    # Ensure Date column is accessible or index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    # 2. Time Filter
    st.markdown("#### 2. Filter Time Range")
    
    # Identify date column (usually 'Year' or 'Date')
    date_col = next((col for col in df.columns if 'year' in col.lower() or 'date' in col.lower()), None)
    
    if date_col:
        # Check if column is datetime or numeric
        if pd.api.types.is_numeric_dtype(df[date_col]):
            min_date = int(df[date_col].min())
            max_date = int(df[date_col].max())
            
            start_year, end_year = st.slider(
                "Select Year Range:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date)
            )
            df = df[(df[date_col] >= start_year) & (df[date_col] <= end_year)]
            
        elif pd.api.types.is_datetime64_any_dtype(df[date_col]):
            min_year = df[date_col].min().year
            max_year = df[date_col].max().year
            
            start_year, end_year = st.slider(
                "Select Year Range:",
                min_value=int(min_year),
                max_value=int(max_year),
                value=(int(min_year), int(max_year))
            )
            df = df[(df[date_col].dt.year >= start_year) & (df[date_col].dt.year <= end_year)]
    
    # 3. Column Selection
    st.markdown("#### 3. Select Variables")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove year/date from analyzable columns if possible to avoid correlation with time unless desired
    numeric_cols = [c for c in numeric_cols if c != date_col]
    
    if not numeric_cols:
         st.warning("No numeric variables found for analysis.")
         return

    selected_cols = st.multiselect("Select Variables to Analyze:", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
    
    if not selected_cols:
        st.warning("Please select at least one variable.")
        return
        
    analysis_df = df[selected_cols]
    
    # 4. Display & Analysis
    st.markdown("#### 4. Analysis Results")
    
    tab1, tab2, tab3 = st.tabs(["📊 Data View", "🔥 Correlation Matrix", "📈 Trend Plot"])
    
    with tab1:
        st.dataframe(analysis_df, width='stretch')
        
    with tab2:
        if len(selected_cols) > 1:
            corr_matrix = analysis_df.corr()
            fig = px.imshow(
                corr_matrix, 
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Select at least two variables for correlation analysis.")
            
    with tab3:
        if date_col:
            # Re-merge date col for plotting
            plot_df = analysis_df.copy()
            plot_df[date_col] = df[date_col]
            
            fig = px.line(plot_df, x=date_col, y=selected_cols, title="Variable Trends Over Time")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No timeline column detected for trend plotting.")

    # 5. Export
    st.markdown("#### 5. Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data (Excel/CSV)",
            data=csv,
            file_name=f"custom_analysis_{selected_dataset_name.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
        
    with col2:
        # Generate a simple text report
        report_text = f"""
        CUSTOM ANALYSIS REPORT
        ======================
        Dataset: {selected_dataset_name}
        Variables: {', '.join(selected_cols)}
        
        Descriptive Statistics:
        -----------------------
        {analysis_df.describe().to_string()}
        
        Correlation Matrix:
        -------------------
        {analysis_df.corr().to_string()}
        
        Generated by U.S.-China Game Theory Analysis Tool
        """
        
        st.download_button(
            label="📄 Download Analysis Report (PDF/Text)",
            data=report_text,
            file_name=f"analysis_report_{selected_dataset_name.replace(' ', '_').lower()}.txt",
            mime="text/plain",
            help="Downloads a formatted text report. Print this file to PDF if needed."
        )


def render_advanced_analytics_page(
    macro_data: pd.DataFrame,
    coop_data: pd.DataFrame,
    tariff_data: pd.DataFrame,
    treasury_data: pd.DataFrame,
    debt_data: pd.DataFrame,
    yield_data: pd.DataFrame,
    harmony_matrix: 'PayoffMatrix',
    pd_matrix: 'PayoffMatrix'
):
    """Wrapper for Advanced Analytics Module."""
    st.markdown('<h2 class="sub-header">📈 Advanced Analytics Hub</h2>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎰 Monte Carlo Simulation", "🛠️ Custom Data Workbench"])
    
    with tab1:
        render_monte_carlo_dashboard(
            macro_data, coop_data, tariff_data, treasury_data, debt_data, yield_data,
            harmony_matrix, pd_matrix
        )
        
    with tab2:
        render_custom_analysis_workbench(
            macro_data, tariff_data, treasury_data, debt_data, yield_data
        )
