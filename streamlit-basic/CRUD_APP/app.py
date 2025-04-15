import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats
import io
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# Set page configuration
st.set_page_config(page_title="Data Analysis CRUD App", layout="wide")

# Initialize session state for storing multiple dataframes and metadata
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}  # Dictionary to store multiple dataframes
if 'current_df' not in st.session_state:
    st.session_state.current_df = None  # Currently selected dataframe name
if 'sheet_name' not in st.session_state:
    st.session_state.sheet_name = None
if 'df' not in st.session_state:
    st.session_state.df = None  # Current working dataframe
if 'filename' not in st.session_state:
    st.session_state.filename = None  # Current working filename
if 'report_content' not in st.session_state:
    st.session_state.report_content = None
if 'report_filename' not in st.session_state:
    st.session_state.report_filename = None
if 'report_path' not in st.session_state:
    st.session_state.report_path = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def save_dataframe(df, filename, sheet_name=None):
    """Save dataframe to file and update session state"""
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create filename with sheet name if present
    full_filename = f"{filename}_{sheet_name}" if sheet_name else filename
    df.to_csv(f'data/{full_filename}.csv', index=False)
    
    # Update session state
    st.session_state.dataframes[full_filename] = {
        'df': df,
        'source': f"{full_filename}.csv",
        'sheet': sheet_name,
        'upload_time': pd.Timestamp.now()
    }
    st.session_state.current_df = full_filename
    st.session_state.sheet_name = sheet_name
    st.session_state.df = df
    st.session_state.filename = full_filename
    
    return full_filename

def load_dataframe(filename):
    """Load dataframe from CSV file and update session state"""
    df = pd.read_csv(f'data/{filename}.csv')
    
    # Update session state
    st.session_state.df = df
    st.session_state.filename = filename
    if filename in st.session_state.dataframes:
        st.session_state.current_df = filename
        st.session_state.sheet_name = st.session_state.dataframes[filename]['sheet']
    
    return df

def get_excel_sheets(file):
    """Get list of sheets from Excel file"""
    try:
        xls = pd.ExcelFile(file)
        return xls.sheet_names
    except Exception as e:
        st.error(f"Error reading Excel sheets: {str(e)}")
        return []

def calculate_feature_importance(df):
    """Calculate basic feature importance metrics"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    importance_dict = {}
    
    for col in numeric_cols:
        # Calculate variance
        variance = df[col].var()
        # Calculate missing ratio
        missing_ratio = df[col].isnull().mean()
        # Calculate unique ratio
        unique_ratio = len(df[col].unique()) / len(df[col])
        
        importance_dict[col] = {
            'variance': variance,
            'missing_ratio': missing_ratio,
            'unique_ratio': unique_ratio
        }
    
    return pd.DataFrame(importance_dict).T

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return outliers

def prepare_data_for_analysis(df):
    """Prepare data for analysis by handling missing values and encoding"""
    df_prepared = df.copy()
    
    # Handle missing values
    numeric_cols = df_prepared.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_prepared.select_dtypes(include=['object']).columns
    
    # Fill numeric columns with median
    for col in numeric_cols:
        df_prepared[col] = df_prepared[col].fillna(df_prepared[col].median())
    
    # Fill categorical columns with mode
    for col in categorical_cols:
        df_prepared[col] = df_prepared[col].fillna(df_prepared[col].mode()[0])
    
    # Encode categorical variables
    for col in categorical_cols:
        df_prepared[f"{col}_encoded"] = pd.factorize(df_prepared[col])[0]
    
    return df_prepared

def calculate_column_similarity(col1, col2):
    """Calculate similarity between two columns based on their values"""
    # Convert to string and get unique values
    set1 = set(col1.astype(str).unique())
    set2 = set(col2.astype(str).unique())
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

def suggest_join_columns(df1, df2):
    """Suggest potential join columns based on name and content similarity"""
    suggestions = []
    
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Check name similarity
            name_similarity = 1.0 if col1.lower() == col2.lower() else 0.0
            
            # Check content similarity if columns have matching dtypes
            if df1[col1].dtype == df2[col2].dtype:
                content_similarity = calculate_column_similarity(df1[col1], df2[col2])
            else:
                content_similarity = 0.0
            
            # Calculate overall similarity score
            similarity_score = (name_similarity * 0.6) + (content_similarity * 0.4)
            
            if similarity_score > 0.3:  # Threshold for suggestion
                suggestions.append({
                    'col1': col1,
                    'col2': col2,
                    'similarity': similarity_score,
                    'name_match': name_similarity == 1.0,
                    'content_similarity': content_similarity
                })
    
    return sorted(suggestions, key=lambda x: x['similarity'], reverse=True)

def resolve_column_conflicts(df1, df2, join_cols):
    """Resolve column naming conflicts between dataframes"""
    # Get non-join columns
    df1_cols = [col for col in df1.columns if col not in join_cols]
    df2_cols = [col for col in df2.columns if col not in join_cols]
    
    # Find conflicting column names
    conflicts = set(df1_cols) & set(df2_cols)
    
    # Rename conflicting columns
    rename_dict_1 = {col: f"{col}_1" for col in conflicts}
    rename_dict_2 = {col: f"{col}_2" for col in conflicts}
    
    return rename_dict_1, rename_dict_2

# Add HTML report generation function at the top of the file after imports
def generate_html_report(model_type, problem_type, feature_cols, metrics_text, insights, improvements, importance_df=None):
    """Generate HTML report with styling and navigation"""
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .nav {{
                position: fixed;
                top: 0;
                right: 0;
                background: #f8f9fa;
                padding: 10px;
                border-left: 1px solid #ddd;
                height: 100%;
                width: 200px;
                overflow-y: auto;
            }}
            .content {{
                margin-right: 220px;
            }}
            h1, h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
            .metric {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .insight {{
                padding: 10px;
                margin: 5px 0;
                border-left: 4px solid #3498db;
            }}
            .improvement {{
                padding: 10px;
                margin: 5px 0;
                border-left: 4px solid #e74c3c;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div class="nav">
            <h3>Contents</h3>
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#metrics">Model Metrics</a></li>
                <li><a href="#features">Feature Information</a></li>
                <li><a href="#insights">Key Insights</a></li>
                <li><a href="#improvements">Suggested Improvements</a></li>
            </ul>
        </div>
        <div class="content">
            <h1 id="overview">Model Analysis Report</h1>
            <p><strong>Model Type:</strong> {model_type}</p>
            <p><strong>Problem Type:</strong> {problem_type}</p>
            
            <h2 id="metrics">Model Metrics</h2>
            <div class="metric">
                <pre>{metrics_text}</pre>
            </div>
            
            <h2 id="features">Feature Information</h2>
            <p>Features used in the model:</p>
            <ul>
                {' '.join([f'<li>{col}</li>' for col in feature_cols])}
            </ul>
            
            {importance_df.to_html() if importance_df is not None else ''}
            
            <h2 id="insights">Key Insights</h2>
            <div class="insights">
                {' '.join([f'<div class="insight">{insight}</div>' for insight in insights])}
            </div>
            
            <h2 id="improvements">Suggested Improvements</h2>
            <div class="improvements">
                {' '.join([f'<div class="improvement">{improvement}</div>' for improvement in improvements])}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# Main app header
st.title("Data Analysis CRUD Application")
st.markdown("---")

# Sidebar for operations
with st.sidebar:
    st.header("Data Management")
    operation = st.radio("Select Operation", 
                        ["Upload Data", "Data Management"])

# Upload Data Section
if operation == "Upload Data":
    st.header("Upload New Dataset")
    
    # Dataset naming section
    st.subheader("Dataset Identification")
    dataset_name = st.text_input("Enter a unique name for this dataset:",
                                help="This name will be used to identify your dataset throughout the application")
    
    if dataset_name:
        # Check if name already exists
        if dataset_name in st.session_state.dataframes and not st.checkbox("Overwrite existing dataset?"):
            st.warning(f"Dataset name '{dataset_name}' already exists. Please choose a different name or check overwrite option.")
        else:
            uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
            
            if uploaded_file is not None:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension in ['xlsx', 'xls']:
                        # Get available sheets
                        sheets = get_excel_sheets(uploaded_file)
                        if sheets:
                            # Allow user to select multiple sheets
                            selected_sheets = st.multiselect("Select sheets to upload", sheets)
                            
                            if selected_sheets:
                                st.write("Preview of selected sheets:")
                                
                                # Create tabs for sheet previews
                                tabs = st.tabs(selected_sheets)
                                dfs = {}  # Store DataFrames for each sheet
                                
                                for sheet, tab in zip(selected_sheets, tabs):
                                    with tab:
                                        df = pd.read_excel(uploaded_file, sheet_name=sheet)
                                        sheet_name = f"{dataset_name}_{sheet}"
                                        dfs[sheet_name] = df
                                        st.write(f"Preview of sheet: {sheet}")
                                        st.dataframe(df.head())
                                        
                                        # Display basic information
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Rows", df.shape[0])
                                        with col2:
                                            st.metric("Columns", df.shape[1])
                                        with col3:
                                            st.metric("Missing Values", df.isna().sum().sum())
                                
                                if st.button("Save Selected Sheets"):
                                    for sheet_name, df in dfs.items():
                                        # Save to session state
                                        st.session_state.dataframes[sheet_name] = {
                                            'df': df,
                                            'source': uploaded_file.name,
                                            'sheet': sheet_name.split('_')[-1],
                                            'upload_time': pd.Timestamp.now()
                                        }
                                        # Save to file system
                                        save_dataframe(df, dataset_name, sheet_name.split('_')[-1])
                                        st.success(f"Sheet saved as '{sheet_name}'")
                                    
                                    # Set the last sheet as current
                                    last_sheet_name = list(dfs.keys())[-1]
                                    st.session_state.current_df = last_sheet_name
                                    st.session_state.sheet_name = last_sheet_name.split('_')[-1]
                        else:
                            st.error("No sheets found in the Excel file")
                    
                    else:  # CSV file
                        df = pd.read_csv(uploaded_file)
                        
                        # Preview data
                        st.subheader("Dataset Preview")
                        st.dataframe(df.head())
                        
                        # Display basic information
                        st.subheader("Dataset Information")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", df.shape[0])
                        with col2:
                            st.metric("Columns", df.shape[1])
                        with col3:
                            st.metric("Missing Values", df.isna().sum().sum())
                        
                        if st.button("Save Dataset"):
                            # Save to session state
                            st.session_state.dataframes[dataset_name] = {
                                'df': df,
                                'source': uploaded_file.name,
                                'sheet': None,
                                'upload_time': pd.Timestamp.now()
                            }
                            # Save to file system
                            save_dataframe(df, dataset_name)
                            
                            # Set as current dataset
                            st.session_state.current_df = dataset_name
                            st.session_state.sheet_name = None
                            
                            st.success(f"Dataset saved as '{dataset_name}'")
                            
                            # Display dataset summary
                            st.subheader("Dataset Summary")
                            st.write("Data Types:")
                            st.write(df.dtypes)
                            
                            if df.select_dtypes(include=['object']).columns.any():
                                st.write("Sample of Categorical Columns:")
                                for col in df.select_dtypes(include=['object']).columns:
                                    st.write(f"{col}: {', '.join(df[col].unique()[:5].astype(str))}")
                            
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    st.info("Please make sure your file is properly formatted and try again.")
    else:
        st.info("Please enter a name for your dataset before uploading.")
    
    # Display currently loaded datasets
    if st.session_state.dataframes:
        st.subheader("Currently Loaded Datasets")
        dataset_info = []
        for name, info in st.session_state.dataframes.items():
            dataset_info.append({
                'Name': str(name),
                'Rows': int(info['df'].shape[0]),
                'Columns': int(info['df'].shape[1]),
                'Source': str(info['source']),
                'Sheet': str(info['sheet'] if info['sheet'] else 'N/A'),
                'Upload Time': pd.to_datetime(info['upload_time'])
            })
        
        st.dataframe(pd.DataFrame(dataset_info).set_index('Name'))

# Data Management Section
elif operation == "Data Management":
    if st.session_state.dataframes:
        # Add dataset selector at the top of Data Management
        st.subheader("Select Dataset")
        selected_dataset = st.selectbox("Choose a dataset to work with:",
                                      list(st.session_state.dataframes.keys()),
                                      index=list(st.session_state.dataframes.keys()).index(st.session_state.current_df) 
                                      if st.session_state.current_df else 0)
        
        # Update current dataset in session state
        if selected_dataset != st.session_state.current_df:
            st.session_state.current_df = selected_dataset
            st.session_state.df = st.session_state.dataframes[selected_dataset]['df']
            st.session_state.filename = selected_dataset
            st.session_state.sheet_name = st.session_state.dataframes[selected_dataset]['sheet']
        
        current_info = st.session_state.dataframes[selected_dataset]
        df = st.session_state.df  # Use the dataframe from session state
        
        # Show dataset preview with tabs for related sheets
        st.subheader("Dataset Preview")
        
        # Find related sheets (datasets with same base name)
        base_name = selected_dataset.split('_')[0]
        related_sheets = {name: info for name, info in st.session_state.dataframes.items() 
                        if name.startswith(base_name + '_') or name == base_name}
        
        if len(related_sheets) > 1:
            # Create tabs for each related sheet
            sheet_tabs = st.tabs([info['sheet'] if info['sheet'] else 'Main' 
                                for name, info in related_sheets.items()])
            
            # Display data in each tab
            for tab, (name, info) in zip(sheet_tabs, related_sheets.items()):
                with tab:
                    # Display basic information
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", info['df'].shape[0])
                    with col2:
                        st.metric("Columns", info['df'].shape[1])
                    with col3:
                        st.metric("Missing Values", info['df'].isna().sum().sum())
                    with col4:
                        st.metric("Source", info['source'])
                    
                    # Display data preview
                    st.dataframe(info['df'].head())
                    
                    # Display column information
                    with st.expander("Column Details"):
                        col_info = pd.DataFrame({
                            'Type': df.dtypes.astype(str),  # Convert dtype to string
                            'Non-Null Count': df.count().astype(int),
                            'Null Count': df.isna().sum().astype(int),
                            'Unique Values': df.nunique().astype(int)
                        })
                        st.dataframe(col_info)
        else:
            # Display single dataset information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            with col4:
                st.metric("Source", current_info['source'])
            
            # Display data preview
            st.dataframe(df.head())
            
            # Display column information
            with st.expander("Column Details"):
                col_info = pd.DataFrame({
                    'Type': df.dtypes.astype(str),  # Convert dtype to string
                    'Non-Null Count': df.count().astype(int),
                    'Null Count': df.isna().sum().astype(int),
                    'Unique Values': df.nunique().astype(int)
                })
                st.dataframe(col_info)
        
        st.markdown("---")
        
        # View/Edit Data Section
        with st.expander("📝 View/Edit Data", expanded=False):
            st.subheader("View and Edit Data")
            
            # List available datasets
            if os.path.exists('data'):
                available_files = [f.split('.')[0] for f in os.listdir('data') if f.endswith('.csv')]
                if available_files:
                    # Group files by base name (for Excel sheets)
                    file_groups = {}
                    for f in available_files:
                        base_name = f.split('_')[0]
                        if base_name in file_groups:
                            file_groups[base_name].append(f)
                        else:
                            file_groups[base_name] = [f]
                    
                    # First select the base file
                    selected_base = st.selectbox("Select Dataset", list(file_groups.keys()))
                    selected_file = None
                    
                    # If there are multiple sheets, show sheet selection
                    if len(file_groups[selected_base]) > 1:
                        sheet_options = [f.replace(f"{selected_base}_", "") for f in file_groups[selected_base]]
                        selected_sheet = st.selectbox("Select Sheet", sheet_options)
                        selected_file = f"{selected_base}_{selected_sheet}"
                    else:
                        selected_file = file_groups[selected_base][0]
                    
                    if selected_file:
                        df = load_dataframe(selected_file)
                        st.session_state.df = df
                        st.session_state.filename = selected_file
                        
                        # Add option to create new sheet
                        st.subheader("Create New Sheet")
                        create_sheet = st.checkbox("Create a new sheet from current data")
                        
                        if create_sheet:
                            # Options for creating new sheet
                            sheet_operation = st.selectbox("Select Operation for New Sheet",
                                                         ["Filter Data", "Select Columns", "Transform Data"])
                            
                            if sheet_operation == "Filter Data":
                                # Add filtering conditions
                                filter_conditions = {}
                                for col in df.columns:
                                    if st.checkbox(f"Filter {col}"):
                                        unique_vals = df[col].unique()
                                        selected_vals = st.multiselect(f"Select values for {col}", unique_vals)
                                        if selected_vals:
                                            filter_conditions[col] = selected_vals
                                
                                if filter_conditions:
                                    filtered_df = df.copy()
                                    for col, vals in filter_conditions.items():
                                        filtered_df = filtered_df[filtered_df[col].isin(vals)]
                                    st.write("Preview of filtered data:")
                                    st.dataframe(filtered_df.head())
                                    new_sheet_name = st.text_input("Enter name for new sheet:", 
                                                                 value=f"{selected_base}_filtered")
                                    if st.button("Save New Sheet"):
                                        save_dataframe(filtered_df, selected_base, new_sheet_name)
                                        st.success(f"New sheet saved as '{selected_base}_{new_sheet_name}'")
                            
                            elif sheet_operation == "Select Columns":
                                selected_cols = st.multiselect("Select columns for new sheet", df.columns)
                                if selected_cols:
                                    new_df = df[selected_cols]
                                    st.write("Preview of selected columns:")
                                    st.dataframe(new_df.head())
                                    new_sheet_name = st.text_input("Enter name for new sheet:", 
                                                                 value=f"{selected_base}_subset")
                                    if st.button("Save New Sheet"):
                                        save_dataframe(new_df, selected_base, new_sheet_name)
                                        st.success(f"New sheet saved as '{selected_base}_{new_sheet_name}'")
                            
                            elif sheet_operation == "Transform Data":
                                transform_options = st.multiselect("Select transformations",
                                                                ["Aggregate", "Pivot", "Transpose"])
                                if "Aggregate" in transform_options:
                                    # Add aggregation options similar to Data Aggregation section
                                    group_cols = st.multiselect("Group by columns", df.columns)
                                    if group_cols:
                                        agg_cols = st.multiselect("Select columns to aggregate",
                                                                [c for c in df.columns if c not in group_cols])
                                        if agg_cols:
                                            agg_funcs = st.multiselect("Select aggregation functions",
                                                                       ["mean", "sum", "count", "min", "max"])
                                            if agg_funcs:
                                                agg_dict = {col: agg_funcs for col in agg_cols}
                                                new_df = df.groupby(group_cols).agg(agg_dict).reset_index()
                                                st.write("Preview of aggregated data:")
                                                st.dataframe(new_df.head())
                                                new_sheet_name = st.text_input("Enter name for new sheet:",
                                                                             value=f"{selected_base}_aggregated")
                                                if st.button("Save New Sheet"):
                                                    save_dataframe(new_df, selected_base, new_sheet_name)
                                                    st.success(f"New sheet saved as '{selected_base}_{new_sheet_name}'")
                    
                    # Add option to create new dataset
                    st.subheader("Create New Dataset")
                    create_dataset = st.checkbox("Create a new dataset")
                    
                    if create_dataset:
                        creation_method = st.radio("Select creation method",
                                                 ["Upload New Data", "Manual Entry", "Generate Sample Data"])
                        
                        if creation_method == "Upload New Data":
                            uploaded_file = st.file_uploader("Upload data file", type=["csv", "xlsx", "xls"])
                            if uploaded_file:
                                try:
                                    file_ext = uploaded_file.name.split('.')[-1].lower()
                                    if file_ext in ['xlsx', 'xls']:
                                        new_df = pd.read_excel(uploaded_file)
                                    else:
                                        new_df = pd.read_csv(uploaded_file)
                                    st.write("Preview of uploaded data:")
                                    st.dataframe(new_df.head())
                                    new_name = st.text_input("Enter name for new dataset:")
                                    if st.button("Save New Dataset"):
                                        save_dataframe(new_df, new_name)
                                        st.success(f"New dataset saved as '{new_name}'")
                                except Exception as e:
                                    st.error(f"Error loading file: {str(e)}")
                        
                        elif creation_method == "Manual Entry":
                            n_rows = st.number_input("Number of rows", min_value=1, value=5)
                            n_cols = st.number_input("Number of columns", min_value=1, value=3)
                            
                            # Create empty dataframe
                            col_names = []
                            data = {}
                            for i in range(n_cols):
                                col_name = st.text_input(f"Column {i+1} name:")
                                if col_name:
                                    col_names.append(col_name)
                                    data[col_name] = [""] * n_rows
                            
                            if col_names:
                                new_df = pd.DataFrame(data)
                                edited_df = st.data_editor(new_df)
                                new_name = st.text_input("Enter name for new dataset:")
                                if st.button("Save New Dataset"):
                                    save_dataframe(edited_df, new_name)
                                    st.success(f"New dataset saved as '{new_name}'")
                        
                        elif creation_method == "Generate Sample Data":
                            n_rows = st.number_input("Number of rows", min_value=1, value=100)
                            include_cols = st.multiselect("Include column types",
                                                        ["Numeric", "Categorical", "Date"])
                            if include_cols:
                                sample_data = {}
                                if "Numeric" in include_cols:
                                    sample_data["numeric_col"] = np.random.randn(n_rows)
                                if "Categorical" in include_cols:
                                    categories = ["A", "B", "C", "D"]
                                    sample_data["category_col"] = np.random.choice(categories, n_rows)
                                if "Date" in include_cols:
                                    dates = pd.date_range(start="2023-01-01", periods=n_rows)
                                    sample_data["date_col"] = dates
                                
                                new_df = pd.DataFrame(sample_data)
                                st.write("Preview of generated data:")
                                st.dataframe(new_df.head())
                                new_name = st.text_input("Enter name for new dataset:")
                                if st.button("Save New Dataset"):
                                    save_dataframe(new_df, new_name)
                                    st.success(f"New dataset saved as '{new_name}'")
                    
                    # Enhanced join operations
                    st.subheader("Join Operations")
                    join_operation = st.radio("Select join operation",
                                            ["No Join", "Join Two Tables", "Join Three Tables"])
                    
                    if join_operation in ["Join Two Tables", "Join Three Tables"]:
                        # First join (common for both options)
                        st.write("### First Join")
                        other_files = [f for f in available_files if f != selected_file]
                        if len(other_files) >= 1:
                            second_file = st.selectbox("Select second dataset", other_files)
                            if second_file:
                                df2 = load_dataframe(second_file)
                                
                                # Display previews
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("First Dataset Preview")
                                    st.dataframe(df.head())
                                with col2:
                                    st.write("Second Dataset Preview")
                                    st.dataframe(df2.head())
                                
                                # Get join suggestions
                                join_suggestions = suggest_join_columns(df, df2)
                                if join_suggestions:
                                    st.write("Suggested Join Columns:")
                                    for suggestion in join_suggestions:
                                        match_type = "Exact match" if suggestion['name_match'] else "Similar content"
                                        st.write(f"- {suggestion['col1']} ↔ {suggestion['col2']} ({match_type})")
                                
                                # First join configuration
                                join_type1 = st.selectbox("Select first join type", 
                                                        ["inner", "left", "right", "outer"],
                                                        key="join1_type")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    left_on1 = st.multiselect("Select columns from first dataset", 
                                                            df.columns,
                                                            key="left_join1_cols")
                                with col2:
                                    right_on1 = st.multiselect("Select columns from second dataset", 
                                                             df2.columns,
                                                             key="right_join1_cols")
                                
                                # Third table join if selected
                                if join_operation == "Join Three Tables":
                                    st.write("### Second Join")
                                    remaining_files = [f for f in other_files if f != second_file]
                                    if remaining_files:
                                        third_file = st.selectbox("Select third dataset", remaining_files)
                                        if third_file:
                                            df3 = load_dataframe(third_file)
                                            
                                            st.write("Third Dataset Preview")
                                            st.dataframe(df3.head())
                                            
                                            # Second join configuration
                                            join_type2 = st.selectbox("Select second join type",
                                                                    ["inner", "left", "right", "outer"],
                                                                    key="join2_type")
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                left_on2 = st.multiselect("Select columns from joined dataset",
                                                                        df.columns.tolist() + df2.columns.tolist(),
                                                                        key="left_join2_cols")
                                            with col2:
                                                right_on2 = st.multiselect("Select columns from third dataset",
                                                                         df3.columns,
                                                                         key="right_join2_cols")
                                
                                # Perform joins
                                if st.button("Perform Join"):
                                    try:
                                        # First join
                                        rename_dict_1, rename_dict_2 = resolve_column_conflicts(df, df2, 
                                                                                             set(left_on1 + right_on1))
                                        df1_renamed = df.rename(columns=rename_dict_1)
                                        df2_renamed = df2.rename(columns=rename_dict_2)
                                        
                                        joined_df = pd.merge(df1_renamed, df2_renamed,
                                                           left_on=left_on1,
                                                           right_on=right_on1,
                                                           how=join_type1)
                                        
                                        # Second join if needed
                                        if join_operation == "Join Three Tables" and 'df3' in locals():
                                            rename_dict_3 = resolve_column_conflicts(joined_df, df3,
                                                                                  set(left_on2 + right_on2))[1]
                                            df3_renamed = df3.rename(columns=rename_dict_3)
                                            
                                            joined_df = pd.merge(joined_df, df3_renamed,
                                                               left_on=left_on2,
                                                               right_on=right_on2,
                                                               how=join_type2)
                                        
                                        # Save joined dataset
                                        base_name = f"{selected_file}_joined"
                                        if join_operation == "Join Three Tables":
                                            base_name += "_3tables"
                                        
                                        custom_name = st.text_input("Enter name for joined dataset:",
                                                                  value=base_name)
                                        
                                        save_dataframe(joined_df, custom_name)
                                        
                                        # Update session state
                                        st.session_state.df = joined_df
                                        st.session_state.filename = custom_name
                                        st.session_state.sheet_name = None
                                        
                                        # Display results
                                        st.success(f"Join operation completed successfully! Saved as '{custom_name}'")
                                        st.write("### Joined Dataset Preview")
                                        st.dataframe(joined_df.head())
                                        
                                        # Display join statistics
                                        st.subheader("Join Statistics")
                                        cols = st.columns(4 if join_operation == "Join Three Tables" else 3)
                                        with cols[0]:
                                            st.metric("Original Rows", df.shape[0])
                                        with cols[1]:
                                            st.metric("Second Table Rows", df2.shape[0])
                                        if join_operation == "Join Three Tables":
                                            with cols[2]:
                                                st.metric("Third Table Rows", df3.shape[0])
                                        with cols[-1]:
                                            st.metric("Final Rows", joined_df.shape[0])
                                        
                                    except Exception as e:
                                        st.error(f"Error performing join: {str(e)}")
                                        st.info("Please check your join configurations and try again.")
                        else:
                            st.info("Not enough datasets available for joining. Please upload more datasets.")
                else:
                    st.info("No datasets available. Please upload a dataset first.")
            else:
                st.info("No datasets available. Please upload a dataset first.")
        
        # Data Validation Section
        with st.expander("✅ Data Validation", expanded=False):
            st.subheader("Data Validation")
            
            # Basic data validation
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Missing Values Analysis")
                missing_data = pd.DataFrame({
                    'Missing Values': df.isnull().sum(),
                    'Percentage': (df.isnull().sum() / len(df)) * 100
                })
                st.dataframe(missing_data)
            
            with col2:
                st.write("Duplicate Rows")
                duplicates = df.duplicated().sum()
                st.metric("Number of duplicate rows", duplicates)
                
                if duplicates > 0:
                    if st.button("Remove Duplicates"):
                        df = df.drop_duplicates()
                        st.session_state.df = df
                        save_dataframe(df, st.session_state.filename)
                        st.success("Duplicates removed successfully!")
            
            # Data type validation
            st.subheader("Data Types")
            st.dataframe(pd.DataFrame({'Data Type': df.dtypes}))
            
            # Value range validation for numeric columns
            st.subheader("Numeric Columns Range Analysis")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                st.write(f"**{col}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min", float(df[col].min()))
                with col2:
                    st.metric("Max", float(df[col].max()))
                with col3:
                    st.metric("Mean", float(df[col].mean()))
        
        # Feature Analysis Section
        with st.expander("🔍 Feature Analysis", expanded=False):
            st.subheader("Feature Analysis")
            
            # Feature importance
            st.write("Feature Importance Metrics")
            importance_df = calculate_feature_importance(df)
            st.dataframe(importance_df)
            
            # Feature correlations
            st.write("Feature Correlations")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix,
                              title="Feature Correlation Heatmap",
                              color_continuous_scale='RdBu')
                st.plotly_chart(fig)
                
                # High correlation pairs
                high_corr = np.where(np.abs(corr_matrix) > 0.8)
                high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                            for x, y in zip(*high_corr) if x != y and x < y]
                
                if high_corr:
                    st.write("Highly Correlated Features (|correlation| > 0.8):")
                    for feat1, feat2, corr in high_corr:
                        st.write(f"- {feat1} & {feat2}: {corr:.2f}")
        
        # Data Quality Analytics Section
        with st.expander("📊 Data Quality Analytics", expanded=False):
            st.subheader("Data Quality Analytics")
            
            # Overall data quality score
            total_rows = len(df)
            missing_percentage = (df.isnull().sum().sum() / (total_rows * len(df.columns))) * 100
            duplicate_percentage = (df.duplicated().sum() / total_rows) * 100
            
            quality_score = 100 - (missing_percentage + duplicate_percentage) / 2
            
            st.metric("Overall Data Quality Score", f"{quality_score:.2f}%")
            
            # Detailed quality metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Column Completeness")
                completeness = (1 - df.isnull().sum() / len(df)) * 100
                st.dataframe(pd.DataFrame({'Completeness (%)': completeness}))
            
            with col2:
                st.write("Column Uniqueness")
                uniqueness = (df.nunique() / len(df)) * 100
                st.dataframe(pd.DataFrame({'Uniqueness (%)': uniqueness}))
            
            # Outlier detection
            st.subheader("Outlier Analysis")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            selected_col = st.selectbox("Select column for outlier analysis", numeric_cols)
            
            outliers = detect_outliers(df, selected_col)
            fig = px.box(df, y=selected_col, title=f"Box Plot with Outliers - {selected_col}")
            st.plotly_chart(fig)
            
            st.write(f"Number of outliers detected: {len(outliers)}")
            if len(outliers) > 0:
                st.write("Outlier Values:", outliers.values)
        
        # Data Preparation Section
        with st.expander("🛠️ Data Preparation", expanded=False):
            st.subheader("Data Preparation")
            
            # Data preparation options
            prep_options = st.multiselect(
                "Select Preparation Steps",
                ["Handle Missing Values",
                 "Remove Duplicates",
                 "Encode Categorical Variables",
                 "Scale Numeric Features",
                 "Feature Engineering",
                 "Data Type Conversion",
                 "Text Processing",
                 "Date/Time Processing",
                 "Outlier Treatment",
                 "Binning/Discretization",
                 "Column Operations"]
            )

            # Handle Missing Values
            numeric_strategy = None
            categorical_strategy = None
            numeric_value = 0
            categorical_value = "missing"
            
            if "Handle Missing Values" in prep_options:
                st.write("### Missing Values Treatment")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                # Numeric columns
                if len(numeric_cols) > 0:
                    st.write("**Numeric Columns Treatment**")
                    numeric_strategy = st.selectbox(
                        "Select strategy for numeric columns",
                        ["mean", "median", "mode", "constant", "interpolate"]
                    )
                    if numeric_strategy == "constant":
                        numeric_value = st.number_input("Enter value for numeric columns", value=0)
                
                # Categorical columns
                if len(categorical_cols) > 0:
                    st.write("**Categorical Columns Treatment**")
                    categorical_strategy = st.selectbox(
                        "Select strategy for categorical columns",
                        ["mode", "constant", "most_frequent", "new_category"]
                    )
                    if categorical_strategy == "constant":
                        categorical_value = st.text_input("Enter value for categorical columns", value="missing")

            # Remove Duplicates
            subset_cols = None
            if "Remove Duplicates" in prep_options:
                st.write("### Duplicate Removal")
                subset_cols = st.multiselect("Select columns to consider for duplicates", df.columns)

            # Encode Categorical Variables
            if "Encode Categorical Variables" in prep_options:
                st.write("### Categorical Encoding")
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    for col in categorical_cols:
                        st.write(f"**{col}**")
                        encoding_method = st.selectbox(
                            f"Encoding method for {col}",
                            ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"],
                            key=f"encode_{col}"
                        )
                        if encoding_method == "Ordinal Encoding":
                            st.write(f"Order for {col} (comma-separated):")
                            st.text_input(
                                f"Enter categories in order for {col}",
                                value=", ".join(df[col].unique()),
                                key=f"order_{col}"
                            )

            # Apply button
            if st.button("Apply Preparation Steps"):
                try:
                    prepared_df = df.copy()
                    
                    # Apply selected preparation steps
                    if "Handle Missing Values" in prep_options:
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        categorical_cols = df.select_dtypes(include=['object']).columns
                        
                        # Handle numeric columns
                        if len(numeric_cols) > 0 and numeric_strategy:
                            if numeric_strategy == "mean":
                                prepared_df[numeric_cols] = prepared_df[numeric_cols].fillna(prepared_df[numeric_cols].mean())
                            elif numeric_strategy == "median":
                                prepared_df[numeric_cols] = prepared_df[numeric_cols].fillna(prepared_df[numeric_cols].median())
                            elif numeric_strategy == "mode":
                                prepared_df[numeric_cols] = prepared_df[numeric_cols].fillna(prepared_df[numeric_cols].mode().iloc[0])
                            elif numeric_strategy == "constant":
                                prepared_df[numeric_cols] = prepared_df[numeric_cols].fillna(numeric_value)
                            elif numeric_strategy == "interpolate":
                                prepared_df[numeric_cols] = prepared_df[numeric_cols].interpolate()
                        
                        # Handle categorical columns
                        if len(categorical_cols) > 0 and categorical_strategy:
                            if categorical_strategy == "mode":
                                prepared_df[categorical_cols] = prepared_df[categorical_cols].fillna(prepared_df[categorical_cols].mode().iloc[0])
                            elif categorical_strategy in ["constant", "new_category"]:
                                prepared_df[categorical_cols] = prepared_df[categorical_cols].fillna(categorical_value)
                    
                    if "Remove Duplicates" in prep_options and subset_cols:
                        prepared_df = prepared_df.drop_duplicates(subset=subset_cols)
                    
                    if "Encode Categorical Variables" in prep_options:
                        categorical_cols = df.select_dtypes(include=['object']).columns
                        for col in categorical_cols:
                            if f"encode_{col}" in st.session_state:
                                method = st.session_state[f"encode_{col}"]
                                if method == "Label Encoding":
                                    prepared_df[f"{col}_encoded"] = pd.factorize(prepared_df[col])[0]
                                elif method == "One-Hot Encoding":
                                    one_hot = pd.get_dummies(prepared_df[col], prefix=col)
                                    prepared_df = pd.concat([prepared_df, one_hot], axis=1)
                                    prepared_df = prepared_df.drop(col, axis=1)
                                elif method == "Ordinal Encoding":
                                    if f"order_{col}" in st.session_state:
                                        categories = [c.strip() for c in st.session_state[f"order_{col}"].split(",")]
                                    prepared_df[f"{col}_encoded"] = pd.Categorical(
                                        prepared_df[col],
                                        categories=categories,
                                        ordered=True
                                    ).codes
            
                    # Preview prepared data
                    st.write("Preview of Prepared Data:")
                    st.dataframe(prepared_df.head())
                    
                    # Save prepared dataset
                    custom_name = st.text_input(
                        "Enter a name for the prepared dataset:", 
                        value=f"{st.session_state.filename}_prepared"
                    )
                    
                    if st.button("Save Prepared Dataset"):
                        save_dataframe(prepared_df, custom_name)
                        st.session_state.df = prepared_df
                        st.session_state.filename = custom_name
                        st.session_state.sheet_name = None
                        st.success(f"Prepared dataset saved as '{custom_name}'")
                        
                        # Display preparation summary
                        st.subheader("Preparation Summary")
                        summary = []
                        if "Handle Missing Values" in prep_options:
                            if numeric_strategy:
                                summary.append(f"- Numeric missing values handled using {numeric_strategy}")
                            if categorical_strategy:
                                summary.append(f"- Categorical missing values handled using {categorical_strategy}")
                        if "Remove Duplicates" in prep_options and subset_cols:
                            summary.append(f"- Duplicates removed based on {', '.join(subset_cols)}")
                        if "Encode Categorical Variables" in prep_options:
                            summary.append("- Categorical variables encoded")
                        st.write("\n".join(summary))
                
                except Exception as e:
                    st.error(f"Error during data preparation: {str(e)}")
                    st.info("Please check your selected options and try again.")
        
        # Data Aggregation Section
        with st.expander("📊 Data Aggregation", expanded=False):
            st.subheader("Data Aggregation and Grouping")
            
            # Column selection for grouping
            all_columns = df.columns.tolist()
            group_by_columns = st.multiselect("Select columns to group by", all_columns)
            
            if group_by_columns:
                # Select columns to aggregate
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                agg_columns = st.multiselect(
                    "Select columns to aggregate", 
                    [col for col in numeric_cols if col not in group_by_columns]
                )
                
                if agg_columns:
                    # Select aggregation functions for each column
                    st.subheader("Aggregation Functions")
                    agg_functions = {}
                    
                    for col in agg_columns:
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            st.write(f"**{col}**")
                        with col2:
                            selected_funcs = st.multiselect(
                                f"Select aggregation functions for {col}",
                                ["mean", "sum", "count", "min", "max", "std", "median"],
                                key=f"agg_func_{col}"
                            )
                            if selected_funcs:
                                agg_functions[col] = selected_funcs
                    
                    if agg_functions:
                        try:
                            # Perform grouping and aggregation
                            grouped_df = df.groupby(group_by_columns).agg(agg_functions)
                            grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
                            grouped_df = grouped_df.reset_index()
                            
                            # Display results
                            st.subheader("Aggregated Data Preview")
                            st.dataframe(grouped_df.head())
                            
                            # Save options
                            st.subheader("Save Options")
                            group_cols_str = "_".join(group_by_columns)
                            custom_name = st.text_input(
                                "Enter a name for the aggregated dataset:", 
                                value=f"{st.session_state.filename}_grouped_by_{group_cols_str}",
                                key="agg_custom_name"
                            )
                            
                            if st.button("Save Aggregated Dataset"):
                                save_dataframe(grouped_df, custom_name)
                                st.session_state.df = grouped_df
                                st.session_state.filename = custom_name
                                st.session_state.sheet_name = None
                                st.success(f"Dataset saved as '{custom_name}'")
                                
                                # Display summary statistics
                                st.subheader("Summary Statistics of Aggregated Data")
                                st.write(grouped_df.describe())
                            
                        except Exception as e:
                            st.error(f"Error during aggregation: {str(e)}")
                            st.info("Please check your selected columns and aggregation functions.")
                    else:
                        st.info("Please select aggregation functions for at least one column.")
                else:
                    st.info("Please select columns to aggregate.")
            else:
                st.info("Please select columns to group by.")
        
        # Data Monitoring Section
        with st.expander("📈 Data Monitoring", expanded=False):
            st.subheader("Data Monitoring")
            
            # Basic statistics monitoring
            st.write("Statistical Monitoring")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in numeric_cols:
                st.write(f"**{col} Statistics**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean", f"{df[col].mean():.2f}")
                with col2:
                    st.metric("Std Dev", f"{df[col].std():.2f}")
                with col3:
                    st.metric("Skewness", f"{df[col].skew():.2f}")
                with col4:
                    st.metric("Kurtosis", f"{df[col].kurtosis():.2f}")
                
                # Distribution plot
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig)
            
            # Data drift detection
            st.subheader("Data Drift Detection")
            st.info("To detect data drift, upload a new version of the dataset and compare distributions.")
            
            drift_file = st.file_uploader("Upload new dataset for drift comparison", type=["csv", "xlsx", "xls"])
            if drift_file is not None:
                try:
                    # Load new dataset
                    file_extension = drift_file.name.split('.')[-1].lower()
                    if file_extension in ['xlsx', 'xls']:
                        new_df = pd.read_excel(drift_file)
                    else:
                        new_df = pd.read_csv(drift_file)
                    
                    # Compare distributions
                    for col in numeric_cols:
                        if col in new_df.columns:
                            st.write(f"**Distribution Comparison for {col}**")
                            
                            # Perform KS test
                            ks_statistic, p_value = stats.ks_2samp(df[col].dropna(), new_df[col].dropna())
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("KS Statistic", f"{ks_statistic:.3f}")
                            with col2:
                                st.metric("P-value", f"{p_value:.3f}")
                            
                            if p_value < 0.05:
                                st.warning(f"Significant distribution change detected in {col}")
                            else:
                                st.success(f"No significant distribution change in {col}")
                            
                            # Plot comparison
                            fig = px.histogram(title=f"Distribution Comparison - {col}",
                                             labels={'value': col, 'dataset': 'Version'})
                            fig.add_histogram(x=df[col], name='Original')
                            fig.add_histogram(x=new_df[col], name='New')
                            st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error comparing datasets: {str(e)}")
        
        # Analyze Data Section
        with st.expander("📉 Analyze Data", expanded=False):
            st.subheader("Analyze Data")
            
            if st.session_state.df is not None:
                df = st.session_state.df
                
                # Enhanced Analysis Options
                analysis_type = st.selectbox("Select Analysis Type", 
                                           ["Summary Statistics", 
                                            "Distribution Analysis",
                                            "Correlation Analysis", 
                                            "Time Series Analysis",
                                            "Pattern Recognition",
                                            "Outlier Detection",
                                            "Data Composition",
                                            "Statistical Tests"])
                
                if analysis_type == "Summary Statistics":
                    st.subheader("Summary Statistics")
                    
                    # Basic statistics
                    st.write("Basic Statistics")
                    st.write(df.describe())
                    
                    # Additional statistics
                    st.write("Additional Statistics")
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    additional_stats = pd.DataFrame({
                        'Skewness': df[numeric_cols].skew(),
                        'Kurtosis': df[numeric_cols].kurtosis(),
                        'Variance': df[numeric_cols].var(),
                        'Coefficient of Variation': df[numeric_cols].std() / df[numeric_cols].mean()
                    })
                    st.write(additional_stats)
                    
                    # Column-wise statistics
                    st.write("Column Information")
                    col_info = pd.DataFrame({
                        'Data Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isna().sum(),
                        'Unique Values': df.nunique(),
                        'Memory Usage': df.memory_usage(deep=True)
                    })
                    st.write(col_info)
                
                elif analysis_type == "Distribution Analysis":
                    st.subheader("Distribution Analysis")
                    
                    # Select columns for analysis
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    selected_cols = st.multiselect("Select columns for distribution analysis", numeric_cols)
                    
                    if selected_cols:
                        plot_type = st.selectbox("Select Plot Type", 
                                               ["Histogram", "Box Plot", "Violin Plot", "KDE Plot"])
                        
                        for col in selected_cols:
                            if plot_type == "Histogram":
                                fig = px.histogram(df, x=col, 
                                                 title=f"Histogram of {col}",
                                                 marginal="box")  # Add box plot on the margin
                            elif plot_type == "Box Plot":
                                fig = px.box(df, y=col, 
                                           title=f"Box Plot of {col}",
                                           points="all")  # Show all points
                            elif plot_type == "Violin Plot":
                                fig = px.violin(df, y=col, 
                                              title=f"Violin Plot of {col}",
                                              box=True)  # Include box plot inside
                            else:  # KDE Plot
                                fig = px.histogram(df, x=col, 
                                                 title=f"KDE Plot of {col}",
                                                 marginal="rug",  # Add rug plot on the margin
                                                 histnorm='probability density')
                            st.plotly_chart(fig)
                            
                            # Distribution statistics
                            st.write(f"Distribution Statistics for {col}:")
                            stats_df = pd.DataFrame({
                                'Metric': ['Mean', 'Median', 'Mode', 'Std Dev', 'Skewness', 'Kurtosis'],
                                'Value': [
                                    df[col].mean(),
                                    df[col].median(),
                                    df[col].mode().iloc[0],
                                    df[col].std(),
                                    df[col].skew(),
                                    df[col].kurtosis()
                                ]
                            })
                            st.write(stats_df)
                
                elif analysis_type == "Correlation Analysis":
                    st.subheader("Correlation Analysis")
                    
                    # Select correlation method
                    corr_method = st.selectbox("Select Correlation Method",
                                             ["Pearson", "Spearman", "Kendall"])
                    
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    if len(numeric_cols) > 1:
                        # Correlation matrix
                        corr_matrix = df[numeric_cols].corr(method=corr_method.lower())
                        
                        # Heatmap
                        fig = px.imshow(corr_matrix,
                                      title=f"{corr_method} Correlation Matrix",
                                      color_continuous_scale='RdBu')
                        st.plotly_chart(fig)
                        
                        # Significant correlations
                        st.write("### Significant Correlations")
                        threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.5)
                        
                        # Get pairs with correlation above threshold
                        high_corr = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                if abs(corr_matrix.iloc[i, j]) >= threshold:
                                    high_corr.append({
                                        'Variable 1': corr_matrix.columns[i],
                                        'Variable 2': corr_matrix.columns[j],
                                        'Correlation': corr_matrix.iloc[i, j]
                                    })
                        
                        if high_corr:
                            st.write(pd.DataFrame(high_corr))
                            
                            # Scatter plots for highly correlated pairs
                            st.write("### Scatter Plots for Correlated Variables")
                            for pair in high_corr:
                                fig = px.scatter(df, 
                                               x=pair['Variable 1'],
                                               y=pair['Variable 2'],
                                               title=f"Correlation: {pair['Correlation']:.3f}")
                                st.plotly_chart(fig)
                        else:
                            st.info(f"No correlations found above threshold {threshold}")
                    else:
                        st.warning("Not enough numeric columns for correlation analysis")
                
                elif analysis_type == "Time Series Analysis":
                    st.subheader("Time Series Analysis")
                    
                    # Identify potential date columns
                    date_cols = []
                    for col in df.columns:
                        try:
                            pd.to_datetime(df[col])
                            date_cols.append(col)
                        except:
                            continue
                    
                    if date_cols:
                        # Select date column
                        date_col = st.selectbox("Select Date Column", date_cols)
                        
                        # Convert to datetime
                        df[date_col] = pd.to_datetime(df[date_col])
                        
                        # Select numeric column for analysis
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        value_col = st.selectbox("Select Value Column", numeric_cols)
                        
                        # Time series plot
                        fig = px.line(df, x=date_col, y=value_col,
                                    title=f"Time Series Plot: {value_col} over time")
                        st.plotly_chart(fig)
                        
                        # Resample options
                        resample_freq = st.selectbox("Select Resampling Frequency",
                                                   ["Day", "Week", "Month", "Quarter", "Year"])
                        
                        freq_map = {"Day": "D", "Week": "W", "Month": "M",
                                  "Quarter": "Q", "Year": "Y"}
                        
                        # Resample data
                        resampled = df.set_index(date_col)[value_col].resample(freq_map[resample_freq])
                        
                        # Select aggregation method
                        agg_method = st.selectbox("Select Aggregation Method",
                                                ["Mean", "Sum", "Min", "Max"])
                        
                        if agg_method == "Mean":
                            resampled = resampled.mean()
                        elif agg_method == "Sum":
                            resampled = resampled.sum()
                        elif agg_method == "Min":
                            resampled = resampled.min()
                        else:
                            resampled = resampled.max()
                        
                        # Plot resampled data
                        fig = px.line(resampled,
                                    title=f"{value_col} by {resample_freq} ({agg_method})")
                        st.plotly_chart(fig)
                        
                        # Moving averages
                        st.write("### Moving Averages")
                        window_size = st.slider("Select Window Size", 2, 30, 7)
                        
                        ma = df.set_index(date_col)[value_col].rolling(window=window_size).mean()
                        
                        fig = px.line(title=f"{window_size}-period Moving Average")
                        fig.add_scatter(x=df[date_col], y=df[value_col],
                                      name="Original", mode="lines")
                        fig.add_scatter(x=df[date_col], y=ma,
                                      name="Moving Average", mode="lines")
                        st.plotly_chart(fig)
                    else:
                        st.warning("No date columns detected in the dataset")
                
                elif analysis_type == "Pattern Recognition":
                    st.subheader("Pattern Recognition")
                    
                    # Select numeric columns
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    selected_cols = st.multiselect("Select columns for pattern analysis", numeric_cols)
                    
                    if selected_cols:
                        # Trend analysis
                        st.write("### Trend Analysis")
                        for col in selected_cols:
                            trend = np.polyfit(range(len(df[col])), df[col], 1)
                            trend_line = np.poly1d(trend)(range(len(df[col])))
                            
                            fig = px.scatter(df, y=col, title=f"Trend Analysis: {col}")
                            fig.add_scatter(y=trend_line, mode="lines", name="Trend Line")
                            st.plotly_chart(fig)
                            
                            # Calculate trend statistics
                            trend_direction = "Increasing" if trend[0] > 0 else "Decreasing"
                            trend_strength = abs(trend[0])
                            st.write(f"Trend Direction: {trend_direction}")
                            st.write(f"Trend Strength: {trend_strength:.4f}")
                        
                        # Seasonality detection
                        st.write("### Seasonality Detection")
                        for col in selected_cols:
                            # Calculate autocorrelation
                            autocorr = pd.Series(df[col]).autocorr()
                            st.write(f"Autocorrelation for {col}: {autocorr:.4f}")
                            
                            # Plot autocorrelation
                            fig = px.line(y=pd.Series(df[col]).autocorr(lag=50),
                                        title=f"Autocorrelation Plot: {col}")
                            st.plotly_chart(fig)
                
                elif analysis_type == "Outlier Detection":
                    st.subheader("Outlier Detection")
                    
                    # Select method
                    method = st.selectbox("Select Outlier Detection Method",
                                        ["IQR Method", "Z-Score Method", "Modified Z-Score"])
                    
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    selected_cols = st.multiselect("Select columns for outlier detection", numeric_cols)
                    
                    if selected_cols:
                        for col in selected_cols:
                            st.write(f"### Outlier Analysis for {col}")
                            
                            if method == "IQR Method":
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                            
                            elif method == "Z-Score Method":
                                z_scores = np.abs(stats.zscore(df[col].dropna()))
                                outliers = df[col][z_scores > 3]
                                st.write(f"Number of outliers detected: {len(outliers)}")
                                if len(outliers) > 0:
                                    st.write("Outlier Values:")
                                    st.write(outliers)
                            
                            else:  # Modified Z-Score
                                median = df[col].median()
                                mad = stats.median_abs_deviation(df[col])
                                modified_z_scores = 0.6745 * (df[col] - median) / mad
                                outliers = df[col][np.abs(modified_z_scores) > 3.5]
                            
                            # Plot with outliers highlighted
                            fig = px.box(df, y=col, title=f"Box Plot with Outliers: {col}")
                            st.plotly_chart(fig)
                            
                            # Outlier statistics
                            st.write(f"Number of outliers detected: {len(outliers)}")
                            if len(outliers) > 0:
                                st.write("Outlier Summary:")
                                st.write(outliers.describe())
                
                elif analysis_type == "Data Composition":
                    st.subheader("Data Composition Analysis")
                    
                    # Overall Dataset Summary
                    st.write("### 📊 Dataset Overview")
                    total_memory = df.memory_usage(deep=True).sum()
                    total_cells = df.size
                    missing_cells = df.isnull().sum().sum()
                    duplicate_rows = df.duplicated().sum()
                    
                    # Display key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{df.shape[0]:,}")
                    with col2:
                        st.metric("Total Columns", f"{df.shape[1]:,}")
                    with col3:
                        st.metric("Memory Usage", f"{total_memory / 1024**2:.2f} MB")
                    with col4:
                        st.metric("Duplicate Rows", f"{duplicate_rows:,}")
                    
                    # Data Quality Score
                    quality_metrics = {
                        'Completeness': (1 - missing_cells/total_cells) * 100,
                        'Uniqueness': (1 - duplicate_rows/df.shape[0]) * 100,
                        'Validity': (df.notnull().all(axis=1).sum() / df.shape[0]) * 100
                    }
                    
                    st.write("### 📈 Data Quality Metrics")
                    quality_df = pd.DataFrame(list(quality_metrics.items()), 
                                           columns=['Metric', 'Score'])
                    
                    # Plot quality metrics
                    fig_quality = px.bar(quality_df, x='Metric', y='Score',
                                       title="Data Quality Scores",
                                       text='Score')
                    fig_quality.update_traces(texttemplate='%{text:.1f}%')
                    st.plotly_chart(fig_quality)
                    
                    # Column Type Analysis
                    st.write("### 📋 Column Type Analysis")
                    dtype_counts = df.dtypes.value_counts()
                    fig_dtypes = px.pie(values=dtype_counts.values,
                                      names=dtype_counts.index.astype(str),
                                      title="Distribution of Data Types")
                    st.plotly_chart(fig_dtypes)
                    
                    # Detailed Column Analysis
                    st.write("### 🔍 Detailed Column Analysis")
                    col_analysis = pd.DataFrame({
                        'Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum(),
                        'Null %': (df.isnull().sum() / len(df)) * 100,
                        'Unique Values': df.nunique(),
                        'Unique %': (df.nunique() / len(df)) * 100,
                        'Memory Usage': df.memory_usage(deep=True)
                    })
                    st.dataframe(col_analysis)
                    
                    # Generate Insights
                    st.write("### 💡 Key Insights")
                    insights = []
                    
                    # Data quality insights
                    if quality_metrics['Completeness'] < 95:
                        insights.append("⚠️ Data completeness is below 95%. Consider handling missing values.")
                    if quality_metrics['Uniqueness'] < 90:
                        insights.append("⚠️ High number of duplicate rows detected. Consider deduplication.")
                    
                    # Memory usage insights
                    if total_memory > 1024**3:  # If more than 1GB
                        insights.append("⚠️ Large memory usage. Consider optimizing data types or sampling.")
                    
                    # Column type insights
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(numeric_cols) > 0:
                        insights.append(f"📊 Dataset contains {len(numeric_cols)} numeric columns suitable for statistical analysis.")
                    if len(categorical_cols) > 0:
                        insights.append(f"📑 Dataset contains {len(categorical_cols)} categorical columns suitable for grouping and encoding.")
                    
                    # Display insights
                    for insight in insights:
                        st.write(insight)
                    
                    # Recommendations
                    st.write("### 🎯 Recommendations")
                    recommendations = []
                    
                    # Data quality recommendations
                    if quality_metrics['Completeness'] < 95:
                        recommendations.append("Consider imputing missing values using appropriate methods (mean, median, mode).")
                    if quality_metrics['Uniqueness'] < 90:
                        recommendations.append("Review and remove duplicate rows if they're not intentional.")
                    
                    # Memory optimization recommendations
                    if total_memory > 1024**3:
                        recommendations.append("Consider optimizing data types (e.g., using categorical types for low-cardinality columns).")
                    
                    # Column-specific recommendations
                    high_cardinality_cols = [col for col in categorical_cols 
                                           if df[col].nunique() > 0.5 * len(df)]
                    if high_cardinality_cols:
                        recommendations.append(f"Consider encoding or binning high-cardinality categorical columns: {', '.join(high_cardinality_cols)}")
                    
                    # Display recommendations
                    for rec in recommendations:
                        st.write("• " + rec)
                    
                    # Export Analysis Report
                    st.write("### 📑 Export Analysis Report")
                    if st.button("Generate Analysis Report"):
                        try:
                            # Create report directory if it doesn't exist
                            if not os.path.exists('outputs'):
                                os.makedirs('outputs')
                            
                            # Generate HTML report
                            report_content = f"""
                            <html>
                            <head>
                                <style>
                                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                    h1, h2 {{ color: #2c3e50; }}
                                    .metric {{ 
                                        background: #f8f9fa;
                                        padding: 15px;
                                        margin: 10px 0;
                                        border-radius: 5px;
                                    }}
                                    .insight {{ 
                                        padding: 10px;
                                        margin: 5px 0;
                                        border-left: 4px solid #3498db;
                                    }}
                                    .recommendation {{
                                        padding: 10px;
                                        margin: 5px 0;
                                        border-left: 4px solid #e74c3c;
                                    }}
                                </style>
                            </head>
                            <body>
                                <h1>Data Analysis Report</h1>
                                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                                
                                <h2>Dataset Overview</h2>
                                <div class="metric">
                                    <p><strong>Total Rows:</strong> {df.shape[0]:,}</p>
                                    <p><strong>Total Columns:</strong> {df.shape[1]:,}</p>
                                    <p><strong>Memory Usage:</strong> {total_memory / 1024**2:.2f} MB</p>
                                    <p><strong>Duplicate Rows:</strong> {duplicate_rows:,}</p>
                                </div>
                                
                                <h2>Data Quality Metrics</h2>
                                <div class="metric">
                                    {''.join([f"<p><strong>{k}:</strong> {v:.1f}%</p>" for k, v in quality_metrics.items()])}
                                </div>
                                
                                <h2>Key Insights</h2>
                                <div class="insights">
                                    {' '.join([f'<div class="insight">{insight}</div>' for insight in insights])}
                                </div>
                                
                                <h2>Recommendations</h2>
                                <div class="recommendations">
                                    {' '.join([f'<div class="recommendation">{rec}</div>' for rec in recommendations])}
                                </div>
                                
                                <h2>Detailed Column Analysis</h2>
                                {col_analysis.to_html()}
                            </body>
                            </html>
                            """
                            
                            # Save report
                            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                            report_filename = f'data_analysis_report_{timestamp}.html'
                            report_path = os.path.join('outputs', report_filename)
                            
                            with open(report_path, 'w', encoding='utf-8') as f:
                                f.write(report_content)
                            
                            # Display success message and provide download link
                            st.success(f"Report generated successfully! Saved as '{report_filename}'")
                            
                            # Display report in app
                            st.components.v1.html(report_content, height=600, scrolling=True)
                            
                            # Download button
                            with open(report_path, 'r', encoding='utf-8') as f:
                                st.download_button(
                                    label="📥 Download Analysis Report",
                                    data=f.read(),
                                    file_name=report_filename,
                                    mime="text/html"
                                )
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                            st.info("Please ensure all data is properly loaded and accessible.")
            else:
                st.info("Please select a dataset first using the View/Edit Data section.")
        
        # Machine Learning Section
        with st.expander("🤖 Machine Learning", expanded=False):
            st.subheader("Create and Export Machine Learning Model")
            
            # Feature Selection first
            st.subheader("Feature Selection")
            
            # Select target variable first
            target_col = st.selectbox("Select Target Variable", df.columns)
            
            if target_col:
                # Automatic problem type detection based on target variable
                target_values = df[target_col]
                is_categorical = (
                    df[target_col].dtype == 'object' or
                    df[target_col].dtype == 'category' or
                    len(df[target_col].unique()) <= 10
                )
                
                problem_type = st.radio(
                    "Select Problem Type",
                    ["Classification", "Regression"],
                    index=0 if is_categorical else 1,
                    key="problem_type_selection"
                )
                
                # Select features after target is chosen
                feature_cols = st.multiselect(
                    "Select Features for Model",
                    [col for col in df.columns if col != target_col]
                )
                
                if feature_cols and target_col:
                    # Prepare data
                    X = df[feature_cols]
                    y = df[target_col]
                    
                    # Handle categorical variables
                    categorical_cols = X.select_dtypes(include=['object']).columns
                    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
                    
                    if len(categorical_cols) > 0:
                        st.write("Categorical columns detected. Encoding...")
                        encoders = {}
                        for col in categorical_cols:
                            encoders[col] = LabelEncoder()
                            X[col] = encoders[col].fit_transform(X[col].astype(str))
                    
                    # Feature scaling option
                    if st.checkbox("Apply Feature Scaling"):
                        scaler = StandardScaler()
                        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                    
                    # Model Selection
                    st.subheader("Model Selection")
                    
                    if problem_type == "Classification":
                        model_type = st.selectbox(
                            "Select Model",
                            ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"]
                        )
                        
                        n_estimators = st.slider("Number of Trees (for Random Forest)", 10, 200, 100)
                        
                        if model_type == "Random Forest":
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                random_state=42
                            )
                        elif model_type == "Logistic Regression":
                            model = LogisticRegression(random_state=42)
                        elif model_type == "Decision Tree":
                            model = DecisionTreeClassifier(random_state=42)
                        else:
                            model = SVC(random_state=42)
                    
                    else:  # Regression
                        model_type = st.selectbox(
                            "Select Model",
                            ["Random Forest", "Linear Regression", "Decision Tree", "SVR"]
                        )
                        
                        n_estimators = st.slider("Number of Trees (for Random Forest)", 10, 200, 100)
                        
                        if model_type == "Random Forest":
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                random_state=42
                            )
                        elif model_type == "Linear Regression":
                            model = LinearRegression()
                        elif model_type == "Decision Tree":
                            model = DecisionTreeRegressor(random_state=42)
                        else:
                            model = SVR()
                    
                    # Training Configuration
                    st.subheader("Training Configuration")
                    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
                    random_state = st.number_input("Random State", 0, 100, 42)
                    
                    if st.button("Train Model"):
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                        
                        # Display train/test split preview
                        st.subheader("Train/Test Split Preview")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Training Set Preview")
                            train_preview = pd.DataFrame(X_train, columns=feature_cols)
                            train_preview['Target'] = y_train
                            st.dataframe(train_preview.head())
                            
                            # Training set statistics
                            st.write("Training Set Statistics:")
                            st.write(f"- Number of samples: {len(X_train)}")
                            if problem_type == "Classification":
                                st.write("- Class distribution:")
                                st.write(pd.Series(y_train).value_counts().to_frame())
                            else:
                                st.write("- Target statistics:")
                                st.write(pd.Series(y_train).describe().to_frame())
                        
                        with col2:
                            st.write("Test Set Preview")
                            test_preview = pd.DataFrame(X_test, columns=feature_cols)
                            test_preview['Target'] = y_test
                            st.dataframe(test_preview.head())
                            
                            # Test set statistics
                            st.write("Test Set Statistics:")
                            st.write(f"- Number of samples: {len(X_test)}")
                            if problem_type == "Classification":
                                st.write("- Class distribution:")
                                st.write(pd.Series(y_test).value_counts().to_frame())
                            else:
                                st.write("- Target statistics:")
                                st.write(pd.Series(y_test).describe().to_frame())
                        
                        # Feature distribution comparison
                        st.subheader("Feature Distribution Comparison")
                        selected_feature = st.selectbox("Select feature to compare", feature_cols)
                        
                        # Create distribution plot
                        fig_dist = px.histogram(
                            pd.concat([
                                pd.DataFrame({'value': X_train[selected_feature], 'set': 'Train'}),
                                pd.DataFrame({'value': X_test[selected_feature], 'set': 'Test'})
                            ]),
                            x='value',
                            color='set',
                            barmode='overlay',
                            title=f"Distribution of {selected_feature} in Train and Test Sets",
                            opacity=0.7
                        )
                        st.plotly_chart(fig_dist)
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Display results
                        st.subheader("Model Performance")
                        
                        if problem_type == "Classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            st.metric("Accuracy", f"{accuracy:.2%}")
                            
                            # Display classification report
                            st.write("Classification Report:")
                            report = classification_report(y_test, y_pred)
                            st.text(report)
                            
                            # Cross-validation configuration
                            if problem_type == "Classification":
                                # Check minimum samples per class
                                min_samples = min(np.bincount(y))
                                max_splits = min(5, min_samples)  # Use minimum of 5 or min_samples
                                if max_splits < 2:
                                    st.warning("Not enough samples per class for cross-validation")
                                    cv_scores = None
                                else:
                                    cv_scores = cross_val_score(model, X, y, cv=max_splits)
                                    st.write("Cross-validation Scores:")
                                    st.write(f"Mean: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
                            else:  # Regression
                                # For regression, check total sample size
                                max_splits = min(5, len(X))
                                if max_splits < 2:
                                    st.warning("Not enough samples for cross-validation")
                                    cv_scores = None
                                else:
                                    cv_scores = cross_val_score(model, X, y, cv=max_splits, scoring='r2')
                                    st.write("Cross-validation R² Scores:")
                                    st.write(f"Mean: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
                        
                        else:  # Regression
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                            st.metric("R² Score", f"{r2:.4f}")
                            
                            # Cross-validation configuration
                            if problem_type == "Classification":
                                # Check minimum samples per class
                                min_samples = min(np.bincount(y))
                                max_splits = min(5, min_samples)  # Use minimum of 5 or min_samples
                                if max_splits < 2:
                                    st.warning("Not enough samples per class for cross-validation")
                                    cv_scores = None
                                else:
                                    cv_scores = cross_val_score(model, X, y, cv=max_splits)
                                    st.write("Cross-validation Scores:")
                                    st.write(f"Mean: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
                            else:  # Regression
                                # For regression, check total sample size
                                max_splits = min(5, len(X))
                                if max_splits < 2:
                                    st.warning("Not enough samples for cross-validation")
                                    cv_scores = None
                                else:
                                    cv_scores = cross_val_score(model, X, y, cv=max_splits, scoring='r2')
                                    st.write("Cross-validation R² Scores:")
                                    st.write(f"Mean: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
                        
                        # Feature importance for applicable models
                        if hasattr(model, 'feature_importances_'):
                            st.subheader("Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            st.bar_chart(importance_df.set_index('Feature'))
                        
                        # Export model
                        st.subheader("Export Model")
                        model_name = st.text_input("Enter model name for saving:", 
                                                 value=f"{model_type}_{problem_type}")
                        
                        if st.button("Export Model"):
                            # Create model info dictionary
                            model_info = {
                                'model': model,
                                'feature_columns': feature_cols,
                                'target_column': target_col,
                                'categorical_encoders': encoders if len(categorical_cols) > 0 else None,
                                'numeric_columns': list(numeric_cols),
                                'problem_type': problem_type,
                                'model_type': model_type,
                                'training_date': pd.Timestamp.now(),
                                'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
                            }
                            
                            # Save model info
                            if not os.path.exists('models'):
                                os.makedirs('models')
                            
                            model_path = f'models/{model_name}.pkl'
                            with open(model_path, 'wb') as f:
                                pickle.dump(model_info, f)
                            
                            st.success(f"Model exported successfully to {model_path}")
                            
                            # Download button for the model
                            with open(model_path, 'rb') as f:
                                model_bytes = f.read()
                                st.download_button(
                                    label="Download Model",
                                    data=model_bytes,
                                    file_name=f"{model_name}.pkl",
                                    mime="application/octet-stream"
                                )
                        
                        # Model Evaluation and Insights
                        st.header("Model Evaluation and Insights")
                        
                        # Create tabs for different evaluation aspects
                        eval_tabs = st.tabs(
                            ["Performance Metrics", "Feature Analysis", "Predictions Analysis"]
                        )

                        with eval_tabs[0]:  # Performance Metrics
                            st.subheader("Model Performance Metrics")
                            
                            if problem_type == "Classification":
                                # Display confusion matrix
                                cm = confusion_matrix(y_test, y_pred)
                                fig_cm = px.imshow(
                                    cm,
                                    labels=dict(x="Predicted", y="Actual"),
                                    title="Confusion Matrix"
                                )
                                
                                # Add text annotations to confusion matrix
                                for i in range(len(cm)):
                                    for j in range(len(cm)):
                                        fig_cm.add_annotation(
                                            x=j,
                                            y=i,
                                            text=str(cm[i, j]),
                                            showarrow=False,
                                            font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                                        )
                                
                                st.plotly_chart(fig_cm)
                                
                                # Display classification report
                                st.text("Classification Report:")
                                st.text(classification_report(y_test, y_pred))
                                
                                # Scatter plot of probabilities
                                if hasattr(model, "predict_proba"):
                                    proba = model.predict_proba(X_test)
                                    fig = px.scatter(
                                        x=proba[:, 1],
                                        y=[1 if p else 0 for p in y_test],
                                        title="Prediction Probabilities vs Actual Values",
                                        labels={"x": "Predicted Probability", "y": "Actual Class"}
                                    )
                                    st.plotly_chart(fig)
                            
                            else:  # Regression
                                # Display regression metrics
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test, y_pred)
                                
                                st.write(f"Mean Squared Error: {mse:.4f}")
                                st.write(f"Root Mean Squared Error: {rmse:.4f}")
                                st.write(f"R² Score: {r2:.4f}")
                                
                                # Residuals plot
                                residuals = y_test - y_pred
                                fig_resid = px.scatter(
                                    x=y_pred,
                                    y=residuals,
                                    title="Residuals Plot",
                                    labels={"x": "Predicted Values", "y": "Residuals"}
                                )
                                st.plotly_chart(fig_resid)
                                
                                # Actual vs Predicted plot
                                fig = px.scatter(
                                    x=y_test,
                                    y=y_pred,
                                    title="Actual vs Predicted Values",
                                    labels={"x": "Actual Values", "y": "Predicted Values"}
                                )
                                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                                        y=[y_test.min(), y_test.max()],
                                                        mode='lines',
                                                        name='Perfect Prediction'))
                                st.plotly_chart(fig)

                        with eval_tabs[1]:  # Feature Analysis
                            st.subheader("Feature Importance Analysis")
                            
                            if hasattr(model, "feature_importances_"):
                                importance_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig_imp = px.bar(
                                    importance_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title="Feature Importance"
                                )
                                st.plotly_chart(fig_imp)
                            
                            # Feature correlations
                            numeric_features = X_test.select_dtypes(include=['float64', 'int64']).columns
                            if len(numeric_features) > 1:
                                corr_matrix = X_test[numeric_features].corr()
                                fig_corr = px.imshow(
                                    corr_matrix,
                                    title="Feature Correlation Matrix"
                                )
                                st.plotly_chart(fig_corr)

                        with eval_tabs[2]:  # Predictions Analysis
                            st.subheader("Predictions Analysis")
                            
                            # Create analysis dataframe
                            analysis_df = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': y_pred,
                                'Difference': y_test - y_pred
                            })
                            
                            # Distribution of actual vs predicted values
                            fig_dist = px.histogram(
                                analysis_df,
                                x=['Actual', 'Predicted'],
                                barmode='overlay',
                                title="Distribution of Actual vs Predicted Values",
                                opacity=0.7
                            )
                            st.plotly_chart(fig_dist)
                            
                            # Distribution of prediction errors
                            fig_err = px.histogram(
                                analysis_df,
                                x='Difference',
                                title="Distribution of Prediction Errors",
                                marginal="box"
                            )
                            st.plotly_chart(fig_err)
                            
                            # Display summary statistics
                            st.write("Summary Statistics of Predictions:")
                            st.write(analysis_df.describe())

                        # Model Report Generation Section
                        if st.button("Generate Model Report"):
                            try:
                                # Create outputs directory if it doesn't exist
                                outputs_dir = 'outputs'
                                if not os.path.exists(outputs_dir):
                                    os.makedirs(outputs_dir)
                                
                                # Format metrics text
                                if problem_type == "Classification":
                                    metrics_text = classification_report(y_test, y_pred)
                                    metrics_text += f"\nAccuracy Score: {accuracy:.4f}"
                                else:  # Regression
                                    metrics_text = f"""
                                    R² Score: {r2:.4f}
                                    Mean Squared Error: {mse:.4f}
                                    Root MSE: {rmse:.4f}
                                    """
                                
                                # Prepare feature importance data if available
                                importance_df = None
                                if hasattr(model, 'feature_importances_'):
                                    importance_df = pd.DataFrame({
                                        'Feature': feature_cols,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=False)
                                
                                # Generate insights based on model performance
                                insights = []
                                improvements = []
                                
                                if problem_type == "Classification":
                                    if accuracy >= 0.9:
                                        insights.append("Model shows excellent accuracy (>90%)")
                                    elif accuracy >= 0.7:
                                        insights.append("Model shows good accuracy (>70%)")
                                    else:
                                        insights.append("Model accuracy needs improvement (<70%)")
                                        improvements.append("Consider collecting more training data")
                                        improvements.append("Try feature engineering or selection")
                                else:
                                    if r2 >= 0.9:
                                        insights.append("Model shows excellent R² score (>0.9)")
                                    elif r2 >= 0.7:
                                        insights.append("Model shows good R² score (>0.7)")
                                    else:
                                        insights.append("Model R² score needs improvement (<0.7)")
                                        improvements.append("Consider feature engineering")
                                        improvements.append("Try non-linear models")
                                
                                # Generate HTML report
                                html_content = generate_html_report(
                                    model_type=model_type,
                                    problem_type=problem_type,
                                    feature_cols=feature_cols,
                                    metrics_text=metrics_text,
                                    insights=insights,
                                    improvements=improvements,
                                    importance_df=importance_df
                                )
                                
                                # Save HTML report
                                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                report_filename = f'model_report_{timestamp}.html'
                                report_path = os.path.join(outputs_dir, report_filename)
                                
                                with open(report_path, 'w', encoding='utf-8') as f:
                                    f.write(html_content)
                                
                                st.success(f"Report generated successfully! Saved as {report_filename}")
                                
                                # Provide download button
                                with open(report_path, 'r', encoding='utf-8') as f:
                                    st.download_button(
                                        label="Download Model Report",
                                        data=f.read(),
                                        file_name=report_filename,
                                        mime="text/html"
                                    )
                                
                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")
                                st.info("Please ensure the model has been trained and all metrics are available.")

            else:
                st.info("Please select features and target variable to proceed.")
        
        # Delete Data Section
        with st.expander("🗑️ Delete Data", expanded=False):
            st.subheader("Delete Data")
            
            if os.path.exists('data'):
                available_files = [f.split('.')[0] for f in os.listdir('data') if f.endswith('.csv')]
                if available_files:
                    file_to_delete = st.selectbox("Select Dataset to Delete", available_files)
                    
                    if st.button("Delete Dataset", type="primary"):
                        os.remove(f'data/{file_to_delete}.csv')
                        if st.session_state.filename == file_to_delete:
                            st.session_state.df = None
                            st.session_state.filename = None
                        st.success(f"Dataset '{file_to_delete}' has been deleted successfully!")
                else:
                    st.info("No datasets available to delete.")
            else:
                st.info("No datasets available to delete.")
        
        # Data Quality Analysis Section
        with st.expander("🔍 Data Quality Analysis", expanded=False):
            st.subheader("Data Quality Analysis")
            
            if df is not None:
                # Outlier Detection
                method = st.selectbox(
                    "Select Outlier Detection Method",
                    ["IQR", "Z-Score", "Isolation Forest"]
                )
                
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    col = st.selectbox("Select Column for Outlier Detection", numeric_cols)
                    
                    if method == "IQR":
                        outliers = detect_outliers(df, col)
                        st.write(f"Number of outliers detected: {len(outliers)}")
                        if len(outliers) > 0:
                            st.write("Outlier Values:")
                            st.write(outliers)
                    
                    elif method == "Z-Score":
                        z_scores = np.abs(stats.zscore(df[col].dropna()))
                        outliers = df[col][z_scores > 3]
                        st.write(f"Number of outliers detected: {len(outliers)}")
                        if len(outliers) > 0:
                            st.write("Outlier Values:")
                            st.write(outliers)
                    
                    else:  # Isolation Forest
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        yhat = iso_forest.fit_predict(df[col].values.reshape(-1, 1))
                        outliers = df[col][yhat == -1]
                        st.write(f"Number of outliers detected: {len(outliers)}")
                        if len(outliers) > 0:
                            st.write("Outlier Values:")
                            st.write(outliers)
                
                # Data Quality Metrics
                quality_metrics = {
                    'Completeness': (1 - df.isnull().mean().mean()) * 100,
                    'Uniqueness': (df.nunique() / len(df)).mean() * 100,
                    'Consistency': (df.select_dtypes(include=['object']).apply(
                        lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 1
                    ).mean() if len(df.select_dtypes(include=['object']).columns) > 0 else 100) * 100
                }
                
                quality_df = pd.DataFrame({
                    'Metric': quality_metrics.keys(),
                    'Score': quality_metrics.values()
                })
                
                st.subheader("Data Quality Scores")
                fig_quality = px.bar(
                    quality_df,
                    x='Metric',
                    y='Score',
                    title="Data Quality Metrics"
                )
                st.plotly_chart(fig_quality)
                
                # Data Type Distribution
                dtype_counts = df.dtypes.value_counts()
                fig_dtypes = px.pie(
                    values=dtype_counts.values,
                    names=dtype_counts.index.astype(str),
                    title="Data Type Distribution"
                )
                st.plotly_chart(fig_dtypes)
                
                # Column-wise Analysis
                col_analysis = pd.DataFrame({
                    'Column': list(df.columns),
                    'Type': list(df.dtypes),
                    'Missing (%)': [df[col].isnull().mean() * 100 for col in df.columns],
                    'Unique (%)': [(df[col].nunique() / len(df)) * 100 for col in df.columns],
                    'Memory Usage (KB)': [df[col].memory_usage(deep=True) / 1024 for col in df.columns]
                })
                
                st.subheader("Column-wise Analysis")
                st.dataframe(col_analysis)
                
                # Download Quality Report
                report_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                report_name = f"quality_report_{report_time}.html"
                
                report_content = f"""
                <html>
                    <head>
                        <title>Data Quality Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1, h2 {{ color: #2c3e50; }}
                            .metric {{ 
                                background: #f8f9fa;
                                padding: 15px;
                                margin: 10px 0;
                                border-radius: 5px;
                            }}
                            table {{
                                border-collapse: collapse;
                                width: 100%;
                                margin: 15px 0;
                            }}
                            th, td {{
                                border: 1px solid #ddd;
                                padding: 8px;
                                text-align: left;
                            }}
                            th {{ background-color: #f8f9fa; }}
                        </style>
                    </head>
                    <body>
                        <h1>Data Quality Report</h1>
                        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        
                        <h2>Dataset Overview</h2>
                        <div class="metric">
                            <p><strong>Number of Rows:</strong> {len(df)}</p>
                            <p><strong>Number of Columns:</strong> {len(df.columns)}</p>
                            <p><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024:.2f} KB</p>
                        </div>
                        
                        <h2>Data Quality Metrics</h2>
                        {quality_df.to_html()}
                        
                        <h2>Column Analysis</h2>
                        {col_analysis.to_html()}
                    </body>
                </html>
                """
                
                st.download_button(
                    label="Download Quality Report",
                    data=report_content,
                    file_name=report_name,
                    mime="text/html"
                )
