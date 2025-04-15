import os
import pandas as pd
import polars as pl
import json
import mimetypes
from django.core.exceptions import ValidationError
from django.conf import settings
import re
import numpy as np
from datetime import datetime
import logging

def get_file_signature(file):
    """Get the first few bytes of a file to check its signature."""
    current_position = file.tell()
    signature = file.read(8)
    file.seek(current_position)
    return signature

def validate_file_type(filename, expected_type):
    """Validate file type based on extension."""
    extension = get_file_extension(filename)
    if expected_type == 'csv' and extension in ['csv', 'txt']:
        return True
    elif expected_type == 'excel' and extension in ['xls', 'xlsx']:
        return True
    elif expected_type == 'json' and extension == 'json':
        return True
    return False

def validate_file_size(file):
    """Validate file size is under the maximum limit."""
    # 50MB limit (in bytes)
    max_size = 50 * 1024 * 1024
    if file.size > max_size:
        raise ValidationError(f'File size must be no more than {max_size/(1024*1024):.0f}MB')

def get_file_extension(filename):
    """Get the file extension from a filename."""
    return os.path.splitext(filename)[1].lower()[1:]

def validate_file_extension(file):
    """Validate file extension."""
    ext = os.path.splitext(file.name)[1].lower()
    valid_extensions = ['.csv', '.xlsx', '.xls', '.json']
    if not ext in valid_extensions:
        raise ValidationError(f'Unsupported file extension. Allowed extensions are: {", ".join(valid_extensions)}')

def validate_file_content(file):
    """Validate file content matches its declared type."""
    import pandas as pd
    import json
    
    try:
        # Try reading as CSV
        pd.read_csv(file, nrows=5)
        file.seek(0)
        return 'csv'
    except:
        file.seek(0)
        try:
            # Try reading as Excel
            pd.read_excel(file, nrows=5)
            file.seek(0)
            return 'excel'
        except:
            file.seek(0)
            try:
                # Try reading as JSON
                json.load(file)
                file.seek(0)
                return 'json'
            except:
                file.seek(0)
                raise ValidationError('File content is not valid CSV, Excel, or JSON format')
    finally:
        file.seek(0)

def validate_profile_config(config):
    """Validate profile configuration."""
    if not isinstance(config, dict):
        raise ValidationError('Profile configuration must be a dictionary')

    # Required fields and their types
    required_fields = {
        'minimal': bool,
        'sample_size': int,
        'correlations': dict,
        'missing_diagrams': dict,
        'duplicates': dict
    }

    # Check required fields
    for field, field_type in required_fields.items():
        if field not in config:
            raise ValidationError(f'Missing required field: {field}')
        if not isinstance(config[field], field_type):
            raise ValidationError(f'Field {field} must be of type {field_type.__name__}')

    # Validate nested dictionaries
    if 'correlations' in config:
        corr_config = config['correlations']
        if not all(isinstance(v, bool) for v in corr_config.values()):
            raise ValidationError('All correlation options must be boolean')

    if 'missing_diagrams' in config:
        missing_config = config['missing_diagrams']
        if not all(isinstance(v, bool) for v in missing_config.values()):
            raise ValidationError('All missing diagram options must be boolean')

    if 'duplicates' in config:
        dup_config = config['duplicates']
        if not all(isinstance(v, bool) for v in dup_config.values()):
            raise ValidationError('All duplicate options must be boolean')

    # Validate sample size
    if config['sample_size'] != -1 and config['sample_size'] <= 0:
        raise ValidationError('Sample size must be -1 (all data) or a positive integer')

def validate_profile_status(status):
    """Validate profile status."""
    valid_statuses = ['not_generated', 'pending', 'processing', 'completed', 'failed']
    if status not in valid_statuses:
        raise ValidationError(f'Invalid profile status. Must be one of: {", ".join(valid_statuses)}')
    return True

def process_uploaded_file(file, file_type):
    """Process the uploaded file based on its type and return a DataFrame."""
    if file_type == 'csv':
        return pd.read_csv(file)
    elif file_type == 'excel':
        return pd.read_excel(file)
    elif file_type == 'json':
        data = json.load(file)
        return pd.DataFrame(data)
    raise ValueError(f'Unsupported file type: {file_type}') 

def polars_to_pandas(df_polars):
    """
    Convert a Polars DataFrame to a Pandas DataFrame with improved type handling.
    
    Args:
        df_polars (pl.DataFrame): Input Polars DataFrame
    
    Returns:
        pd.DataFrame: Converted Pandas DataFrame
    """
    # Convert Polars DataFrame to Pandas
    df_dict = df_polars.to_pandas()
    
    # Additional type conversion and handling
    for col in df_dict.columns:
        # Handle Polars specific types
        polars_dtype = df_polars[col].dtype
        
        # Check for numeric types
        if pl.Float64 == polars_dtype or pl.Float32 == polars_dtype or \
           pl.Int64 == polars_dtype or pl.Int32 == polars_dtype or \
           pl.UInt64 == polars_dtype or pl.UInt32 == polars_dtype:
            df_dict[col] = pd.to_numeric(df_dict[col], errors='coerce')
        
        # Check for datetime types
        elif pl.Date == polars_dtype or pl.Datetime == polars_dtype:
            df_dict[col] = pd.to_datetime(df_dict[col], errors='coerce')
    
    return df_dict

def read_dataset_file(dataset_or_file, file_type=None):
    """
    Read a dataset file into a Polars DataFrame with robust type handling.
    
    Args:
        dataset_or_file: Dataset or file object
        file_type: Optional file type specification
    
    Returns:
        pl.DataFrame: Loaded dataset
    """
    # Determine the file path and type
    if hasattr(dataset_or_file, 'file'):
        # If it's a dataset object
        file_path = dataset_or_file.file.path
        file_type = dataset_or_file.file_type
    elif hasattr(dataset_or_file, 'path'):
        # If it's a file object
        file_path = dataset_or_file.path
        # Try to infer file type from extension if not provided
        if not file_type:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.csv', '.txt']:
                file_type = 'csv'
            elif ext in ['.xls', '.xlsx']:
                file_type = 'excel'
            elif ext == '.json':
                file_type = 'json'
            else:
                raise ValueError(f'Unsupported file type: {ext}')
    else:
        raise ValueError('Invalid input: must be a dataset or file object')
    
    try:
        # Polars-based reading with robust type inference
        if file_type == 'csv':
            df = pl.read_csv(
                file_path, 
                try_parse_dates=True,  # Attempt to parse datetime columns
                null_values=['', 'NA', 'null', 'None', 'NaN'],
                ignore_errors=True  # Ignore parsing errors
            )
        elif file_type == 'excel':
            # Convert Excel to CSV first for Polars
            temp_csv = file_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
            pd.read_excel(file_path).to_csv(temp_csv, index=False)
            df = pl.read_csv(
                temp_csv, 
                try_parse_dates=True,
                null_values=['', 'NA', 'null', 'None', 'NaN'],
                ignore_errors=True
            )
            os.remove(temp_csv)  # Clean up temporary CSV
        elif file_type == 'json':
            df = pl.read_json(file_path)
        else:
            raise ValueError(f'Unsupported file type: {file_type}')
        
        return df
    
    except Exception as e:
        raise ValueError(f'Error reading dataset with Polars: {str(e)}')

def generate_profile_config():
    """Generate default profile configuration."""
    return {
        'samples': {
            'head': 10,
            'tail': 10
        },
        'correlations': {
            'pearson': True,
            'spearman': False,
            'kendall': False
        },
        'missing_diagrams': {
            'matrix': True,
            'bar': True,
            'heatmap': False
        }
    } 

def clean_column_name(name):
    """Clean column name for consistent formatting."""
    # Remove special characters and replace spaces with underscores
    cleaned = re.sub(r'[^\w\s-]', '', name)
    cleaned = re.sub(r'[-\s]+', '_', cleaned)
    
    # Ensure it starts with a letter or underscore
    if cleaned[0].isdigit():
        cleaned = '_' + cleaned
    
    return cleaned.lower()

def format_file_size(size):
    """Format file size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{int(size)} {unit}" if size.is_integer() else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def is_valid_file_size(file):
    """Check if file size is within limits."""
    max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 10 * 1024 * 1024)  # Default 10MB
    return file.size <= max_size 

def read_dataset(dataset):
    """Read a dataset file into a pandas DataFrame with improved data type handling."""
    try:
        file_path = dataset.file.path
        import numpy as np
        import pandas as pd
        
        # Define common NA/null values
        na_values = ['', ' ', 'nan', 'NaN', 'NULL', 'null', 'None', 'none', 'NA', 'na', '#N/A']
        
        # Read the file based on its type with initial dtype inference disabled
        if dataset.file_type == 'csv':
            df = pd.read_csv(file_path, dtype=str, na_values=na_values, keep_default_na=True)
        elif dataset.file_type == 'excel':
            df = pd.read_excel(file_path, dtype=str, na_values=na_values, keep_default_na=True)
        elif dataset.file_type == 'json':
            df = pd.read_json(file_path, dtype=str)
        else:
            raise ValueError(f"Unsupported file type: {dataset.file_type}")
        
        # Convert columns to appropriate types
        for column in df.columns:
            # Try numeric conversion first
            numeric_series = pd.to_numeric(df[column], errors='coerce')
            if numeric_series.notna().any():
                # Check if the column should be integer
                if numeric_series.dropna().apply(lambda x: float(x).is_integer()).all():
                    df[column] = numeric_series.astype('Int64')  # Use nullable integer type
                else:
                    df[column] = numeric_series.astype('float64')
                continue
            
            # Try datetime conversion
            try:
                datetime_series = pd.to_datetime(df[column], errors='coerce')
                if datetime_series.notna().any():
                    df[column] = datetime_series
                    continue
            except (ValueError, TypeError):
                pass
            
            # Try boolean conversion
            if df[column].dropna().isin(['True', 'False', 'true', 'false', '0', '1']).all():
                df[column] = df[column].map({'True': True, 'true': True, '1': True,
                                           'False': False, 'false': False, '0': False})
                continue
            
            # If no other type fits, keep as string but clean it
            df[column] = df[column].astype(str).replace('nan', np.nan)
        
        # Update dataset metadata
        dataset.total_rows = len(df)
        dataset.total_columns = len(df.columns)
        dataset.column_names = df.columns.tolist()
        dataset.column_types = {col: str(df[col].dtype) for col in df.columns}
        dataset.save(update_fields=['total_rows', 'total_columns', 'column_names', 'column_types'])
        
        return df
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Error reading dataset: {str(e)}")

def calculate_numeric_stats(df, column):
    """Calculate numeric statistics for a given column."""
    try:
        stats = {
            'mean': float(df[column].mean()),
            'median': float(df[column].median()),
            'min': float(df[column].min()),
            'max': float(df[column].max()),
            'std': float(df[column].std())
        }
        return stats
    except Exception as e:
        print(f"Error calculating stats for column {column}: {str(e)}")
        return None

def validate_dataset_rules(dataset, rules):
    """Validate a dataset against a set of governance rules with enhanced validation."""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Read the dataset
    try:
        df = read_dataset(dataset)
    except Exception as e:
        logger.error(f"Error reading dataset {dataset.id}: {str(e)}")
        return [{
            'rule_name': rule.name,
            'status': 'error',
            'message': f'Failed to read dataset: {str(e)}'
        } for rule in rules]
    
    # Store validation results
    validation_results = []
    
    for rule in rules:
        try:
            column = rule.column_name
            start_time = datetime.now()
            
            # Skip if column doesn't exist
            if column not in df.columns:
                validation_results.append({
                    'rule_name': rule.name,
                    'status': 'error',
                    'message': f'Column {column} not found in dataset',
                    'execution_time': (datetime.now() - start_time).total_seconds()
                })
                continue
            
            # Get column data
            series = df[column]
            violations = pd.Series(False, index=series.index)
            message = f'Validation for {column}'
            
            # Validate based on rule type
            if rule.rule_type == 'range':
                params = rule.parameters
                min_val = float(params.get('min', float('-inf')))
                max_val = float(params.get('max', float('inf')))
                
                # Convert to numeric, handling non-numeric values
                numeric_series = pd.to_numeric(series, errors='coerce')
                violations = (numeric_series < min_val) | (numeric_series > max_val) | numeric_series.isna()
                message = f'Range validation ({min_val} to {max_val}) for {column}'
            
            elif rule.rule_type == 'required':
                violations = series.isna() | (series == '')
                message = f'Required field validation for {column}'
            
            elif rule.rule_type == 'unique':
                duplicates = series.duplicated(keep=False)
                violations = duplicates & ~series.isna()
                message = f'Uniqueness validation for {column}'
            
            elif rule.rule_type == 'categorical':
                allowed_values = set(rule.parameters.get('allowed_values', []))
                violations = ~series.isin(allowed_values)
                message = f'Categorical validation for {column}'
            
            elif rule.rule_type == 'date_format':
                date_format = rule.parameters.get('format', '%Y-%m-%d')
                
                def is_valid_date(x):
                    if pd.isna(x):
                        return True
                    try:
                        if isinstance(x, str):
                            datetime.strptime(x, date_format)
                        return True
                    except (ValueError, TypeError):
                        return False
                
                violations = ~series.apply(is_valid_date)
                message = f'Date format validation ({date_format}) for {column}'
            
            elif rule.rule_type == 'format':
                pattern = rule.parameters.get('pattern', '')
                
                def matches_pattern(x):
                    if pd.isna(x):
                        return True
                    try:
                        return bool(re.match(pattern, str(x)))
                    except (TypeError, re.error):
                        return False
                
                violations = ~series.apply(matches_pattern)
                message = f'Format validation ({pattern}) for {column}'
            
            elif rule.rule_type == 'cross_column':
                comp_column = rule.parameters.get('comparison_column')
                operator = rule.parameters.get('operator')
                
                if comp_column not in df.columns:
                    validation_results.append({
                        'rule_name': rule.name,
                        'status': 'error',
                        'message': f'Comparison column {comp_column} not found',
                        'execution_time': (datetime.now() - start_time).total_seconds()
                    })
                    continue
                
                # Convert both columns to numeric for comparison
                col1 = pd.to_numeric(series, errors='coerce')
                col2 = pd.to_numeric(df[comp_column], errors='coerce')
                
                if operator == '==':
                    violations = col1 != col2
                elif operator == '>':
                    violations = col1 <= col2
                elif operator == '>=':
                    violations = col1 < col2
                elif operator == '<':
                    violations = col1 >= col2
                elif operator == '<=':
                    violations = col1 > col2
                
                message = f'Cross-column validation ({column} {operator} {comp_column})'
            
            # Get violation details
            violation_count = violations.sum()
            violation_indices = violations[violations].index.tolist()
            violation_values = series[violations].tolist()
            
            # Create validation result
            execution_time = (datetime.now() - start_time).total_seconds()
            validation_results.append({
                'rule_name': rule.name,
                'status': 'pass' if violation_count == 0 else 'fail',
                'message': message,
                'violation_count': int(violation_count),
                'total_count': len(series),
                'violation_percentage': float(violation_count / len(series) * 100),
                'violation_indices': violation_indices[:100],  # Limit to first 100 violations
                'violation_values': violation_values[:100],    # Limit to first 100 violations
                'execution_time': execution_time
            })
            
            # Save validation result to database
            RuleValidationResult.objects.create(
                dataset=dataset,
                rule=rule,
                passed=violation_count == 0,
                failed_rows={
                    'indices': violation_indices[:100],
                    'values': violation_values[:100]
                } if violation_count > 0 else None,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error validating rule {rule.name}: {str(e)}")
            validation_results.append({
                'rule_name': rule.name,
                'status': 'error',
                'message': f'Validation error: {str(e)}',
                'execution_time': (datetime.now() - start_time).total_seconds()
            })
    
    return validation_results 