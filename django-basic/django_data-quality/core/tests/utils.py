from django.core.files.uploadedfile import SimpleUploadedFile
from ..models import Dataset
import pandas as pd
import numpy as np
import tempfile
import os

def create_test_dataset():
    """Create a test dataset for testing."""
    # Create a test CSV file with numeric and categorical data
    data = {
        'id': range(1, 21),
        'name': [f'Item {i}' for i in range(1, 21)],
        'value': [i * 1.5 for i in range(1, 21)],  # Numeric values
        'category': ['A', 'B', 'C', 'A', 'B'] * 4  # Categorical values
    }
    df = pd.DataFrame(data)
    
    # Save to a temporary file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, 'test_data.csv')
    df.to_csv(file_path, index=False)
    
    # Create a SimpleUploadedFile from the temp file
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    # Clean up the temp file
    os.unlink(file_path)
    os.rmdir(temp_dir)
    
    # Create the test file
    test_file = SimpleUploadedFile(
        'test_data.csv',
        file_content,
        content_type='text/csv'
    )
    
    # Create and return the dataset
    dataset = Dataset.objects.create(
        name='Test Dataset',
        description='A test dataset',
        file=test_file,
        file_type='csv',
        total_rows=20,
        total_columns=4,
        column_names=['id', 'name', 'value', 'category'],
        column_types={
            'id': 'int64',
            'name': 'object',
            'value': 'float64',
            'category': 'object'
        }
    )
    
    return dataset

def generate_profile_config(minimal=False):
    """Generate a test profile configuration."""
    return {
        'minimal': minimal,
        'sample_size': -1,
        'pearson': True,
        'spearman': False,
        'kendall': False,
        'missing_matrix': True,
        'missing_bar': True,
        'missing_heatmap': False,
        'duplicates_table': True,
        'duplicates_matrix': False
    } 