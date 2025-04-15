from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from ..utils import (
    validate_file_type,
    validate_file_content,
    validate_file_size,
    get_file_extension,
    read_dataset_file,
    generate_profile_config,
    clean_column_name,
    format_file_size
)
from .utils import create_test_dataset
import pandas as pd
import numpy as np
import tempfile
import json
import os

class FileValidationTests(TestCase):
    def setUp(self):
        self.dataset = create_test_dataset()

    def test_validate_file_type(self):
        """Test file type validation."""
        # Test valid file types
        self.assertTrue(validate_file_type('test.csv', 'csv'))
        self.assertTrue(validate_file_type('test.xlsx', 'excel'))
        self.assertTrue(validate_file_type('test.json', 'json'))
        
        # Test invalid file types
        self.assertFalse(validate_file_type('test.txt', 'csv'))
        self.assertFalse(validate_file_type('test.doc', 'excel'))
        self.assertFalse(validate_file_type('test.xml', 'json'))
        
        # Test case sensitivity
        self.assertTrue(validate_file_type('test.CSV', 'csv'))
        self.assertTrue(validate_file_type('test.XLSX', 'excel'))
        self.assertTrue(validate_file_type('test.JSON', 'json'))

    def test_validate_file_content(self):
        """Test file content validation."""
        # Test valid CSV content
        df = pd.DataFrame({'test': range(5)})
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            tmp.seek(0)
            content = tmp.read()
            self.assertTrue(validate_file_content(content, 'csv'))
        
        # Test valid Excel content
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as tmp:
            df.to_excel(tmp.name, index=False)
            tmp.seek(0)
            content = tmp.read()
            self.assertTrue(validate_file_content(content, 'excel'))
        
        # Test valid JSON content
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            df.to_json(tmp.name, orient='records')
            tmp.seek(0)
            content = tmp.read()
            self.assertTrue(validate_file_content(content, 'json'))
        
        # Test invalid content
        self.assertFalse(validate_file_content(b'invalid content', 'csv'))
        self.assertFalse(validate_file_content(b'invalid content', 'excel'))
        self.assertFalse(validate_file_content(b'invalid content', 'json'))

    def test_validate_file_size(self):
        """Test file size validation."""
        # Test small file
        small_content = b'test,data\n1,2'
        self.assertTrue(validate_file_size(small_content))
        
        # Test large file
        large_df = pd.DataFrame({'test': range(1000000)})  # 1M rows
        with tempfile.NamedTemporaryFile() as tmp:
            large_df.to_csv(tmp.name, index=False)
            tmp.seek(0)
            large_content = tmp.read()
            self.assertFalse(validate_file_size(large_content))

    def test_get_file_extension(self):
        """Test file extension extraction."""
        self.assertEqual(get_file_extension('test.csv'), 'csv')
        self.assertEqual(get_file_extension('test.xlsx'), 'xlsx')
        self.assertEqual(get_file_extension('test.json'), 'json')
        self.assertEqual(get_file_extension('test'), '')
        self.assertEqual(get_file_extension('test.CSV'), 'csv')
        self.assertEqual(get_file_extension('path/to/test.csv'), 'csv')

    def test_read_dataset_file(self):
        """Test dataset file reading."""
        # Test CSV reading
        df = pd.DataFrame({
            'id': range(5),
            'name': [f'Item {i}' for i in range(5)],
            'value': np.random.randn(5)
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            result_df = read_dataset_file(tmp.name, 'csv')
            pd.testing.assert_frame_equal(df, result_df)
        
        # Test Excel reading
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as tmp:
            df.to_excel(tmp.name, index=False)
            result_df = read_dataset_file(tmp.name, 'excel')
            pd.testing.assert_frame_equal(df, result_df)
        
        # Test JSON reading
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            df.to_json(tmp.name, orient='records')
            result_df = read_dataset_file(tmp.name, 'json')
            pd.testing.assert_frame_equal(df, result_df)

    def test_generate_profile_config(self):
        """Test profile configuration generation."""
        config = generate_profile_config()
        
        # Test required fields
        self.assertIn('minimal', config)
        self.assertIn('sample_size', config)
        self.assertIn('correlations', config)
        self.assertIn('duplicates', config)
        self.assertIn('missing_diagrams', config)
        self.assertIn('plot', config)
        self.assertIn('html', config)
        
        # Test plot options
        self.assertIn('histogram', config['plot'])
        self.assertIn('correlation', config['plot'])
        self.assertIn('missing', config['plot'])
        
        # Test HTML options
        self.assertIn('minify_html', config['html'])
        self.assertIn('use_local_assets', config['html'])

    def test_clean_column_name(self):
        """Test column name cleaning."""
        self.assertEqual(clean_column_name('Test Column'), 'test_column')
        self.assertEqual(clean_column_name('Test-Column'), 'test_column')
        self.assertEqual(clean_column_name('Test.Column'), 'test_column')
        self.assertEqual(clean_column_name('Test Column!@#'), 'test_column')
        self.assertEqual(clean_column_name('123Test Column'), '_123test_column')
        self.assertEqual(clean_column_name(''), '_empty')

    def test_format_file_size(self):
        """Test file size formatting."""
        self.assertEqual(format_file_size(0), '0 B')
        self.assertEqual(format_file_size(1024), '1.0 KB')
        self.assertEqual(format_file_size(1024 * 1024), '1.0 MB')
        self.assertEqual(format_file_size(1024 * 1024 * 1024), '1.0 GB')
        self.assertEqual(format_file_size(500), '500 B')
        self.assertEqual(format_file_size(1500), '1.5 KB')
        self.assertEqual(format_file_size(1024 * 1024 * 1.5), '1.5 MB') 