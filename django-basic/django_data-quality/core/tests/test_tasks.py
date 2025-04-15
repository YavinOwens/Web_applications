from django.test import TestCase
from django.core.files.base import ContentFile
from ..tasks import generate_profile_report
from .utils import create_test_dataset, generate_profile_config
import os
import json
import pandas as pd
import numpy as np
import tempfile

class TasksTests(TestCase):
    def setUp(self):
        self.dataset = create_test_dataset()

    def tearDown(self):
        # Clean up any files created during tests
        if self.dataset.file:
            if os.path.exists(self.dataset.file.path):
                os.remove(self.dataset.file.path)
        if self.dataset.profile_report:
            if os.path.exists(self.dataset.profile_report.path):
                os.remove(self.dataset.profile_report.path)
        if self.dataset.profile_json:
            if os.path.exists(self.dataset.profile_json.path):
                os.remove(self.dataset.profile_json.path)

    def test_generate_profile_report_basic(self):
        """Test basic profile report generation."""
        config = generate_profile_config(minimal=True)
        result = generate_profile_report(self.dataset, config)
        
        self.assertTrue(result)
        self.assertEqual(self.dataset.profile_status, 'ready')
        self.assertIsNotNone(self.dataset.profile_report)
        self.assertIsNotNone(self.dataset.profile_json)
        self.assertIsNotNone(self.dataset.profile_last_updated)
        self.assertEqual(self.dataset.profile_config, config)

    def test_generate_profile_report_with_sampling(self):
        """Test profile generation with data sampling."""
        # Create a larger dataset
        df = pd.DataFrame({
            'id': range(1000),
            'value': np.random.randn(1000)
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            with open(tmp.name, 'rb') as f:
                self.dataset.file.save('large_dataset.csv', ContentFile(f.read()))
        
        config = generate_profile_config()
        config['sample_size'] = 100
        result = generate_profile_report(self.dataset, config)
        
        self.assertTrue(result)
        self.assertEqual(self.dataset.profile_status, 'ready')

    def test_generate_profile_report_with_invalid_file(self):
        """Test profile generation with invalid file."""
        # Create invalid CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp.write(b'invalid,csv\ndata')
            tmp.flush()
            with open(tmp.name, 'rb') as f:
                self.dataset.file.save('invalid.csv', ContentFile(f.read()))
        
        with self.assertRaises(Exception):
            generate_profile_report(self.dataset)
        
        self.assertEqual(self.dataset.profile_status, 'failed')

    def test_generate_profile_report_with_all_options(self):
        """Test profile generation with all options enabled."""
        config = {
            'minimal': False,
            'sample_size': -1,  # Use all data
            'correlations': True,
            'duplicates': True,
            'missing_diagrams': True,
            'interactions': None,
            'categorical_maximum_correlation_distinct': 100,
            'plot': {
                'histogram': True,
                'correlation': True,
                'missing': True,
                'image': True,
            },
            'explorative': True,
            'html': {
                'style': {'full_width': True},
                'minify_html': True,
                'use_local_assets': True,
            }
        }
        
        result = generate_profile_report(self.dataset, config)
        self.assertTrue(result)
        self.assertEqual(self.dataset.profile_status, 'ready')

        # Verify JSON report contains all sections
        with open(self.dataset.profile_json.path) as f:
            report_data = json.load(f)
            self.assertIn('correlations', report_data)
            self.assertIn('missing', report_data)
            self.assertIn('duplicates', report_data)

    def test_generate_profile_report_file_cleanup(self):
        """Test that old profile files are properly cleaned up."""
        # Generate initial profile
        config = generate_profile_config(minimal=True)
        generate_profile_report(self.dataset, config)
        
        # Store paths of initial files
        initial_report_path = self.dataset.profile_report.path
        initial_json_path = self.dataset.profile_json.path
        
        # Generate new profile
        generate_profile_report(self.dataset, config)
        
        # Verify old files are deleted
        self.assertFalse(os.path.exists(initial_report_path))
        self.assertFalse(os.path.exists(initial_json_path))
        
        # Verify new files exist
        self.assertTrue(os.path.exists(self.dataset.profile_report.path))
        self.assertTrue(os.path.exists(self.dataset.profile_json.path))

    def test_generate_profile_report_error_handling(self):
        """Test error handling during profile generation."""
        # Test with missing file
        self.dataset.file.delete()
        
        with self.assertRaises(Exception):
            generate_profile_report(self.dataset)
        
        self.assertEqual(self.dataset.profile_status, 'failed')
        
        # Test with invalid configuration
        self.dataset = create_test_dataset()  # Create new dataset with valid file
        invalid_config = {'invalid': 'config'}
        
        with self.assertRaises(Exception):
            generate_profile_report(self.dataset, invalid_config)
        
        self.assertEqual(self.dataset.profile_status, 'failed') 