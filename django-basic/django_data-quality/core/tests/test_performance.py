import os
import time
import random
import pandas as pd
import numpy as np
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from django.conf import settings

from ..models import Dataset, DataGovernanceRule, DataQualityAnalysis
from ..tasks import generate_profile_report, analyze_dataset
from ..utils import validate_dataset_rules

class PerformanceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        """Create large test datasets for performance testing."""
        super().setUpClass()
        
        # Create small dataset
        small_data = pd.DataFrame({
            'id': range(1000),
            'age': np.random.randint(18, 65, 1000),
            'salary': np.random.normal(50000, 15000, 1000),
            'department': random.choices(['IT', 'HR', 'Finance', 'Marketing'], k=1000)
        })
        small_csv_path = '/tmp/small_dataset.csv'
        small_data.to_csv(small_csv_path, index=False)
        
        # Create large dataset
        large_data = pd.DataFrame({
            'id': range(100000),
            'age': np.random.randint(18, 65, 100000),
            'salary': np.random.normal(50000, 15000, 100000),
            'department': random.choices(['IT', 'HR', 'Finance', 'Marketing'], k=100000)
        })
        large_csv_path = '/tmp/large_dataset.csv'
        large_data.to_csv(large_csv_path, index=False)
        
        cls.small_dataset_file = SimpleUploadedFile(
            'small_dataset.csv', 
            open(small_csv_path, 'rb').read(), 
            content_type='text/csv'
        )
        
        cls.large_dataset_file = SimpleUploadedFile(
            'large_dataset.csv', 
            open(large_csv_path, 'rb').read(), 
            content_type='text/csv'
        )

    def test_small_dataset_profile_generation_performance(self):
        """Test profile generation performance for a small dataset."""
        dataset = Dataset.objects.create(
            name=f'Small Performance Test {timezone.now().timestamp()}',
            file=self.small_dataset_file,
            file_type='csv'
        )
        
        start_time = time.time()
        generate_profile_report(dataset)
        end_time = time.time()
        
        # Profile generation should complete within 5 seconds
        self.assertLess(end_time - start_time, 5, 
            f"Profile generation took too long: {end_time - start_time} seconds")
        
        # Verify profile was generated successfully
        dataset.refresh_from_db()
        self.assertEqual(dataset.profile_status, 'completed')

    def test_large_dataset_profile_generation_performance(self):
        """Test profile generation performance for a large dataset."""
        dataset = Dataset.objects.create(
            name=f'Large Performance Test {timezone.now().timestamp()}',
            file=self.large_dataset_file,
            file_type='csv'
        )
        
        start_time = time.time()
        generate_profile_report(dataset)
        end_time = time.time()
        
        # Profile generation for large dataset should complete within 30 seconds
        self.assertLess(end_time - start_time, 30, 
            f"Large dataset profile generation took too long: {end_time - start_time} seconds")
        
        # Verify profile was generated successfully
        dataset.refresh_from_db()
        self.assertEqual(dataset.profile_status, 'completed')

    def test_dataset_analysis_performance(self):
        """Test dataset analysis performance."""
        dataset = Dataset.objects.create(
            name=f'Analysis Performance Test {timezone.now().timestamp()}',
            file=self.large_dataset_file,
            file_type='csv'
        )
        
        start_time = time.time()
        analysis = analyze_dataset(
            dataset, 
            include_correlations=True, 
            include_missing_analysis=True, 
            include_outliers=True
        )
        end_time = time.time()
        
        # Analysis should complete within 10 seconds
        self.assertLess(end_time - start_time, 10, 
            f"Dataset analysis took too long: {end_time - start_time} seconds")
        
        # Verify analysis was created successfully
        self.assertIsNotNone(analysis)
        self.assertTrue(hasattr(analysis, 'correlation_matrix'))
        self.assertTrue(hasattr(analysis, 'missing_analysis'))
        self.assertTrue(hasattr(analysis, 'outliers'))

    def test_concurrent_rule_validation_performance(self):
        """Test performance of concurrent rule validations."""
        # Create multiple datasets with rules
        datasets = []
        for i in range(5):
            dataset = Dataset.objects.create(
                name=f'Concurrent Rule Test {i} {timezone.now().timestamp()}',
                file=self.small_dataset_file,
                file_type='csv'
            )
            
            # Create multiple rules for each dataset
            DataGovernanceRule.objects.create(
                name=f'Age Rule {i}',
                dataset=dataset,
                rule_type='range',
                column_name='age',
                parameters={'min': 18, 'max': 65}
            )
            
            DataGovernanceRule.objects.create(
                name=f'Salary Rule {i}',
                dataset=dataset,
                rule_type='range',
                column_name='salary',
                parameters={'min': 30000, 'max': 100000}
            )
            
            datasets.append(dataset)
        
        # Simulate concurrent rule validations
        start_time = time.time()
        for dataset in datasets:
            rules = DataGovernanceRule.objects.filter(dataset=dataset)
            validate_dataset_rules(dataset, rules)
        end_time = time.time()
        
        # Concurrent rule validations should complete within 10 seconds
        self.assertLess(end_time - start_time, 10, 
            f"Concurrent rule validations took too long: {end_time - start_time} seconds")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        super().tearDownClass()
        try:
            os.remove('/tmp/small_dataset.csv')
            os.remove('/tmp/large_dataset.csv')
        except Exception:
            pass 