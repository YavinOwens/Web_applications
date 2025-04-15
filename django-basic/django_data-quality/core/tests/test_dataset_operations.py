import os
import tempfile
import pandas as pd
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from .base import BaseTestCase
from core.models import Dataset, DataQualityAnalysis

class DatasetOperationsTestCase(BaseTestCase):
    def setUp(self):
        super().setUp()
        
        # Create additional test datasets
        self.large_dataset = self.create_large_dataset()
        self.empty_dataset = self.create_empty_dataset()

    def create_large_dataset(self):
        """
        Create a large dataset for performance testing
        """
        temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        large_data = pd.DataFrame({
            'id': range(1, 10001),
            'name': [f'User{i}' for i in range(1, 10001)],
            'age': [25 + (i % 50) for i in range(1, 10001)],
            'salary': [50000 + (i * 100) for i in range(1, 10001)]
        })
        large_data.to_csv(temp_csv.name, index=False)
        temp_csv.close()
        
        return Dataset.objects.create(
            name='Large Test Dataset',
            description='A large dataset for performance testing',
            file=SimpleUploadedFile(
                name='large_test_data.csv', 
                content=open(temp_csv.name, 'rb').read()
            ),
            file_type='csv',
            uploaded_by=self.user,
            total_rows=10000,
            total_columns=4
        )

    def create_empty_dataset(self):
        """
        Create an empty dataset for edge case testing
        """
        return Dataset.objects.create(
            name='Empty Test Dataset',
            description='An empty dataset for edge case testing',
            file=SimpleUploadedFile(
                name='empty_test_data.csv', 
                content=b''
            ),
            file_type='csv',
            uploaded_by=self.user,
            total_rows=0,
            total_columns=0
        )

    def test_dataset_creation(self):
        """
        Test dataset creation with valid data
        """
        self.assertIsNotNone(self.dataset)
        self.assertEqual(self.dataset.name, 'Test Dataset')
        self.assertEqual(self.dataset.total_rows, 10)
        self.assertEqual(self.dataset.total_columns, 3)

    def test_dataset_preview(self):
        """
        Test dataset preview functionality
        """
        preview = self.dataset.get_preview_data()
        self.assertIsNotNone(preview)
        self.assertEqual(len(preview), 10)  # All rows
        self.assertEqual(list(preview.columns), ['id', 'name', 'age', 'salary'])

    def test_dataset_analysis(self):
        """
        Test dataset analysis generation
        """
        analysis, created = DataQualityAnalysis.objects.get_or_create(
            dataset=self.dataset
        )
        
        self.assertIsNotNone(analysis)
        self.assertTrue(created)  # First time creation
        
        # Verify numeric stats
        self.assertIn('age', analysis.numeric_stats)
        self.assertIn('salary', analysis.numeric_stats)

    def test_large_dataset_performance(self):
        """
        Test performance of operations on a large dataset
        """
        # Measure time to read dataset
        import time
        start_time = time.time()
        preview = self.large_dataset.get_preview_data()
        read_time = time.time() - start_time
        
        self.assertIsNotNone(preview)
        self.assertEqual(len(preview), 10000)
        self.assertLess(read_time, 5)  # Should read 10,000 rows in less than 5 seconds

    def test_empty_dataset_handling(self):
        """
        Test handling of an empty dataset
        """
        preview = self.empty_dataset.get_preview_data()
        self.assertIsNone(preview)
        
        # Attempt to generate profile
        profile = self.empty_dataset.generate_profile()
        self.assertIsNone(profile)

    def test_dataset_upload_view(self):
        """
        Test dataset upload view
        """
        self.client.login(username='testuser', password='testpassword')
        
        # Prepare a test CSV file
        test_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        test_data = pd.DataFrame({
            'name': ['John', 'Jane', 'Alice'],
            'age': [30, 25, 35],
            'city': ['New York', 'London', 'Paris']
        })
        test_data.to_csv(test_csv.name, index=False)
        test_csv.close()
        
        with open(test_csv.name, 'rb') as file:
            response = self.client.post(
                reverse('core:dataset_upload'), 
                {
                    'name': 'New Test Dataset',
                    'description': 'A test dataset for upload',
                    'file': file,
                    'file_type': 'csv'
                }
            )
        
        # Clean up test file
        os.unlink(test_csv.name)
        
        # Check response
        self.assertEqual(response.status_code, 302)  # Redirect after successful upload
        
        # Verify dataset was created
        new_dataset = Dataset.objects.filter(name='New Test Dataset').first()
        self.assertIsNotNone(new_dataset)
        self.assertEqual(new_dataset.total_rows, 3)
        self.assertEqual(new_dataset.total_columns, 3)

    def tearDown(self):
        # Clean up large and empty datasets
        if hasattr(self, 'large_dataset'):
            self.large_dataset.delete()
        if hasattr(self, 'empty_dataset'):
            self.empty_dataset.delete()
        
        super().tearDown() 