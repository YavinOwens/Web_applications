from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from ..models import Dataset, DataQualityAnalysis, DataGovernanceRule
from .utils import create_test_dataset, generate_profile_config
import json
import pandas as pd
import tempfile
import os
from django.utils import timezone

class ViewsTests(TestCase):
    def setUp(self):
        self.client = Client()
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

    def test_dataset_list_view(self):
        """Test the dataset list view."""
        response = self.client.get(reverse('core:dataset_list'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/dataset_list.html')
        self.assertContains(response, self.dataset.name)

    def test_dataset_upload_view(self):
        """Test the dataset upload view."""
        # Create a test file
        file_content = b'id,name,value\n1,test,10\n2,test2,20'
        test_file = SimpleUploadedFile('test.csv', file_content, content_type='text/csv')
        
        # Test GET request
        response = self.client.get(reverse('core:dataset_upload'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/dataset_upload.html')
        
        # Test POST request with valid data
        data = {
            'name': f'Test Dataset {timezone.now().timestamp()}',
            'description': 'A test dataset',
            'file': test_file,
            'file_type': 'csv'
        }
        response = self.client.post(
            reverse('core:dataset_upload'),
            data,
            format='multipart/form-data'
        )
        
        # Check if there are any form errors
        if response.status_code != 302:
            print("Form errors:", response.context['form'].errors if 'form' in response.context else 'No form in context')
        
        # Verify redirect to dataset detail page
        self.assertEqual(response.status_code, 302)
        
        # Get the created dataset
        dataset = Dataset.objects.get(name=data['name'])
        
        # Verify redirect URL
        self.assertEqual(response.url, reverse('core:dataset_detail', args=[dataset.id]))
        
        # Verify dataset was created with correct data
        self.assertEqual(dataset.description, data['description'])
        self.assertEqual(dataset.file_type, data['file_type'])

    def test_dataset_detail_view(self):
        """Test the dataset detail view."""
        response = self.client.get(reverse('core:dataset_detail', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/dataset_detail.html')
        self.assertContains(response, self.dataset.name)

    def test_dataset_delete_view(self):
        """Test the dataset delete view."""
        response = self.client.post(reverse('core:dataset_delete', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 302)  # Redirect after deletion
        self.assertFalse(Dataset.objects.filter(id=self.dataset.id).exists())

    def test_generate_profile_view_get(self):
        """Test GET request to generate profile view."""
        response = self.client.get(reverse('core:generate_profile', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/profile_config.html')

    def test_generate_profile_view_post(self):
        """Test POST request to generate profile view."""
        config = generate_profile_config(minimal=True)
        response = self.client.post(
            reverse('core:generate_profile', args=[self.dataset.id]),
            config
        )
        self.assertEqual(response.status_code, 302)  # Redirect after success
        self.dataset.refresh_from_db()
        self.assertEqual(self.dataset.profile_status, 'processing')

    def test_profile_status_view(self):
        """Test the profile status view."""
        response = self.client.get(reverse('core:profile_status', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('status', data)
        self.assertIn('last_updated', data)

class DatasetAnalysisTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.dataset = create_test_dataset()

    def tearDown(self):
        # Clean up any files created during tests
        if self.dataset.file:
            if os.path.exists(self.dataset.file.path):
                os.remove(self.dataset.file.path)

    def test_dataset_analyze_view_get(self):
        """Test GET request to dataset analyze view."""
        response = self.client.get(reverse('core:dataset_analyze', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/analyze_dataset.html')

    def test_dataset_analyze_view_post_success(self):
        """Test successful POST request to dataset analyze view."""
        data = {
            'analyze_correlations': True,
            'include_missing_analysis': True,
            'detect_outliers': True
        }
        response = self.client.post(
            reverse('core:dataset_analyze', args=[self.dataset.id]),
            data
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/analysis_results.html')
        
        # Verify analysis was created with correct settings
        analysis = DataQualityAnalysis.objects.filter(dataset=self.dataset).latest('created_at')
        self.assertIsNotNone(analysis.correlation_matrix)
        self.assertIsNotNone(analysis.missing_value_stats)
        self.assertIsNotNone(analysis.outlier_stats)

    def test_dataset_analyze_view_post_error(self):
        """Test POST request to dataset analyze view with error."""
        # Don't include any fields to make the form invalid
        data = {}
        response = self.client.post(
            reverse('core:dataset_analyze', args=[self.dataset.id]),
            data
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/analyze_dataset.html')
        self.assertFalse(response.context['form'].is_valid())
        self.assertContains(response, 'Please correct the errors below')

    def test_dataset_analyze_invalid_dataset(self):
        """Test analysis request for non-existent dataset."""
        response = self.client.get(reverse('core:dataset_analyze', args=[999]))
        self.assertEqual(response.status_code, 404)

    def test_dataset_analyze_correlations_only(self):
        """Test analysis with only correlations enabled."""
        data = {
            'analyze_correlations': True,
            'include_missing_analysis': False,
            'detect_outliers': False
        }
        response = self.client.post(
            reverse('core:dataset_analyze', args=[self.dataset.id]),
            data
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/analysis_results.html')
        
        # Verify analysis was created with correct settings
        analysis = DataQualityAnalysis.objects.filter(dataset=self.dataset).latest('created_at')
        self.assertIsNotNone(analysis.correlation_matrix)
        self.assertIsNone(analysis.missing_value_stats)
        self.assertIsNone(analysis.outlier_stats)

    def test_dataset_analyze_missing_analysis_only(self):
        """Test analysis with only missing value analysis enabled."""
        data = {
            'analyze_correlations': False,
            'include_missing_analysis': True,
            'detect_outliers': False
        }
        response = self.client.post(
            reverse('core:dataset_analyze', args=[self.dataset.id]),
            data
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/analysis_results.html')
        
        # Verify analysis was created with correct settings
        analysis = DataQualityAnalysis.objects.filter(dataset=self.dataset).latest('created_at')
        self.assertIsNone(analysis.correlation_matrix)
        self.assertIsNotNone(analysis.missing_value_stats)
        self.assertIsNone(analysis.outlier_stats)

    def test_dataset_analyze_outliers_only(self):
        """Test analysis with only outlier detection enabled."""
        data = {
            'analyze_correlations': False,
            'include_missing_analysis': False,
            'detect_outliers': True
        }
        response = self.client.post(
            reverse('core:dataset_analyze', args=[self.dataset.id]),
            data
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/analysis_results.html')
        
        # Verify analysis was created with correct settings
        analysis = DataQualityAnalysis.objects.filter(dataset=self.dataset).latest('created_at')
        self.assertIsNone(analysis.correlation_matrix)
        self.assertIsNone(analysis.missing_value_stats)
        self.assertIsNotNone(analysis.outlier_stats)

class ValidationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.dataset = create_test_dataset()
        self.rule = DataGovernanceRule.objects.create(
            name='Test Rule',
            dataset=self.dataset,
            rule_type='format',
            column_name='test',
            parameters={'pattern': r'\d+'}
        )

    def tearDown(self):
        if self.dataset.file:
            if os.path.exists(self.dataset.file.path):
                os.remove(self.dataset.file.path)

    def test_validate_rules_view_get(self):
        """Test GET request to validate rules view."""
        response = self.client.get(reverse('core:validate_rules', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/validate_rules.html')

    def test_validate_rules_view_post(self):
        """Test POST request to validate rules view."""
        response = self.client.post(reverse('core:validate_rules', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/validate_rules.html')
        self.assertIn('validation_results', response.context)

    def test_validation_results_view(self):
        """Test the validation results view."""
        response = self.client.get(reverse('core:validation_results'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/validation_results.html') 

class ErrorHandlingTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.dataset = create_test_dataset()

    def tearDown(self):
        # Clean up any files created during tests
        if self.dataset.file:
            if os.path.exists(self.dataset.file.path):
                os.remove(self.dataset.file.path)

    def test_dataset_upload_invalid_file_type(self):
        """Test uploading a dataset with an unsupported file type."""
        # Create an unsupported file type
        file_content = b'{"invalid": "json"}'
        test_file = SimpleUploadedFile('test.txt', file_content, content_type='text/plain')
        
        data = {
            'name': f'Invalid File Type Dataset {timezone.now().timestamp()}',
            'description': 'A dataset with an invalid file type',
            'file': test_file,
            'file_type': 'txt'  # Unsupported file type
        }
        
        response = self.client.post(
            reverse('core:dataset_upload'),
            data,
            format='multipart/form-data'
        )
        
        # Expect the form to be invalid
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['form'].is_valid())
        self.assertIn('file_type', response.context['form'].errors)

    def test_dataset_upload_empty_file(self):
        """Test uploading an empty dataset file."""
        # Create an empty file
        test_file = SimpleUploadedFile('empty.csv', b'', content_type='text/csv')
        
        data = {
            'name': f'Empty Dataset {timezone.now().timestamp()}',
            'description': 'An empty dataset',
            'file': test_file,
            'file_type': 'csv'
        }
        
        response = self.client.post(
            reverse('core:dataset_upload'),
            data,
            format='multipart/form-data'
        )
        
        # Expect the form to be invalid
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['form'].is_valid())
        self.assertIn('file', response.context['form'].errors)

    def test_dataset_upload_oversized_file(self):
        """Test uploading a dataset file that exceeds size limit."""
        # Create a large file (simulate oversized file)
        large_content = b'x' * (10 * 1024 * 1024)  # 10 MB file
        test_file = SimpleUploadedFile('large.csv', large_content, content_type='text/csv')
        
        data = {
            'name': f'Oversized Dataset {timezone.now().timestamp()}',
            'description': 'An oversized dataset',
            'file': test_file,
            'file_type': 'csv'
        }
        
        response = self.client.post(
            reverse('core:dataset_upload'),
            data,
            format='multipart/form-data'
        )
        
        # Expect the form to be invalid
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['form'].is_valid())
        self.assertIn('file', response.context['form'].errors)

    def test_generate_profile_invalid_config(self):
        """Test generating a profile with an invalid configuration."""
        # Create an invalid profile configuration
        invalid_config = {
            'invalid_key': 'invalid_value'
        }
        
        response = self.client.post(
            reverse('core:generate_profile', args=[self.dataset.id]),
            invalid_config
        )
        
        # Expect the form to be invalid
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['form'].is_valid())
        self.assertIn('__all__', response.context['form'].errors)

    def test_dataset_analyze_invalid_configuration(self):
        """Test dataset analysis with invalid configuration."""
        # Create an invalid analysis configuration
        invalid_data = {
            'invalid_key': True
        }
        
        response = self.client.post(
            reverse('core:dataset_analyze', args=[self.dataset.id]),
            invalid_data
        )
        
        # Expect the form to be invalid
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['form'].is_valid())
        self.assertContains(response, 'Please correct the errors below')

    def test_validate_rules_nonexistent_dataset(self):
        """Test rule validation for a non-existent dataset."""
        response = self.client.post(
            reverse('core:validate_rules', args=[999999])  # Non-existent dataset ID
        )
        
        # Expect a 404 error
        self.assertEqual(response.status_code, 404)

    def test_download_profile_invalid_format(self):
        """Test downloading a profile with an invalid format."""
        response = self.client.get(
            reverse('core:download_profile', args=[self.dataset.id, 'invalid_format'])
        )
        
        # Expect a 400 Bad Request
        self.assertEqual(response.status_code, 400)

    def test_profile_config_invalid_json(self):
        """Test updating profile configuration with invalid JSON."""
        response = self.client.post(
            reverse('core:profile_config', args=[self.dataset.id]),
            content_type='application/json',
            data='invalid json'
        )
        
        # Expect a 400 Bad Request
        self.assertEqual(response.status_code, 400) 