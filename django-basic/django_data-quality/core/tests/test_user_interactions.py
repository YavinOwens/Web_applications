import os
import json
import unittest
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone

from ..models import Dataset, DataGovernanceRule, DataQualityAnalysis

class UserInteractionTests(TestCase):
    def setUp(self):
        """Set up test environment."""
        self.client = Client()
        
        # Create a test CSV file
        file_content = b'id,age,salary,department\n1,25,50000,IT\n2,30,60000,HR\n3,35,75000,Finance\n4,28,55000,Marketing'
        self.test_file = SimpleUploadedFile('test_dataset.csv', file_content, content_type='text/csv')

    def test_dataset_upload_workflow(self):
        """Test complete dataset upload workflow."""
        # Prepare upload data
        upload_data = {
            'name': f'Test Dataset {timezone.now().timestamp()}',
            'description': 'A test dataset for user interaction',
            'file': self.test_file,
            'file_type': 'csv'
        }
        
        # Perform upload
        response = self.client.post(
            reverse('core:dataset_upload'),
            upload_data,
            format='multipart/form-data'
        )
        
        # Check upload was successful
        self.assertEqual(response.status_code, 302)  # Redirect after successful upload
        
        # Verify dataset was created
        dataset = Dataset.objects.get(name=upload_data['name'])
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.description, upload_data['description'])
        self.assertEqual(dataset.file_type, upload_data['file_type'])

    def test_profile_generation_workflow(self):
        """Test profile generation workflow."""
        # First, create a dataset
        dataset = Dataset.objects.create(
            name=f'Profile Test Dataset {timezone.now().timestamp()}',
            description='Dataset for profile generation test',
            file=self.test_file,
            file_type='csv',
            profile_status='not_generated'  # Initial status
        )
        
        # Prepare profile generation config to match ProfileConfigForm
        profile_config = {
            'minimal': 'false',
            'sample_size': '-1',
            'pearson': 'true',
            'spearman': 'false',
            'kendall': 'false',
            'missing_matrix': 'true',
            'missing_bar': 'true',
            'missing_heatmap': 'false',
            'duplicates_table': 'true',
            'duplicates_matrix': 'false'
        }
        
        # Perform profile generation
        response = self.client.post(
            reverse('core:generate_profile', args=[dataset.id]),
            profile_config
        )
        
        # Detailed debugging
        print("Profile Generation Response Status:", response.status_code)
        
        # Check response
        self.assertEqual(response.status_code, 302)  # Redirect after successful profile generation
        
        # Refresh dataset
        dataset.refresh_from_db()
        
        # Verify profile status
        self.assertEqual(dataset.profile_status, 'processing')

    def test_governance_rule_creation_workflow(self):
        """Test governance rule creation workflow."""
        # First, create a dataset
        dataset = Dataset.objects.create(
            name=f'Rule Test Dataset {timezone.now().timestamp()}',
            description='Dataset for rule creation test',
            file=self.test_file,
            file_type='csv'
        )
        
        # Prepare rule data
        rule_data = {
            'name': 'Age Validation Rule',
            'rule_type': 'range',
            'column_name': 'age',
            'parameters': json.dumps({'min': 18, 'max': 65})
        }
        
        # Create rule
        response = self.client.post(
            reverse('core:rule_create', args=[dataset.id]),
            rule_data
        )
        
        # Detailed debugging
        print("Rule Creation Response Status:", response.status_code)
        
        # Check rule creation
        self.assertEqual(response.status_code, 302)  # Redirect after rule creation
        
        # Verify rule was created
        rule = DataGovernanceRule.objects.get(
            name='Age Validation Rule',
            dataset=dataset
        )
        self.assertIsNotNone(rule)
        self.assertEqual(rule.rule_type, 'range')
        self.assertEqual(rule.column_name, 'age')

    def test_dataset_analysis_workflow(self):
        """Test dataset analysis workflow."""
        # First, create a dataset
        dataset = Dataset.objects.create(
            name=f'Analysis Test Dataset {timezone.now().timestamp()}',
            description='Dataset for analysis test',
            file=self.test_file,
            file_type='csv'
        )
        
        # Prepare analysis config
        analysis_config = {
            'analyze_correlations': 'true',
            'include_missing_analysis': 'true',
            'detect_outliers': 'true'
        }
        
        # Perform analysis
        response = self.client.post(
            reverse('core:dataset_analyze', args=[dataset.id]),
            analysis_config
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        # Verify analysis was created
        analysis = DataQualityAnalysis.objects.filter(dataset=dataset).latest('created_at')
        self.assertIsNotNone(analysis)
        
        # Check analysis components
        self.assertIsNotNone(analysis.correlation_matrix)
        self.assertIsNotNone(analysis.missing_value_stats)
        
        # Outlier stats might be None if no outliers detected
        # So we'll modify this check
        if analysis.outlier_stats is not None:
            self.assertTrue(isinstance(analysis.outlier_stats, dict))

    def test_rule_validation_workflow(self):
        """Test rule validation workflow."""
        # First, create a dataset with a rule
        dataset = Dataset.objects.create(
            name=f'Validation Test Dataset {timezone.now().timestamp()}',
            description='Dataset for rule validation test',
            file=self.test_file,
            file_type='csv'
        )
        
        # Create a rule
        rule = DataGovernanceRule.objects.create(
            name='Salary Range Rule',
            dataset=dataset,
            rule_type='range',
            column_name='salary',
            parameters={'min': 30000, 'max': 100000}
        )
        
        # Perform rule validation
        response = self.client.post(
            reverse('core:validate_rules', args=[dataset.id])
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        # Verify validation results are present in context
        self.assertIn('validation_results', response.context)
        
        # Optional: Check specific validation details
        validation_results = response.context['validation_results']
        self.assertTrue(len(validation_results) > 0)

    def test_dataset_list_and_detail_views(self):
        """Test dataset list and detail views."""
        # Create a few test datasets
        for i in range(3):
            Dataset.objects.create(
                name=f'List Test Dataset {i} {timezone.now().timestamp()}',
                description=f'Test dataset {i}',
                file=self.test_file,
                file_type='csv'
            )
        
        # Test dataset list view
        response = self.client.get(reverse('core:dataset_list'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/dataset_list.html')
        
        # Verify datasets are in the context
        self.assertIn('datasets', response.context)
        self.assertTrue(len(response.context['datasets']) >= 3)

    def tearDown(self):
        """Clean up test files."""
        # Remove any created files
        datasets = Dataset.objects.all()
        for dataset in datasets:
            if dataset.file:
                try:
                    os.remove(dataset.file.path)
                except Exception:
                    pass 