from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from ..models import Dataset, DataGovernanceRule, RuleValidationResult
import pandas as pd
import tempfile
import json
import os
import shutil

class ComprehensiveTestCase(TestCase):
    def setUp(self):
        """Set up test environment."""
        self.client = Client()
        
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')
        
        # Create test dataset
        self.create_test_dataset()
        
        # Create test rule
        self.create_test_rule()

    def create_test_dataset(self):
        """Helper method to create a test dataset."""
        # Create a test CSV file
        df = pd.DataFrame({
            'id': range(1, 6),
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 
                     'david@test.com', 'eve@test.com']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            tmp.seek(0)
            
            self.dataset = Dataset.objects.create(
                name='Test Dataset',
                file=SimpleUploadedFile('test.csv', tmp.read()),
                file_type='csv',
                created_by=self.user
            )
        
        # Clean up the temporary file
        os.unlink(tmp.name)

    def create_test_rule(self):
        """Helper method to create a test rule."""
        self.rule = DataGovernanceRule.objects.create(
            name='Test Rule',
            description='Test rule for validation',
            rule_type='format',
            dataset=self.dataset,
            column_name='email',
            parameters={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
            created_by=self.user
        )

    def test_ajax_endpoints(self):
        """Test AJAX endpoints for dynamic updates."""
        # Test get_dataset_columns endpoint
        response = self.client.get(
            reverse('core:get_dataset_columns', args=[self.dataset.id]),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(response.status_code, 200)
        columns = json.loads(response.content)
        self.assertIn('email', columns)
        
        # Test grid_data_api endpoint
        response = self.client.get(
            reverse('core:grid_data_api', args=[self.dataset.id]),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'success')
        self.assertTrue('columnDefs' in data)
        self.assertTrue('rowData' in data)
        
        # Test profile_status endpoint
        response = self.client.get(
            reverse('core:profile_status', args=[self.dataset.id]),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(response.status_code, 200)
        status = json.loads(response.content)
        self.assertTrue('status' in status)

    def test_advanced_filtering(self):
        """Test advanced filtering in dataset list view."""
        # Test sorting by creation date
        response = self.client.get(
            reverse('core:dataset_list'),
            {'sort': '-created_at'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/dataset_list.html')
        
        # Test filtering by name
        response = self.client.get(
            reverse('core:dataset_list'),
            {'search': 'Test Dataset'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Dataset')
        
        # Test combined filtering and sorting
        response = self.client.get(
            reverse('core:dataset_list'),
            {'search': 'Test', 'sort': '-name'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Dataset')

    def test_complete_workflow(self):
        """Test complete workflow from dataset upload to validation."""
        # Test dataset upload
        with open(self.dataset.file.path, 'rb') as file:
            response = self.client.post(
                reverse('core:dataset_upload'),
                {
                    'name': 'New Dataset',
                    'description': 'Test description',
                    'file_type': 'csv',
                    'file': file
                }
            )
        self.assertEqual(response.status_code, 302)  # Redirect after success
        
        # Test rule creation
        response = self.client.post(
            reverse('core:rule_create'),
            {
                'name': 'New Rule',
                'description': 'Test rule description',
                'rule_type': 'format',
                'dataset': self.dataset.id,
                'column_name': 'email',
                'parameters': json.dumps({'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'})
            }
        )
        self.assertEqual(response.status_code, 302)  # Redirect after success
        
        # Test rule validation
        response = self.client.post(
            reverse('core:validate_rules', args=[self.dataset.id])
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify validation results
        validation_results = RuleValidationResult.objects.filter(
            dataset=self.dataset,
            rule=self.rule
        )
        self.assertTrue(validation_results.exists())

    def test_api_security(self):
        """Test API endpoint security."""
        # Test unauthorized access
        self.client.logout()
        
        # Test grid_data_api endpoint
        response = self.client.get(
            reverse('core:grid_data_api', args=[self.dataset.id]),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(response.status_code, 302)  # Redirect to login
        
        # Test with invalid session
        response = self.client.get(
            reverse('core:grid_data_api', args=[self.dataset.id]),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
            HTTP_AUTHORIZATION='Invalid'
        )
        self.assertEqual(response.status_code, 302)  # Redirect to login
        
        # Test accessing non-existent dataset
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(
            reverse('core:grid_data_api', args=[99999]),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(response.status_code, 404)

    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test invalid file upload
        response = self.client.post(
            reverse('core:dataset_upload'),
            {
                'name': 'Invalid Dataset',
                'description': 'Test description',
                'file_type': 'csv',
                'file': SimpleUploadedFile('test.txt', b'invalid data')
            }
        )
        self.assertEqual(response.status_code, 200)  # Return to form with errors
        
        # Test missing required fields in rule creation
        response = self.client.post(
            reverse('core:rule_create'),
            {}
        )
        self.assertEqual(response.status_code, 200)  # Return to form with errors
        
        # Test invalid rule parameters
        response = self.client.post(
            reverse('core:rule_create'),
            {
                'name': 'Invalid Rule',
                'description': 'Test rule description',
                'rule_type': 'format',
                'dataset': self.dataset.id,
                'column_name': 'email',
                'parameters': 'invalid json'
            }
        )
        self.assertEqual(response.status_code, 200)  # Return to form with errors
        
        # Test validation with missing dataset
        response = self.client.post(
            reverse('core:validate_rules', args=[99999])
        )
        self.assertEqual(response.status_code, 404)

    def test_bulk_validation(self):
        """Test bulk validation of multiple rules."""
        # Create additional rules
        rule2 = DataGovernanceRule.objects.create(
            name='Age Rule',
            description='Test age validation',
            rule_type='range',
            dataset=self.dataset,
            column_name='age',
            parameters={'min': 0, 'max': 120},
            created_by=self.user
        )
        
        # Test bulk validation
        response = self.client.post(
            reverse('core:bulk_validate_rules', args=[self.dataset.id])
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify results for both rules
        validation_results = RuleValidationResult.objects.filter(dataset=self.dataset)
        self.assertEqual(validation_results.count(), 2)
        
        # Check individual results
        email_validation = validation_results.get(rule=self.rule)
        age_validation = validation_results.get(rule=rule2)
        self.assertTrue(email_validation.passed)  # All test emails are valid
        self.assertTrue(age_validation.passed)    # All test ages are within range

    def test_export_validation_results(self):
        """Test exporting validation results."""
        # Create validation result
        validation_result = RuleValidationResult.objects.create(
            dataset=self.dataset,
            rule=self.rule,
            passed=False,
            failed_rows=[
                {'row': 1, 'value': 'invalid@email', 'error': 'Invalid email format'},
                {'row': 2, 'value': 'another@invalid', 'error': 'Invalid email format'}
            ],
            validated_by=self.user
        )
        
        # Test export endpoint
        response = self.client.post(
            reverse('core:export_validation_results'),
            json.dumps({
                'ruleId': self.rule.id,
                'validationId': validation_result.id
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')

    def test_row_details_api(self):
        """Test getting detailed information about specific rows."""
        # Test valid row request
        response = self.client.get(
            reverse('core:get_row_details', args=[0]),
            {'dataset_id': self.dataset.id}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue('row_data' in data)
        self.assertTrue('metadata' in data)
        
        # Test invalid row number
        response = self.client.get(
            reverse('core:get_row_details', args=[999]),
            {'dataset_id': self.dataset.id}
        )
        self.assertEqual(response.status_code, 400)
        
        # Test missing dataset_id
        response = self.client.get(
            reverse('core:get_row_details', args=[0])
        )
        self.assertEqual(response.status_code, 400)

    def test_profile_generation(self):
        """Test profile generation workflow."""
        # Test profile generation request
        response = self.client.post(
            reverse('core:generate_profile', args=[self.dataset.id])
        )
        self.assertEqual(response.status_code, 302)  # Redirect after triggering
        
        # Test profile status endpoint
        response = self.client.get(
            reverse('core:profile_status', args=[self.dataset.id])
        )
        self.assertEqual(response.status_code, 200)
        status = json.loads(response.content)
        self.assertTrue('status' in status)
        
        # Test YData profile generation
        response = self.client.post(
            reverse('core:generate_ydata_profile', args=[self.dataset.id])
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue('status' in data)

    def test_documentation_access(self):
        """Test documentation page access and content."""
        response = self.client.get(reverse('core:documentation'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/documentation.html')
        
        # Test documentation content
        self.assertIn('documentation_content', response.context)
        
        # Test documentation when file is missing
        if os.path.exists(os.path.join(settings.BASE_DIR, 'core', 'docs', 'source')):
            shutil.rmtree(os.path.join(settings.BASE_DIR, 'core', 'docs', 'source'))
        
        response = self.client.get(reverse('core:documentation'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('Documentation Not Found', response.context['documentation_content'])

    def test_validation_dashboard(self):
        """Test validation dashboard functionality."""
        # Create some validation results
        RuleValidationResult.objects.create(
            dataset=self.dataset,
            rule=self.rule,
            passed=True,
            validated_by=self.user
        )
        
        # Test dashboard access
        response = self.client.get(reverse('core:validation_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'core/validation_dashboard.html')
        
        # Verify context data
        self.assertTrue('validation_stats' in response.context)
        self.assertTrue('recent_results' in response.context)
        self.assertTrue('rules_with_status' in response.context)
        
        # Test dashboard with no results
        RuleValidationResult.objects.all().delete()
        response = self.client.get(reverse('core:validation_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total_results'], 0)

    def tearDown(self):
        """Clean up test environment."""
        # Clean up uploaded files
        if self.dataset.file:
            if os.path.exists(self.dataset.file.path):
                os.remove(self.dataset.file.path)
        
        # Clean up user
        self.user.delete() 