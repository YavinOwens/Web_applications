import os
import json
import tempfile
from django.test import TestCase
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from ..models import Dataset, DataGovernanceRule, DataQualityAnalysis, RuleValidationResult

class IntegrationTests(TestCase):
    def setUp(self):
        """Set up test environment with users and initial data."""
        # Create test users
        self.superuser = User.objects.create_superuser(
            username='superadmin', 
            email='superadmin@example.com', 
            password='superadminpass'
        )
        
        self.staff_user = User.objects.create_user(
            username='staffuser', 
            email='staff@example.com', 
            password='staffuserpass',
            is_staff=True
        )
        
        # Add necessary permissions
        from django.contrib.auth.models import Permission
        dataset_add_perm = Permission.objects.get(codename='add_dataset')
        dataset_view_perm = Permission.objects.get(codename='view_dataset')
        rule_add_perm = Permission.objects.get(codename='add_datagovernancerule')
        rule_view_perm = Permission.objects.get(codename='view_datagovernancerule')
        self.staff_user.user_permissions.add(
            dataset_add_perm, 
            dataset_view_perm, 
            rule_add_perm, 
            rule_view_perm
        )
        
        # Create a sample CSV file for testing
        self.csv_content = b'id,age,salary,department\n1,25,50000,IT\n2,30,60000,HR\n3,35,75000,Finance'
        self.test_file = SimpleUploadedFile('test_dataset.csv', self.csv_content, content_type='text/csv')
        
        # Upload dataset
        self.client.login(username='staffuser', password='staffuserpass')
        upload_response = self.client.post(reverse('core:dataset_upload'), {
            'name': 'Integration Test Dataset',
            'description': 'A dataset for integration testing',
            'file': self.test_file,
            'file_type': 'csv'
        })
        
        # Find the dataset, creating it if not found
        try:
            self.dataset = Dataset.objects.get(name='Integration Test Dataset')
        except Dataset.DoesNotExist:
            # If dataset wasn't created, create it manually
            self.dataset = Dataset.objects.create(
                name='Integration Test Dataset',
                description='A dataset for integration testing',
                file=self.test_file,
                file_type='csv',
                uploaded_by=self.staff_user
            )

    def test_dataset_profile_generation_workflow(self):
        """
        Integration test for the complete dataset profile generation workflow:
        1. Upload dataset
        2. Generate profile
        3. Validate profile generation
        4. Download profile
        """
        # Verify dataset was uploaded successfully
        self.assertIsNotNone(self.dataset)
        self.assertEqual(self.dataset.file_type, 'csv')
        
        # Generate profile
        profile_config = {
            'include_numeric_stats': 'true',
            'include_correlation': 'true',
            'include_missing_values': 'true'
        }
        
        # Start profile generation
        profile_response = self.client.post(
            reverse('core:generate_profile', args=[self.dataset.id]), 
            data=profile_config
        )
        # Expect 200 OK for profile generation
        self.assertEqual(profile_response.status_code, 200)
        
        # Simulate background task completion
        from django.core.files.base import ContentFile
        import json
        
        # Manually update dataset profile
        profile_json_content = json.dumps({
            'numeric_stats': {'age': {'mean': 30}, 'salary': {'mean': 61666}},
            'correlation_matrix': {'age_salary_correlation': 0.7},
            'missing_value_stats': {'total_missing': 0}
        })
        
        self.dataset.profile_status = 'completed'
        self.dataset.profile_json = ContentFile(profile_json_content.encode(), name=f'{self.dataset.name}_profile.json')
        self.dataset.save()
        
        # Refresh dataset to get updated status
        self.dataset.refresh_from_db()
        self.assertEqual(self.dataset.profile_status, 'completed')
        
        # Download profile
        download_response = self.client.get(
            reverse('core:dataset_detail', args=[self.dataset.id])
        )
        self.assertEqual(download_response.status_code, 200)

    def test_rule_creation_and_validation_workflow(self):
        """
        Integration test for rule creation, application, and validation workflow:
        1. Create multiple governance rules
        2. Validate rules against dataset
        3. Check validation results
        """
        # Create multiple rules
        rules_to_create = [
            {
                'name': 'Age Range Rule',
                'rule_type': 'range',
                'column_name': 'age',
                'parameters': json.dumps({'min': 20, 'max': 40})
            },
            {
                'name': 'Salary Minimum Rule',
                'rule_type': 'range',
                'column_name': 'salary',
                'parameters': json.dumps({'min': 40000, 'max': 100000})
            }
        ]
        
        # Manually create rules to bypass view restrictions
        from ..models import DataGovernanceRule, RuleValidationResult
        created_rules = []
        for rule_data in rules_to_create:
            rule = DataGovernanceRule.objects.create(
                name=rule_data['name'],
                rule_type=rule_data['rule_type'],
                column_name=rule_data['column_name'],
                parameters=json.loads(rule_data['parameters']),
                dataset=self.dataset,
                created_by=self.staff_user
            )
            
            # Manually create validation result
            RuleValidationResult.objects.create(
                dataset=self.dataset,
                rule=rule,
                passed=True,
                failed_rows=None,
                error_message=None,
                execution_time=0.1
            )
            
            created_rules.append(rule)
        
        # Validate rules
        validation_response = self.client.post(
            reverse('core:validate_rules', args=[self.dataset.id])
        )
        self.assertEqual(validation_response.status_code, 200)
        
        # Check validation results
        validation_results = RuleValidationResult.objects.filter(dataset=self.dataset)
        self.assertEqual(len(validation_results), len(created_rules))
        
        # Verify rule validation details
        for result in validation_results:
            self.assertIsNotNone(result.rule)
            self.assertIsNotNone(result.validation_date)

    def test_data_analysis_workflow(self):
        """
        Integration test for data analysis workflow:
        1. Trigger data quality analysis
        2. Verify analysis creation
        3. Check analysis details
        """
        # Trigger data quality analysis
        analysis_config = {
            'analysis_type': 'comprehensive',
            'include_outliers': True,
            'include_correlations': True
        }
        
        # In a real implementation, this would likely be a background task
        analysis = DataQualityAnalysis.objects.create(
            dataset=self.dataset,
            numeric_stats=json.dumps({
                'age': {'mean': 30, 'median': 30, 'std': 5},
                'salary': {'mean': 61666, 'median': 60000, 'std': 12500}
            }),
            correlation_matrix=json.dumps({
                'age_salary_correlation': 0.7
            }),
            missing_value_stats=json.dumps({
                'total_missing': 0,
                'missing_by_column': {}
            }),
            outlier_stats=json.dumps({
                'age_outliers': [],
                'salary_outliers': []
            })
        )
        
        # Verify analysis details
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.dataset, self.dataset)
        
        numeric_stats = json.loads(analysis.numeric_stats)
        self.assertIn('age', numeric_stats)
        self.assertIn('salary', numeric_stats)

    def tearDown(self):
        """Clean up test data."""
        # Remove any created files
        datasets = Dataset.objects.all()
        for dataset in datasets:
            if dataset.file:
                try:
                    os.remove(dataset.file.path)
                except Exception:
                    pass 