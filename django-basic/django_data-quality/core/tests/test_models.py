from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ValidationError
from ..models import Dataset, DataGovernanceRule, RuleValidationResult, DataQualityAnalysis
from .utils import create_test_dataset, generate_profile_config
import os
import json
import tempfile
import pandas as pd

class DatasetModelTests(TestCase):
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

    def test_dataset_creation(self):
        """Test dataset creation with valid data."""
        self.assertEqual(self.dataset.name, 'Test Dataset')
        self.assertEqual(self.dataset.file_type, 'csv')
        self.assertIsNotNone(self.dataset.file)
        self.assertEqual(self.dataset.profile_status, 'pending')
        self.assertIsNone(self.dataset.profile_report)
        self.assertIsNone(self.dataset.profile_json)
        self.assertIsNone(self.dataset.profile_last_updated)

    def test_dataset_str_representation(self):
        """Test the string representation of a dataset."""
        self.assertEqual(str(self.dataset), 'Test Dataset')

    def test_dataset_file_validation(self):
        """Test file type validation."""
        # Test invalid file type
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            tmp.write(b'invalid data')
            tmp.seek(0)
            with self.assertRaises(ValidationError):
                Dataset.objects.create(
                    name='Invalid Dataset',
                    file_type='txt',
                    file=SimpleUploadedFile('test.txt', tmp.read())
                )

        # Test valid CSV file
        df = pd.DataFrame({'test': range(5)})
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            tmp.seek(0)
            dataset = Dataset.objects.create(
                name='Valid Dataset',
                file_type='csv',
                file=SimpleUploadedFile('test.csv', tmp.read())
            )
            self.assertTrue(os.path.exists(dataset.file.path))

    def test_dataset_profile_management(self):
        """Test profile management functionality."""
        # Test profile status update
        self.dataset.update_profile_status('processing')
        self.assertEqual(self.dataset.profile_status, 'processing')

        # Test profile config update
        config = generate_profile_config()
        self.dataset.update_profile_config(config)
        self.assertEqual(self.dataset.profile_config, config)

        # Test profile report update
        report_content = '<html>Test Report</html>'
        json_content = '{"test": "data"}'
        
        self.dataset.update_profile_files(
            report_content.encode(),
            json_content.encode()
        )
        
        self.assertTrue(os.path.exists(self.dataset.profile_report.path))
        self.assertTrue(os.path.exists(self.dataset.profile_json.path))
        self.assertIsNotNone(self.dataset.profile_last_updated)

    def test_dataset_file_cleanup(self):
        """Test file cleanup on dataset deletion."""
        file_path = self.dataset.file.path
        self.assertTrue(os.path.exists(file_path))
        
        self.dataset.delete()
        self.assertFalse(os.path.exists(file_path))

    def test_dataset_profile_cleanup(self):
        """Test profile cleanup when generating new profile."""
        # Create initial profile
        report_content = '<html>Initial Report</html>'
        json_content = '{"initial": "data"}'
        
        self.dataset.update_profile_files(
            report_content.encode(),
            json_content.encode()
        )
        
        initial_report_path = self.dataset.profile_report.path
        initial_json_path = self.dataset.profile_json.path
        
        # Update with new profile
        new_report_content = '<html>New Report</html>'
        new_json_content = '{"new": "data"}'
        
        self.dataset.update_profile_files(
            new_report_content.encode(),
            new_json_content.encode()
        )
        
        # Verify old files are deleted
        self.assertFalse(os.path.exists(initial_report_path))
        self.assertFalse(os.path.exists(initial_json_path))
        
        # Verify new files exist
        self.assertTrue(os.path.exists(self.dataset.profile_report.path))
        self.assertTrue(os.path.exists(self.dataset.profile_json.path))

    def test_dataset_file_type_validation(self):
        """Test file type validation on model level."""
        # Test invalid file type
        with self.assertRaises(ValidationError):
            Dataset.objects.create(
                name='Invalid Type',
                file_type='invalid'
            )
        
        # Test valid file types
        valid_types = ['csv', 'excel', 'json']
        for file_type in valid_types:
            dataset = Dataset.objects.create(
                name=f'Valid {file_type}',
                file_type=file_type
            )
            self.assertEqual(dataset.file_type, file_type)

    def test_dataset_name_validation(self):
        """Test dataset name validation."""
        # Test empty name
        with self.assertRaises(ValidationError):
            Dataset.objects.create(
                name='',
                file_type='csv'
            )
        
        # Test duplicate name
        with self.assertRaises(ValidationError):
            Dataset.objects.create(
                name=self.dataset.name,
                file_type='csv'
            )

    def test_dataset_profile_status_validation(self):
        """Test profile status validation."""
        valid_statuses = ['pending', 'processing', 'ready', 'failed']
        
        for status in valid_statuses:
            self.dataset.profile_status = status
            self.dataset.save()
            self.assertEqual(self.dataset.profile_status, status)
        
        # Test invalid status
        with self.assertRaises(ValidationError):
            self.dataset.profile_status = 'invalid'
            self.dataset.save()

    def test_dataset_profile_config_validation(self):
        """Test profile configuration validation."""
        # Test invalid JSON
        with self.assertRaises(ValidationError):
            self.dataset.profile_config = 'invalid json'
            self.dataset.save()
        
        # Test valid config
        config = generate_profile_config()
        self.dataset.profile_config = config
        self.dataset.save()
        self.assertEqual(self.dataset.profile_config, config) 

class DataGovernanceRuleTests(TestCase):
    def setUp(self):
        self.rule = DataGovernanceRule.objects.create(
            name='Test Rule',
            description='Test rule description',
            rule_type='range',
            column_name='test_column',
            rule_parameters={'min': 0, 'max': 100}
        )

    def test_rule_creation(self):
        """Test rule creation with valid data."""
        self.assertEqual(self.rule.name, 'Test Rule')
        self.assertEqual(self.rule.rule_type, 'range')
        self.assertEqual(self.rule.column_name, 'test_column')
        self.assertEqual(self.rule.rule_parameters, {'min': 0, 'max': 100})
        self.assertIsNotNone(self.rule.created_date)
        self.assertIsNotNone(self.rule.last_modified)

    def test_rule_str_representation(self):
        """Test the string representation of a rule."""
        self.assertEqual(str(self.rule), 'Test Rule')

    def test_rule_type_validation(self):
        """Test rule type validation."""
        with self.assertRaises(ValidationError):
            DataGovernanceRule.objects.create(
                name='Invalid Rule',
                rule_type='invalid_type',
                column_name='test_column',
                rule_parameters={}
            )

    def test_rule_parameters_validation(self):
        """Test rule parameters validation for different rule types."""
        # Test range rule
        with self.assertRaises(ValidationError):
            DataGovernanceRule.objects.create(
                name='Invalid Range Rule',
                rule_type='range',
                column_name='test_column',
                rule_parameters={'min': 0}  # missing max parameter
            )

        # Test format rule
        with self.assertRaises(ValidationError):
            DataGovernanceRule.objects.create(
                name='Invalid Format Rule',
                rule_type='format',
                column_name='test_column',
                rule_parameters={}  # missing pattern parameter
            )

        # Test regex rule
        with self.assertRaises(ValidationError):
            DataGovernanceRule.objects.create(
                name='Invalid Regex Rule',
                rule_type='regex',
                column_name='test_column',
                rule_parameters={}  # missing pattern parameter
            )

    def test_rule_update(self):
        """Test rule update functionality."""
        # Update rule parameters
        new_parameters = {'min': -10, 'max': 10}
        self.rule.rule_parameters = new_parameters
        self.rule.save()
        
        # Refresh from database
        self.rule.refresh_from_db()
        self.assertEqual(self.rule.rule_parameters, new_parameters)

    def test_rule_duplicate_name(self):
        """Test duplicate rule name validation."""
        with self.assertRaises(ValidationError):
            DataGovernanceRule.objects.create(
                name=self.rule.name,  # duplicate name
                rule_type='range',
                column_name='test_column',
                rule_parameters={'min': 0, 'max': 100}
            ) 

class RuleValidationResultTests(TestCase):
    def setUp(self):
        self.dataset = create_test_dataset()
        self.rule = DataGovernanceRule.objects.create(
            name='Test Rule',
            rule_type='range',
            column_name='test_column',
            rule_parameters={'min': 0, 'max': 100}
        )
        self.validation_result = RuleValidationResult.objects.create(
            rule=self.rule,
            dataset=self.dataset,
            passed=True,
            failed_rows=None,
            error_message=None
        )

    def test_validation_result_creation(self):
        """Test validation result creation with valid data."""
        self.assertEqual(self.validation_result.rule, self.rule)
        self.assertEqual(self.validation_result.dataset, self.dataset)
        self.assertTrue(self.validation_result.passed)
        self.assertIsNone(self.validation_result.failed_rows)
        self.assertIsNone(self.validation_result.error_message)
        self.assertIsNotNone(self.validation_result.validation_date)

    def test_validation_result_str_representation(self):
        """Test the string representation of a validation result."""
        expected_str = f"{self.rule.name} - Passed"
        self.assertEqual(str(self.validation_result), expected_str)

        # Test failed result string representation
        failed_result = RuleValidationResult.objects.create(
            rule=self.rule,
            dataset=self.dataset,
            passed=False,
            failed_rows=[1, 2, 3],
            error_message="Test error"
        )
        expected_str = f"{self.rule.name} - Failed"
        self.assertEqual(str(failed_result), expected_str)

    def test_validation_result_with_failed_rows(self):
        """Test validation result with failed rows."""
        failed_rows = [
            {'row': 1, 'value': 150},
            {'row': 2, 'value': -10}
        ]
        result = RuleValidationResult.objects.create(
            rule=self.rule,
            dataset=self.dataset,
            passed=False,
            failed_rows=failed_rows,
            error_message="Values outside allowed range"
        )
        
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failed_rows), 2)
        self.assertEqual(result.failed_rows[0]['value'], 150)
        self.assertEqual(result.error_message, "Values outside allowed range")

    def test_validation_result_cascade_deletion(self):
        """Test that validation results are deleted when related objects are deleted."""
        # Test deletion when rule is deleted
        rule_id = self.rule.id
        self.rule.delete()
        self.assertFalse(
            RuleValidationResult.objects.filter(rule_id=rule_id).exists()
        )

        # Create new rule and result for dataset deletion test
        new_rule = DataGovernanceRule.objects.create(
            name='Test Rule 2',
            rule_type='range',
            column_name='test_column',
            rule_parameters={'min': 0, 'max': 100}
        )
        result = RuleValidationResult.objects.create(
            rule=new_rule,
            dataset=self.dataset,
            passed=True
        )
        
        # Test deletion when dataset is deleted
        dataset_id = self.dataset.id
        self.dataset.delete()
        self.assertFalse(
            RuleValidationResult.objects.filter(dataset_id=dataset_id).exists()
        ) 

class DataQualityAnalysisTests(TestCase):
    def setUp(self):
        self.dataset = create_test_dataset()
        self.analysis = DataQualityAnalysis.objects.create(
            dataset=self.dataset,
            column_names=['id', 'name', 'value'],
            column_types={'id': 'int', 'name': 'str', 'value': 'float'},
            missing_values={'id': 0, 'name': 2, 'value': 1},
            unique_values={'id': 100, 'name': 95, 'value': 98},
            value_distributions={
                'id': {'0-25': 25, '26-50': 25, '51-75': 25, '76-100': 25},
                'name': {'A-M': 50, 'N-Z': 45},
                'value': {'negative': 30, 'positive': 69}
            },
            numeric_stats={
                'id': {
                    'mean': 50.0,
                    'std': 29.0,
                    'min': 0,
                    'max': 100
                },
                'value': {
                    'mean': 0.1,
                    'std': 1.0,
                    'min': -3.0,
                    'max': 3.0
                }
            },
            categorical_stats={
                'name': {
                    'unique_count': 95,
                    'most_common': ['Item 1', 'Item 2'],
                    'least_common': ['Item 99', 'Item 100']
                }
            }
        )

    def test_analysis_creation(self):
        """Test analysis creation with valid data."""
        self.assertEqual(self.analysis.dataset, self.dataset)
        self.assertEqual(len(self.analysis.column_names), 3)
        self.assertEqual(len(self.analysis.column_types), 3)
        self.assertIsNotNone(self.analysis.analyzed_at)

    def test_analysis_str_representation(self):
        """Test the string representation of an analysis."""
        expected_str = f"Analysis of {self.dataset.name} at {self.analysis.analyzed_at}"
        self.assertEqual(str(self.analysis), expected_str)

    def test_get_missing_values_percent(self):
        """Test calculation of missing values percentage."""
        percentages = self.analysis.get_missing_values_percent()
        self.assertEqual(percentages['id'], 0.0)
        self.assertEqual(percentages['name'], 2.0)
        self.assertEqual(percentages['value'], 1.0)

    def test_get_column_quality_score(self):
        """Test calculation of column quality scores."""
        id_score = self.analysis.get_column_quality_score('id')
        name_score = self.analysis.get_column_quality_score('name')
        value_score = self.analysis.get_column_quality_score('value')

        self.assertEqual(id_score, 100.0)  # No missing values
        self.assertEqual(name_score, 98.0)  # 2% missing values
        self.assertEqual(value_score, 99.0)  # 1% missing values

    def test_analysis_cascade_deletion(self):
        """Test that analysis is deleted when dataset is deleted."""
        analysis_id = self.analysis.id
        self.dataset.delete()
        self.assertFalse(
            DataQualityAnalysis.objects.filter(id=analysis_id).exists()
        )

    def test_analysis_json_validation(self):
        """Test validation of JSON fields."""
        # Test invalid column types
        with self.assertRaises(ValidationError):
            DataQualityAnalysis.objects.create(
                dataset=self.dataset,
                column_names=['test'],
                column_types='invalid',  # Should be a dict
                missing_values={'test': 0},
                unique_values={'test': 1},
                value_distributions={'test': {'A': 1}}
            )

        # Test mismatched column names and types
        with self.assertRaises(ValidationError):
            DataQualityAnalysis.objects.create(
                dataset=self.dataset,
                column_names=['test'],
                column_types={'other': 'int'},
                missing_values={'test': 0},
                unique_values={'test': 1},
                value_distributions={'test': {'A': 1}}
            ) 