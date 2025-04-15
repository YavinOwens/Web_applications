import os
import tempfile
import pandas as pd
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.core.exceptions import ValidationError
import json

from .base import BaseTestCase
from core.models import Dataset, DataGovernanceRule
from core.forms import DatasetUploadForm, DataGovernanceRuleForm

class FormValidationTestCase(BaseTestCase):
    def setUp(self):
        super().setUp()
        
        # Create a temporary CSV file for testing
        self.temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        test_data = pd.DataFrame({
            'id': range(1, 11),
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 
                     'Frank', 'Grace', 'Heidi', 'Ivan', 'Julia'],
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'salary': [50000, 60000, 70000, 80000, 90000, 
                       100000, 110000, 120000, 130000, 140000]
        })
        test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()

    def test_dataset_upload_form_valid(self):
        """
        Test dataset upload form with valid data
        """
        with open(self.temp_csv.name, 'rb') as file:
            form_data = {
                'name': 'Valid Dataset',
                'description': 'A valid dataset for testing',
                'file_type': 'csv'
            }
            files = {'file': SimpleUploadedFile(
                name='test_data.csv', 
                content=file.read(), 
                content_type='text/csv'
            )}
            
            # Create a request mock to simulate authenticated user
            from django.test import RequestFactory
            request = RequestFactory().post('/upload', data=form_data)
            request.user = self.user
            
            form = DatasetUploadForm(form_data, files)
            form.request = request  # Simulate request attribute
            
            self.assertTrue(form.is_valid(), form.errors)
            
            # Save the form
            dataset = form.save()
            
            # Verify dataset attributes
            self.assertEqual(dataset.name, 'Valid Dataset')
            self.assertEqual(dataset.file_type, 'csv')
            
            # Manually trigger metadata update
            dataset.get_preview_data()
            dataset.refresh_from_db()
            
            # Verify rows and columns
            self.assertEqual(dataset.total_rows, 10)
            self.assertEqual(dataset.total_columns, 4)

    def test_dataset_upload_form_invalid(self):
        """
        Test dataset upload form with invalid data
        """
        # Test missing required fields
        form_data = {
            'description': 'Incomplete dataset'
        }
        form = DatasetUploadForm(form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('name', form.errors)
        self.assertIn('file', form.errors)

    def test_dataset_upload_form_file_validation(self):
        """
        Test file type and size validation
        """
        # Create an invalid file type
        invalid_file = SimpleUploadedFile(
            'invalid.txt', 
            b'Invalid file content', 
            content_type='text/plain'
        )
        
        form_data = {
            'name': 'Invalid File Type Dataset',
            'description': 'Dataset with invalid file type',
            'file_type': 'csv'
        }
        files = {'file': invalid_file}
        
        form = DatasetUploadForm(form_data, files)
        self.assertFalse(form.is_valid())
        self.assertIn('file', form.errors)

    def test_governance_rule_form_valid(self):
        """
        Test governance rule form with valid data
        """
        # Ensure the dataset columns are correctly set up
        df = pd.read_csv(self.temp_csv.name)
        columns = list(df.columns)
        
        form_data = {
            'name': 'Valid Age Rule',
            'description': 'Age must be between 20 and 75',
            'dataset': self.dataset.id,
            'column_name': 'age',
            'rule_type': 'range',
            'parameters_json': json.dumps({"min": 20, "max": 75})
        }
        
        form = DataGovernanceRuleForm(form_data, dataset_id=self.dataset.id, dataset_columns=columns)
        form.fields['column_name'].choices = [(col, col) for col in columns]
        
        # Validate the form
        is_valid = form.is_valid()
        
        # If form is not valid, print out errors for debugging
        if not is_valid:
            print("Form Errors:", form.errors)
        
        self.assertTrue(is_valid, "Form validation failed")
        
        # Save the form
        rule = form.save(commit=False)
        rule.created_by = self.user
        rule.save()
        
        # Verify rule attributes
        self.assertEqual(rule.name, 'Valid Age Rule')
        self.assertEqual(rule.column_name, 'age')
        self.assertEqual(rule.rule_type, 'range')
        self.assertEqual(rule.parameters, {"min": 20, "max": 75})

    def test_governance_rule_form_invalid(self):
        """
        Test governance rule form with invalid data
        """
        # Test missing required fields
        form_data = {
            'description': 'Incomplete rule'
        }
        
        form = DataGovernanceRuleForm(form_data, dataset_id=self.dataset.id)
        self.assertFalse(form.is_valid())
        self.assertIn('name', form.errors)
        self.assertIn('column_name', form.errors)
        self.assertIn('rule_type', form.errors)

    def test_governance_rule_form_invalid_column(self):
        """
        Test governance rule form with non-existent column
        """
        form_data = {
            'name': 'Invalid Column Rule',
            'description': 'Rule for non-existent column',
            'dataset': self.dataset.id,
            'column_name': 'non_existent_column',
            'rule_type': 'range',
            'parameters': '{"min": 20, "max": 75}'
        }
        
        form = DataGovernanceRuleForm(form_data, dataset_id=self.dataset.id)
        self.assertFalse(form.is_valid())
        self.assertIn('column_name', form.errors)

    def test_governance_rule_form_invalid_parameters(self):
        """
        Test governance rule form with invalid parameters
        """
        form_data = {
            'name': 'Invalid Parameters Rule',
            'description': 'Rule with invalid parameters',
            'dataset': self.dataset.id,
            'column_name': 'age',
            'rule_type': 'range',
            'parameters': 'Invalid JSON'
        }
        
        form = DataGovernanceRuleForm(form_data, dataset_id=self.dataset.id)
        self.assertFalse(form.is_valid())
        self.assertIn('parameters', form.errors)

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
        
        super().tearDown() 