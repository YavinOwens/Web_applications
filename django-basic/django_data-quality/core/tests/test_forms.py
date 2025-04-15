from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from ..forms import DatasetUploadForm, DataGovernanceRuleForm
import tempfile
import pandas as pd

class DatasetUploadFormTests(TestCase):
    def test_valid_csv_upload(self):
        """Test form validation with valid CSV file."""
        # Create a valid CSV file
        df = pd.DataFrame({'test': range(5)})
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            tmp.seek(0)
            file_data = tmp.read()

        form_data = {
            'name': 'Test Dataset',
            'description': 'Test Description',
            'file_type': 'csv'
        }
        file_dict = {'file': SimpleUploadedFile('test.csv', file_data, content_type='text/csv')}
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertTrue(form.is_valid())

    def test_valid_excel_upload(self):
        """Test form validation with valid Excel file."""
        # Create a valid Excel file
        df = pd.DataFrame({'test': range(5)})
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as tmp:
            df.to_excel(tmp.name, index=False)
            tmp.seek(0)
            file_data = tmp.read()

        form_data = {
            'name': 'Test Dataset',
            'description': 'Test Description',
            'file_type': 'excel'
        }
        file_dict = {'file': SimpleUploadedFile('test.xlsx', file_data, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertTrue(form.is_valid())

    def test_valid_json_upload(self):
        """Test form validation with valid JSON file."""
        # Create a valid JSON file
        df = pd.DataFrame({'test': range(5)})
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            df.to_json(tmp.name, orient='records')
            tmp.seek(0)
            file_data = tmp.read()

        form_data = {
            'name': 'Test Dataset',
            'description': 'Test Description',
            'file_type': 'json'
        }
        file_dict = {'file': SimpleUploadedFile('test.json', file_data, content_type='application/json')}
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertTrue(form.is_valid())

    def test_invalid_file_type(self):
        """Test form validation with invalid file type."""
        form_data = {
            'name': 'Test Dataset',
            'description': 'Test Description',
            'file_type': 'txt'
        }
        file_dict = {'file': SimpleUploadedFile('test.txt', b'invalid data', content_type='text/plain')}
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form.is_valid())
        self.assertIn('file_type', form.errors)

    def test_missing_file(self):
        """Test form validation with missing file."""
        form_data = {
            'name': 'Test Dataset',
            'description': 'Test Description',
            'file_type': 'csv'
        }
        
        form = DatasetUploadForm(form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('file', form.errors)

    def test_empty_name(self):
        """Test form validation with empty name."""
        form_data = {
            'name': '',
            'description': 'Test Description',
            'file_type': 'csv'
        }
        file_dict = {'file': SimpleUploadedFile('test.csv', b'test,data\n1,2', content_type='text/csv')}
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form.is_valid())
        self.assertIn('name', form.errors)

    def test_file_content_validation(self):
        """Test form validation of file contents."""
        # Test with invalid CSV content
        form_data = {
            'name': 'Test Dataset',
            'description': 'Test Description',
            'file_type': 'csv'
        }
        file_dict = {'file': SimpleUploadedFile('test.csv', b'invalid,csv,content', content_type='text/csv')}
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form.is_valid())
        self.assertIn('file', form.errors)

        # Test with invalid Excel content
        file_dict = {'file': SimpleUploadedFile('test.xlsx', b'invalid excel content', content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        form_data['file_type'] = 'excel'
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form.is_valid())
        self.assertIn('file', form.errors)

        # Test with invalid JSON content
        file_dict = {'file': SimpleUploadedFile('test.json', b'invalid json content', content_type='application/json')}
        form_data['file_type'] = 'json'
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form.is_valid())
        self.assertIn('file', form.errors)

    def test_file_size_validation(self):
        """Test form validation of file size."""
        # Create a large file
        large_df = pd.DataFrame({'test': range(1000000)})  # 1M rows
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            large_df.to_csv(tmp.name, index=False)
            tmp.seek(0)
            file_data = tmp.read()

        form_data = {
            'name': 'Large Dataset',
            'description': 'Test Description',
            'file_type': 'csv'
        }
        file_dict = {'file': SimpleUploadedFile('large.csv', file_data, content_type='text/csv')}
        
        form = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form.is_valid())
        self.assertIn('file', form.errors)

    def test_file_extension_validation(self):
        """Test form validation of file extensions."""
        form_data = {
            'name': 'Test Dataset',
            'description': 'Test Description',
            'file_type': 'csv'
        }
        
        # Test with mismatched extension
        file_dict = {'file': SimpleUploadedFile('test.xlsx', b'csv,data\n1,2', content_type='text/csv')}
        form = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form.is_valid())
        self.assertIn('file', form.errors)

        # Test with no extension
        file_dict = {'file': SimpleUploadedFile('test', b'csv,data\n1,2', content_type='text/csv')}
        form = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form.is_valid())
        self.assertIn('file', form.errors)

    def test_duplicate_name_validation(self):
        """Test form validation of duplicate dataset names."""
        # Create first dataset
        form_data = {
            'name': 'Test Dataset',
            'description': 'Test Description',
            'file_type': 'csv'
        }
        file_dict = {'file': SimpleUploadedFile('test1.csv', b'test,data\n1,2', content_type='text/csv')}
        
        form1 = DatasetUploadForm(form_data, file_dict)
        self.assertTrue(form1.is_valid())
        form1.save()

        # Try to create second dataset with same name
        file_dict = {'file': SimpleUploadedFile('test2.csv', b'test,data\n3,4', content_type='text/csv')}
        form2 = DatasetUploadForm(form_data, file_dict)
        self.assertFalse(form2.is_valid())
        self.assertIn('name', form2.errors) 

class DataGovernanceRuleFormTests(TestCase):
    def test_valid_range_rule(self):
        """Test form validation with valid range rule."""
        form_data = {
            'name': 'Test Range Rule',
            'description': 'Test range validation',
            'rule_type': 'range',
            'column_name': 'test_column',
            'rule_parameters': '{"min": 0, "max": 100}'
        }
        form = DataGovernanceRuleForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_valid_format_rule(self):
        """Test form validation with valid format rule."""
        form_data = {
            'name': 'Test Format Rule',
            'description': 'Test format validation',
            'rule_type': 'format',
            'column_name': 'test_column',
            'rule_parameters': '{"pattern": "^[A-Z]{2}\\d{4}$"}'
        }
        form = DataGovernanceRuleForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_valid_regex_rule(self):
        """Test form validation with valid regex rule."""
        form_data = {
            'name': 'Test Regex Rule',
            'description': 'Test regex validation',
            'rule_type': 'regex',
            'column_name': 'email',
            'rule_parameters': '{"pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"}'
        }
        form = DataGovernanceRuleForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_invalid_rule_type(self):
        """Test form validation with invalid rule type."""
        form_data = {
            'name': 'Invalid Rule',
            'description': 'Test invalid rule type',
            'rule_type': 'invalid',
            'column_name': 'test_column',
            'rule_parameters': '{}'
        }
        form = DataGovernanceRuleForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('rule_type', form.errors)

    def test_invalid_rule_parameters(self):
        """Test form validation with invalid rule parameters."""
        # Test invalid JSON
        form_data = {
            'name': 'Invalid Parameters',
            'description': 'Test invalid parameters',
            'rule_type': 'range',
            'column_name': 'test_column',
            'rule_parameters': 'invalid json'
        }
        form = DataGovernanceRuleForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('rule_parameters', form.errors)

        # Test missing required parameters for range rule
        form_data['rule_parameters'] = '{"min": 0}'  # missing max
        form = DataGovernanceRuleForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('rule_parameters', form.errors)

        # Test missing required parameters for format rule
        form_data.update({
            'rule_type': 'format',
            'rule_parameters': '{}'  # missing pattern
        })
        form = DataGovernanceRuleForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('rule_parameters', form.errors)

    def test_empty_name(self):
        """Test form validation with empty name."""
        form_data = {
            'name': '',
            'description': 'Test empty name',
            'rule_type': 'range',
            'column_name': 'test_column',
            'rule_parameters': '{"min": 0, "max": 100}'
        }
        form = DataGovernanceRuleForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('name', form.errors)

    def test_empty_column_name(self):
        """Test form validation with empty column name."""
        form_data = {
            'name': 'Test Rule',
            'description': 'Test empty column',
            'rule_type': 'range',
            'column_name': '',
            'rule_parameters': '{"min": 0, "max": 100}'
        }
        form = DataGovernanceRuleForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('column_name', form.errors)

    def test_duplicate_rule_name(self):
        """Test form validation with duplicate rule name."""
        # Create initial rule
        form_data = {
            'name': 'Test Rule',
            'description': 'Test duplicate name',
            'rule_type': 'range',
            'column_name': 'test_column',
            'rule_parameters': '{"min": 0, "max": 100}'
        }
        form = DataGovernanceRuleForm(data=form_data)
        self.assertTrue(form.is_valid())
        form.save()

        # Try to create another rule with the same name
        form = DataGovernanceRuleForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('name', form.errors) 