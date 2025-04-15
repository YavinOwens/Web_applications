import os
import tempfile
import pandas as pd
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from core.models import Dataset, DataGovernanceRule, RuleValidationResult

class BaseTestCase(TestCase):
    def setUp(self):
        # Create test user
        self.user = User.objects.create_user(
            username='testuser', 
            password='testpassword',
            email='testuser@example.com'
        )
        
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
        
        # Create a test dataset
        self.dataset = Dataset.objects.create(
            name='Test Dataset',
            description='A test dataset for validation',
            file=SimpleUploadedFile(
                name='test_data.csv', 
                content=open(self.temp_csv.name, 'rb').read()
            ),
            file_type='csv',
            uploaded_by=self.user,
            total_rows=10,
            total_columns=3
        )
        
        # Create some test rules
        self.age_range_rule = DataGovernanceRule.objects.create(
            name='Age Range Rule',
            description='Age must be between 20 and 75',
            dataset=self.dataset,
            column_name='age',
            rule_type='range',
            parameters={'min': 20, 'max': 75},
            created_by=self.user
        )
        
        self.salary_range_rule = DataGovernanceRule.objects.create(
            name='Salary Range Rule',
            description='Salary must be between 40000 and 150000',
            dataset=self.dataset,
            column_name='salary',
            rule_type='range',
            parameters={'min': 40000, 'max': 150000},
            created_by=self.user
        )
        
        self.name_required_rule = DataGovernanceRule.objects.create(
            name='Name Required Rule',
            description='Name cannot be empty',
            dataset=self.dataset,
            column_name='name',
            rule_type='required',
            parameters={},
            created_by=self.user
        )

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
        
        # Delete test objects
        self.dataset.delete()
        self.user.delete() 