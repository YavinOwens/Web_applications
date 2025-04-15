import json
from django.urls import reverse
from .base import BaseTestCase
from core.models import RuleValidationResult
from core.views import validate_rule
import pandas as pd

class RuleValidationTestCase(BaseTestCase):
    def test_range_rule_validation(self):
        """
        Test range rule validation with valid and invalid data
        """
        # Create a DataFrame with some out-of-range values
        test_data = pd.DataFrame({
            'age': [10, 20, 30, 40, 80],  # 10 and 80 are out of range
            'salary': [30000, 50000, 100000, 200000, 160000]  # 30000 and 200000 are out of range
        })
        
        # Validate age range rule
        failed_rows_age = validate_rule(test_data, self.age_range_rule)
        self.assertEqual(len(failed_rows_age), 2)  # 10 and 80 should fail
        
        # Validate salary range rule
        failed_rows_salary = validate_rule(test_data, self.salary_range_rule)
        self.assertEqual(len(failed_rows_salary), 2)  # 30000 and 200000 should fail

    def test_required_rule_validation(self):
        """
        Test required rule validation
        """
        # Create a DataFrame with some empty values
        test_data = pd.DataFrame({
            'name': ['Alice', '', 'Charlie', None, 'Eve']
        })
        
        # Validate name required rule
        failed_rows = validate_rule(test_data, self.name_required_rule)
        self.assertEqual(len(failed_rows), 2)  # Empty string and None should fail

    def test_unique_rule_validation(self):
        """
        Test unique rule validation
        """
        # Create a unique rule
        unique_rule = self.create_unique_rule('name')
        
        # Create a DataFrame with duplicate values
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob']
        })
        
        # Validate unique rule
        failed_rows = validate_rule(test_data, unique_rule)
        self.assertEqual(len(failed_rows), 2)  # Duplicate 'Alice' and 'Bob' should fail

    def test_rule_validation_view(self):
        """
        Test the rule validation view
        """
        # Login the test user
        self.client.login(username='testuser', password='testpassword')
        
        # Trigger rule validation
        response = self.client.post(
            reverse('core:rule_validate', args=[self.age_range_rule.id])
        )
        
        # Check response
        self.assertEqual(response.status_code, 302)  # Redirect after validation
        
        # Check validation result
        validation_result = RuleValidationResult.objects.filter(
            rule=self.age_range_rule
        ).first()
        
        self.assertIsNotNone(validation_result)
        self.assertFalse(validation_result.passed)  # Some rows should fail
        
        # Check failed rows
        failed_rows = json.loads(validation_result.failed_rows)
        self.assertTrue(len(failed_rows) > 0)

    def create_unique_rule(self, column_name):
        """
        Helper method to create a unique rule
        """
        return self.model_class.objects.create(
            name=f'Unique {column_name} Rule',
            description=f'{column_name} must be unique',
            dataset=self.dataset,
            column_name=column_name,
            rule_type='unique',
            parameters={},
            created_by=self.user
        ) 