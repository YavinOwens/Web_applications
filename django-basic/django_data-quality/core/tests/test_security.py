import os
import json
from django.test import TestCase
from django.contrib.auth.models import User, Permission
from django.contrib.contenttypes.models import ContentType
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from ..models import Dataset, DataGovernanceRule, DataQualityAnalysis
from .base import BaseTestCase

class SecurityTestCase(BaseTestCase):
    def setUp(self):
        super().setUp()
        
        # Create different types of users
        self.admin_user = User.objects.create_superuser(
            username='admin', 
            email='admin@example.com', 
            password='adminpassword'
        )
        
        self.staff_user = User.objects.create_user(
            username='staff', 
            email='staff@example.com', 
            password='staffpassword',
            is_staff=True
        )
        
        # Add specific permissions to staff user
        dataset_view_perm = Permission.objects.get(codename='view_dataset')
        dataset_add_perm = Permission.objects.get(codename='add_dataset')
        rule_add_perm = Permission.objects.get(codename='add_datagovernancerule')
        self.staff_user.user_permissions.add(dataset_view_perm, dataset_add_perm, rule_add_perm)
        
        self.regular_user = User.objects.create_user(
            username='regular', 
            email='regular@example.com', 
            password='regularpassword'
        )

    def test_admin_full_access(self):
        """
        Test that admin users have full access to all views and actions
        """
        self.client.login(username='admin', password='adminpassword')
        
        # Test dataset views
        response = self.client.get(reverse('core:dataset_list'))
        self.assertEqual(response.status_code, 200)
        
        response = self.client.get(reverse('core:dataset_upload'))
        self.assertEqual(response.status_code, 200)
        
        # Test rule views
        response = self.client.get(reverse('core:rule_list'))
        self.assertEqual(response.status_code, 200)
        
        response = self.client.get(reverse('core:rule_create'))
        self.assertEqual(response.status_code, 200)

    def test_staff_restricted_access(self):
        """
        Test that staff users have most, but not all, permissions
        """
        self.client.login(username='staff', password='staffpassword')
        
        # Test dataset views
        response = self.client.get(reverse('core:dataset_list'))
        self.assertEqual(response.status_code, 200)
        
        response = self.client.get(reverse('core:dataset_upload'))
        self.assertEqual(response.status_code, 200)
        
        # Test rule views
        response = self.client.get(reverse('core:rule_list'))
        self.assertEqual(response.status_code, 200)
        
        response = self.client.get(reverse('core:rule_create'))
        self.assertEqual(response.status_code, 200)

    def test_regular_user_restricted_access(self):
        """
        Test that regular users have limited access
        """
        self.client.login(username='regular', password='regularpassword')
        
        # Test dataset views
        response = self.client.get(reverse('core:dataset_list'))
        self.assertEqual(response.status_code, 200)
        
        # Expect redirect for dataset upload
        response = self.client.get(reverse('core:dataset_upload'))
        self.assertEqual(response.status_code, 302, "Expected redirect for unauthorized dataset upload")
        
        # Test rule views
        response = self.client.get(reverse('core:rule_list'))
        self.assertEqual(response.status_code, 200)
        
        # Expect redirect for rule creation
        response = self.client.get(reverse('core:rule_create_for_dataset', args=[self.dataset.id]))
        self.assertEqual(response.status_code, 302, "Expected redirect for unauthorized rule creation")

    def test_dataset_ownership(self):
        """
        Test that users can only modify their own datasets
        """
        # Create a dataset owned by admin
        admin_dataset = Dataset.objects.create(
            name='Admin Dataset',
            description='Dataset owned by admin',
            file=self.dataset.file,
            uploaded_by=self.admin_user,
            file_type='csv'  # Explicitly set file_type
        )
        
        # Try to modify admin's dataset as a regular user
        self.client.login(username='regular', password='regularpassword')
        
        response = self.client.post(
            reverse('core:dataset_delete', args=[admin_dataset.id])
        )
        
        # Should be forbidden or redirected
        self.assertIn(response.status_code, [302, 403])

    def test_rule_creation_permissions(self):
        """
        Test rule creation permissions
        """
        # Login as regular user
        self.client.login(username='regular', password='regularpassword')
        
        # Try to create a rule
        response = self.client.post(
            reverse('core:rule_create_for_dataset', args=[self.dataset.id]),
            data={
                'name': 'Test Rule',
                'description': 'A test rule',
                'column_name': 'age',
                'rule_type': 'range',
                'parameters': json.dumps({'min': 20, 'max': 60})
            }
        )
        
        # Should be forbidden or redirected
        self.assertIn(response.status_code, [302, 403], 
            f"Unexpected response status: {response.status_code}")

    def test_validation_permissions(self):
        """
        Test rule validation permissions
        """
        # Create a rule
        rule = DataGovernanceRule.objects.create(
            name='Test Validation Rule',
            dataset=self.dataset,
            column_name='age',
            rule_type='range',
            parameters={'min': 20, 'max': 60},
            created_by=self.admin_user
        )
        
        # Login as regular user
        self.client.login(username='regular', password='regularpassword')
        
        # Try to validate the rule
        response = self.client.post(
            reverse('core:rule_validate', args=[rule.id])
        )
        
        # Should be forbidden or redirected
        self.assertIn(response.status_code, [302, 403])

    def tearDown(self):
        # Clean up additional users
        self.admin_user.delete()
        self.staff_user.delete()
        self.regular_user.delete()
        
        super().tearDown() 