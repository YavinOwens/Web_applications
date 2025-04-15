from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from bs4 import BeautifulSoup
from ..models import Dataset, DataGovernanceRule, RuleValidationResult
import json

class UITestCase(TestCase):
    def setUp(self):
        """Set up test environment."""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')

    def assert_bootstrap_class(self, html, selector, expected_class):
        """Helper method to check Bootstrap classes."""
        soup = BeautifulSoup(html, 'html.parser')
        elements = soup.select(selector)
        self.assertTrue(elements, f"No elements found for selector: {selector}")
        for element in elements:
            self.assertIn(expected_class, element.get('class', []))

    def test_navigation_styling(self):
        """Test navigation bar styling and active states."""
        response = self.client.get(reverse('core:dashboard'))
        self.assertEqual(response.status_code, 200)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check navbar styling
        navbar = soup.find('nav', class_='navbar')
        self.assertIsNotNone(navbar)
        self.assertIn('navbar-expand-lg', navbar['class'])
        
        # Check active state
        active_link = soup.find('a', class_='active')
        self.assertIsNotNone(active_link)
        self.assertEqual(active_link.text.strip().lower(), 'dashboard')

    def test_form_styling(self):
        """Test form styling and validation states."""
        response = self.client.get(reverse('core:dataset_upload'))
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check form elements
        form = soup.find('form')
        self.assertIsNotNone(form)
        self.assertIn('needs-validation', form.get('class', []))
        
        # Check input styling
        inputs = soup.find_all('input')
        for input_field in inputs:
            self.assertIn('form-control', input_field.get('class', []))

    def test_button_interactions(self):
        """Test button states and interactions."""
        # Test submit button state changes
        response = self.client.get(reverse('core:rule_create'))
        soup = BeautifulSoup(response.content, 'html.parser')
        
        submit_button = soup.find('button', type='submit')
        self.assertIsNotNone(submit_button)
        self.assertIn('btn', submit_button['class'])
        self.assertIn('btn-primary', submit_button['class'])

    def test_modal_functionality(self):
        """Test modal dialogs and their interactions."""
        response = self.client.get(reverse('core:dataset_list'))
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check delete confirmation modal
        modal = soup.find('div', class_='modal')
        self.assertIsNotNone(modal)
        self.assertTrue(modal.find('button', class_='btn-danger'))

    def test_grid_styling(self):
        """Test AG Grid styling and functionality."""
        response = self.client.get(reverse('core:analyze_with_grid', args=[1]))
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check grid container
        grid_div = soup.find('div', id='myGrid')
        self.assertIsNotNone(grid_div)
        self.assertIn('ag-theme-alpine', grid_div['class'])

    def test_responsive_layout(self):
        """Test responsive layout breakpoints."""
        response = self.client.get(reverse('core:dashboard'))
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check container classes
        containers = soup.find_all('div', class_='container')
        self.assertTrue(containers)
        
        # Check responsive columns
        cols = soup.find_all('div', class_=lambda x: x and 'col-' in x)
        self.assertTrue(cols)

    def test_form_validation_feedback(self):
        """Test form validation feedback styling."""
        # Test invalid form submission
        response = self.client.post(
            reverse('core:dataset_upload'),
            {'name': ''}  # Missing required fields
        )
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check error feedback styling
        error_messages = soup.find_all('div', class_='invalid-feedback')
        self.assertTrue(error_messages)

    def test_accordion_functionality(self):
        """Test accordion components in rule creation."""
        response = self.client.get(reverse('core:rule_create'))
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check accordion structure
        accordion = soup.find('div', class_='accordion')
        self.assertIsNotNone(accordion)
        
        # Check accordion items
        items = accordion.find_all('div', class_='accordion-item')
        self.assertTrue(items)

    def test_toast_notifications(self):
        """Test toast notification styling and positioning."""
        # Create a dataset to trigger success message
        response = self.client.get(reverse('core:dataset_list'))
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check toast container
        toast_container = soup.find('div', class_='toast-container')
        self.assertIsNotNone(toast_container)

    def tearDown(self):
        """Clean up test environment."""
        self.user.delete() 