from django.test import TestCase, Client
from django.urls import reverse
import os
from django.conf import settings

class DocumentationViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        
        # Ensure documentation file exists
        docs_dir = os.path.join(settings.BASE_DIR, 'core', 'docs', 'source')
        os.makedirs(docs_dir, exist_ok=True)
        
        # Create a test markdown file
        test_docs_path = os.path.join(docs_dir, 'index.md')
        with open(test_docs_path, 'w') as f:
            f.write('''# Test Documentation

This is a test documentation file.

## Features
- Feature 1
- Feature 2
''')

    def test_documentation_view(self):
        """
        Test that the documentation view returns a 200 status code
        and contains expected content
        """
        response = self.client.get(reverse('core:documentation'))
        
        # Check response status
        self.assertEqual(response.status_code, 200)
        
        # Check content
        self.assertContains(response, 'Test Documentation')
        self.assertContains(response, 'Feature 1')
        self.assertContains(response, 'Feature 2')

    def tearDown(self):
        # Clean up test documentation file
        test_docs_path = os.path.join(settings.BASE_DIR, 'core', 'docs', 'source', 'index.md')
        if os.path.exists(test_docs_path):
            os.remove(test_docs_path) 