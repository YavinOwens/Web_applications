import os
import django
import pytest
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
import pandas as pd
import tempfile
from ..models import Dataset, DataGovernanceRule

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

@pytest.fixture
def test_user():
    """Create a test user."""
    return User.objects.create_user(
        username='testuser',
        password='testpass123',
        email='test@example.com'
    )

@pytest.fixture
def test_dataset(test_user):
    """Create a test dataset."""
    # Create test data
    df = pd.DataFrame({
        'id': range(1, 6),
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 
                 'david@test.com', 'eve@test.com']
    })
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp.seek(0)
        
        dataset = Dataset.objects.create(
            name='Test Dataset',
            description='Dataset for testing',
            file=SimpleUploadedFile('test.csv', tmp.read()),
            file_type='csv',
            created_by=test_user
        )
    
    # Clean up temp file
    os.unlink(tmp.name)
    return dataset

@pytest.fixture
def test_rule(test_dataset, test_user):
    """Create a test rule."""
    return DataGovernanceRule.objects.create(
        name='Test Rule',
        description='Rule for testing',
        rule_type='format',
        dataset=test_dataset,
        column_name='email',
        parameters={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
        created_by=test_user
    )

@pytest.fixture
def authenticated_client(client, test_user):
    """Create an authenticated client."""
    client.login(username='testuser', password='testpass123')
    return client

@pytest.fixture(autouse=True)
def media_storage(settings, tmpdir):
    """Configure temporary media storage."""
    settings.MEDIA_ROOT = tmpdir.strpath

@pytest.fixture(autouse=True)
def use_test_database(db):
    """Ensure test database is used."""
    pass

def pytest_configure():
    """Configure test settings."""
    settings.DEBUG = False
    settings.USE_TZ = True
    settings.DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    }
    settings.INSTALLED_APPS = [
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'core',
    ]
    settings.MIDDLEWARE = [
        'django.middleware.security.SecurityMiddleware',
        'django.contrib.sessions.middleware.SessionMiddleware',
        'django.middleware.common.CommonMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.contrib.auth.middleware.AuthenticationMiddleware',
        'django.contrib.messages.middleware.MessageMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',
    ]
    settings.TEMPLATES = [
        {
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': True,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.debug',
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                ],
            },
        },
    ]
    settings.STATIC_URL = '/static/'
    settings.MEDIA_URL = '/media/'
    settings.SECRET_KEY = 'test_key'
    settings.DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField' 