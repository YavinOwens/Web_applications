from django import forms
from django.core.exceptions import ValidationError
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Field, Div, Row, Column, HTML
from .models import Dataset, DataGovernanceRule
from .utils import validate_file_size, validate_file_extension, validate_file_content, validate_profile_config
import json
import re

class DatasetUploadForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_class = 'form-horizontal'
        self.helper.label_class = 'col-lg-2'
        self.helper.field_class = 'col-lg-8'
        self.helper.layout = Layout(
            Field('name', css_class='form-control'),
            Field('description', css_class='form-control', rows=3),
            Field('file', css_class='form-control'),
            Field('file_type', css_class='form-control'),
            Div(
                Submit('submit', 'Upload Dataset', css_class='btn btn-primary'),
                css_class='text-center mt-4'
            )
        )

    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file', 'file_type']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'file_type': forms.Select(attrs={'class': 'form-select'}),
        }
        help_texts = {
            'file': 'Upload a CSV, Excel, or JSON file (max 50MB)',
            'file_type': 'Select the type of file you are uploading',
        }

    def clean_file(self):
        """Validate uploaded file."""
        file = self.cleaned_data.get('file')
        if not file:
            raise forms.ValidationError('Please select a file to upload.')
        
        try:
            # Validate file size
            validate_file_size(file)
            
            # Validate file extension
            validate_file_extension(file)
            
            # Only validate content if file type is selected
            file_type = self.cleaned_data.get('file_type')
            if file_type:
                detected_type = validate_file_content(file)
                if detected_type != file_type:
                    raise forms.ValidationError(
                        f'File content does not match selected type. '
                        f'Detected type: {detected_type}'
                    )
        except ValidationError as e:
            raise forms.ValidationError(str(e))
        except Exception as e:
            raise forms.ValidationError(f'Error validating file: {str(e)}')
        
        return file

    def clean_name(self):
        """Validate dataset name."""
        name = self.cleaned_data.get('name')
        if not name:
            raise forms.ValidationError('Dataset name is required')
        
        # Check for duplicate names
        existing = Dataset.objects.filter(name=name)
        if self.instance and self.instance.pk:
            existing = existing.exclude(pk=self.instance.pk)
        if existing.exists():
            raise forms.ValidationError('A dataset with this name already exists')
        
        return name

    def clean(self):
        """Validate the form data."""
        cleaned_data = super().clean()
        file_type = cleaned_data.get('file_type')
        file = cleaned_data.get('file')
        
        if file and not file_type:
            raise forms.ValidationError({
                'file_type': 'Please select a file type'
            })
        
        # Validate file type
        if file_type and file_type not in dict(Dataset.FILE_TYPE_CHOICES):
            raise forms.ValidationError({
                'file_type': f'Invalid file type. Must be one of: {", ".join(dict(Dataset.FILE_TYPE_CHOICES).keys())}'
            })
        
        return cleaned_data

    def save(self, commit=True):
        """
        Override save method to handle uploaded_by field.
        
        If the user is authenticated, set the uploaded_by field.
        """
        instance = super().save(commit=False)
        
        # Check if user is authenticated
        if hasattr(self, 'request') and self.request.user.is_authenticated:
            instance.uploaded_by = self.request.user
        
        if commit:
            instance.save()
        
        return instance

class ProfileConfigForm(forms.Form):
    minimal = forms.BooleanField(required=False)
    sample_size = forms.IntegerField(min_value=-1, initial=-1)
    
    # Correlation options
    pearson = forms.BooleanField(required=False)
    spearman = forms.BooleanField(required=False)
    kendall = forms.BooleanField(required=False)
    
    # Missing value diagrams
    missing_matrix = forms.BooleanField(required=False)
    missing_bar = forms.BooleanField(required=False)
    missing_heatmap = forms.BooleanField(required=False)
    
    # Duplicate analysis
    duplicates_table = forms.BooleanField(required=False)
    duplicates_matrix = forms.BooleanField(required=False)
    
    def get_profile_config(self):
        """Convert form data to profile configuration dictionary."""
        cleaned_data = self.cleaned_data
        
        config = {
            'minimal': cleaned_data.get('minimal', False),
            'sample_size': cleaned_data.get('sample_size', -1),
            'correlations': {
                'pearson': cleaned_data.get('pearson', True),
                'spearman': cleaned_data.get('spearman', False),
                'kendall': cleaned_data.get('kendall', False)
            },
            'missing_diagrams': {
                'matrix': cleaned_data.get('missing_matrix', True),
                'bar': cleaned_data.get('missing_bar', True),
                'heatmap': cleaned_data.get('missing_heatmap', False)
            },
            'duplicates': {
                'table': cleaned_data.get('duplicates_table', True),
                'matrix': cleaned_data.get('duplicates_matrix', False)
            }
        }
        
        return config

class AnalysisConfigForm(forms.Form):
    """Form for configuring dataset analysis."""
    analyze_correlations = forms.BooleanField(required=False, initial=True)
    include_missing_analysis = forms.BooleanField(required=False, initial=True)
    detect_outliers = forms.BooleanField(required=False, initial=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_id = 'analysis-form'
        self.helper.layout = Layout(
            Field('analyze_correlations'),
            Field('include_missing_analysis'),
            Field('detect_outliers'),
        )

    def clean(self):
        cleaned_data = super().clean()
        # At least one analysis type must be selected
        if not any([
            cleaned_data.get('analyze_correlations'),
            cleaned_data.get('include_missing_analysis'),
            cleaned_data.get('detect_outliers')
        ]):
            raise forms.ValidationError('At least one analysis type must be selected.')
        return cleaned_data

class JSONTextAreaWidget(forms.Textarea):
    """Custom widget to handle JSON input in a textarea."""
    def __init__(self, attrs=None):
        if attrs is None:
            attrs = {}
        attrs.setdefault('class', 'form-control')
        attrs.setdefault('rows', 3)
        super().__init__(attrs)
    
    def format_value(self, value):
        """Format the JSON value for display in the textarea."""
        if value is None:
            return ''
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, indent=2)
        except (TypeError, ValueError):
            return str(value)

class DataGovernanceRuleForm(forms.ModelForm):
    """Form for creating and editing data governance rules."""
    
    parameters_json = forms.CharField(
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        help_text='Enter parameters as a valid JSON object. Example: {"min": 0, "max": 100}',
        required=False
    )
    
    # Dynamic column selection field
    column_name = forms.ChoiceField(
        required=True,
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Select the column to apply the rule to'
    )
    
    class Meta:
        model = DataGovernanceRule
        fields = ['name', 'description', 'dataset', 'rule_type', 'column_name', 'parameters']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'dataset': forms.Select(attrs={'class': 'form-control'}),
            'rule_type': forms.Select(attrs={'class': 'form-control', 'id': 'id_rule_type'}),
            'parameters': forms.HiddenInput()
        }
    
    def __init__(self, *args, **kwargs):
        """
        Custom initialization to support optional dataset pre-selection
        and dynamic column selection.
        """
        dataset_id = kwargs.pop('dataset_id', None)
        dataset_columns = kwargs.pop('dataset_columns', [])
        
        super().__init__(*args, **kwargs)
        
        # Create form helper with explicit configuration for bootstrap5
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_class = 'form-horizontal'
        self.helper.label_class = 'col-lg-2'
        self.helper.field_class = 'col-lg-8'
        
        # Limit dataset choices if a specific dataset is pre-selected
        if dataset_id:
            self.fields['dataset'].queryset = Dataset.objects.filter(id=dataset_id)
        
        # Populate column choices dynamically
        if dataset_columns:
            self.fields['column_name'].choices = [(col, col) for col in dataset_columns]
        else:
            # If no columns provided, get columns from all datasets
            all_columns = set()
            for dataset in Dataset.objects.all():
                if dataset.column_names:
                    all_columns.update(dataset.column_names)
            self.fields['column_name'].choices = [(col, col) for col in sorted(all_columns)]
        
        # Add custom layout
        self.helper.layout = Layout(
            Field('name', css_class='form-control'),
            Field('description', css_class='form-control', rows=3),
            Field('dataset', css_class='form-control'),
            Field('rule_type', css_class='form-control'),
            Field('column_name', css_class='form-control'),
            Field('parameters_json', css_class='form-control', rows=3),
            Div(
                Submit('submit', 'Create Rule', css_class='btn btn-primary'),
                css_class='text-center mt-4'
            )
        )
    
    def clean(self):
        """Validate and process form data."""
        cleaned_data = super().clean()
        
        # Convert parameters_json to parameters
        parameters_json = cleaned_data.get('parameters_json', '{}')
        try:
            parameters = json.loads(parameters_json)
            cleaned_data['parameters'] = parameters
        except json.JSONDecodeError:
            raise forms.ValidationError({'parameters_json': 'Invalid JSON format'})
        
        return cleaned_data
    
    def save(self, commit=True):
        """Override save method to handle parameters."""
        instance = super().save(commit=False)
        
        # Set parameters from the JSON input
        instance.parameters = self.cleaned_data.get('parameters', {})
        
        if commit:
            instance.save()
        
        return instance 