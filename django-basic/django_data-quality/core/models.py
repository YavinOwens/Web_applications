from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
import json
from django.utils import timezone
import os
from .utils import validate_profile_config, validate_profile_status, validate_file_content, validate_file_extension
import re
from django.urls import reverse
from django.template.defaultfilters import filesizeformat
import polars as pl
from django.conf import settings

class Dataset(models.Model):
    """Model to store uploaded datasets."""
    FILE_TYPE_CHOICES = [
        ('csv', 'CSV'),
        ('excel', 'Excel'),
        ('json', 'JSON')
    ]
    
    PROFILE_STATUS_CHOICES = [
        ('not_generated', 'Not Generated'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to='datasets/')
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Profile related fields
    profile_status = models.CharField(
        max_length=20,
        choices=PROFILE_STATUS_CHOICES,
        default='not_generated'
    )
    profile_last_updated = models.DateTimeField(null=True, blank=True)
    profile_config = models.JSONField(null=True, blank=True)
    profile_report = models.FileField(upload_to='profiles/reports/', null=True, blank=True)
    profile_json = models.FileField(upload_to='profiles/json/', null=True, blank=True)
    
    # Metadata fields
    total_rows = models.IntegerField(null=True, blank=True)
    total_columns = models.IntegerField(null=True, blank=True)
    column_names = models.JSONField(null=True, blank=True)
    column_types = models.JSONField(null=True, blank=True)
    is_sensitive = models.BooleanField(default=False, help_text="Mark if dataset contains sensitive information")
    uploaded_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='uploaded_datasets',
        help_text="User who uploaded the dataset"
    )

    def __str__(self):
        return self.name

    def delete(self, *args, **kwargs):
        """Delete the dataset and its associated file."""
        if self.file:
            self.file.delete(save=False)
        super().delete(*args, **kwargs)

    def clean(self):
        """Validate the dataset."""
        if not self.file:
            raise ValidationError({'file': 'This field cannot be blank.'})
        
        # Validate file type
        if self.file_type not in [choice[0] for choice in self.FILE_TYPE_CHOICES]:
            raise ValidationError({'file_type': 'Invalid file type. Must be one of: csv, excel, json'})

    def save(self, *args, **kwargs):
        """Save the dataset after validation."""
        self.full_clean()
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('core:dataset_detail', args=[str(self.id)])
    
    def get_file_size(self):
        """Return the file size in a human-readable format."""
        if self.file and hasattr(self.file, 'size'):
            return filesizeformat(self.file.size)
        return '0 B'
    
    def get_preview_data(self, num_rows=5):
        """Get preview data from the dataset file."""
        if not self.file:
            return None
            
        try:
            if self.file_type == 'csv':
                df = pl.read_csv(self.file.path, infer_schema_length=None)
            elif self.file_type == 'excel':
                df = pl.read_excel(self.file.path)
            elif self.file_type == 'json':
                df = pl.read_json(self.file.path)
            else:
                return None

            preview_df = df.head(num_rows)
            
            # Update metadata
            self.total_rows = len(df)
            self.total_columns = len(df.columns)
            self.column_names = df.columns
            self.column_types = {col: str(df[col].dtype) for col in df.columns}
            self.save(update_fields=['total_rows', 'total_columns', 'column_names', 'column_types'])
            
            return {
                'columns': preview_df.columns,
                'rows': preview_df.rows(),
                'total_rows': self.total_rows,
                'total_columns': self.total_columns,
                'column_types': self.column_types
            }
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return None

    def update_profile_status(self, status):
        """Update the profile status and handle related changes."""
        if status not in dict(self.PROFILE_STATUS_CHOICES):
            raise ValueError(f'Invalid status: {status}')
        
        self.profile_status = status
        if status == 'completed':
            self.profile_last_updated = timezone.now()
            
        self.save(update_fields=['profile_status', 'profile_last_updated'])

    def update_profile_files(self, report_file=None, json_data=None):
        """Update profile files with proper cleanup."""
        if report_file:
            if self.profile_report:
                if os.path.exists(self.profile_report.path):
                    os.remove(self.profile_report.path)
            self.profile_report = report_file
            
        if json_data:
            if self.profile_json:
                if os.path.exists(self.profile_json.path):
                    os.remove(self.profile_json.path)
            self.profile_json = json_data
            
        self.profile_last_updated = timezone.now()
        self.save()

    def update_profile_config(self, config):
        """Update profile configuration with validation."""
        try:
            validate_profile_config(config)
            self.profile_config = config
            self.save(update_fields=['profile_config'])
        except ValidationError as e:
            raise ValidationError(f'Invalid profile configuration: {str(e)}')

    def get_file_type_display(self):
        return dict(self.FILE_TYPE_CHOICES).get(self.file_type, self.file_type)

    def update_metadata(self, total_rows=None, total_columns=None, column_names=None, column_types=None):
        """Update dataset metadata after file upload or analysis."""
        if total_rows is not None:
            self.total_rows = total_rows
        if total_columns is not None:
            self.total_columns = total_columns
        if column_names is not None:
            self.column_names = column_names
        if column_types is not None:
            self.column_types = column_types
        self.save()

    def generate_profile(self):
        """
        Generate a comprehensive profile for the dataset.
        
        Returns:
            Dictionary containing dataset profile information
        """
        import logging
        from .utils import read_dataset
        import numpy as np
        import pandas as pd
        
        logger = logging.getLogger(__name__)
        
        try:
            # Read the dataset
            df = read_dataset(self)
            
            # Prepare profile data
            profile_data = {
                'metadata': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'file_type': self.file_type,
                    'file_size': self.get_file_size()
                },
                'columns': {},
                'missing_values': {},
                'data_types': {}
            }
            
            # Analyze each column
            for column in df.columns:
                # Determine column type
                try:
                    column_data = df[column]
                    
                    # Determine data type
                    if pd.api.types.is_numeric_dtype(column_data):
                        data_type = 'numeric'
                        # Add checks to prevent empty slice calculations
                        min_val = float(column_data.min()) if not pd.isnull(column_data.min()) and len(column_data) > 0 else None
                        max_val = float(column_data.max()) if not pd.isnull(column_data.max()) and len(column_data) > 0 else None
                        mean_val = float(column_data.mean()) if not pd.isnull(column_data.mean()) and len(column_data) > 0 else None
                        median_val = float(column_data.median()) if not pd.isnull(column_data.median()) and len(column_data) > 0 else None
                        std_val = float(column_data.std()) if not pd.isnull(column_data.std()) and len(column_data) > 0 else None
                        
                        profile_data['columns'][column] = {
                            'type': data_type,
                            'min': min_val,
                            'max': max_val,
                            'mean': mean_val,
                            'median': median_val,
                            'std': std_val
                        }
                    elif pd.api.types.is_datetime64_any_dtype(column_data):
                        data_type = 'datetime'
                        profile_data['columns'][column] = {
                            'type': data_type,
                            'min_date': str(column_data.min()) if not pd.isnull(column_data.min()) else None,
                            'max_date': str(column_data.max()) if not pd.isnull(column_data.max()) else None
                        }
                    elif pd.api.types.is_categorical_dtype(column_data):
                        data_type = 'categorical'
                        profile_data['columns'][column] = {
                            'type': data_type,
                            'unique_values': column_data.nunique(),
                            'top_values': column_data.value_counts().head(5).to_dict()
                        }
                    else:
                        data_type = 'string'
                        profile_data['columns'][column] = {
                            'type': data_type,
                            'unique_values': column_data.nunique(),
                            'sample_values': column_data.dropna().sample(min(5, len(column_data))).tolist()
                        }
                    
                    # Missing values
                    missing_count = column_data.isnull().sum()
                    profile_data['missing_values'][column] = {
                        'count': int(missing_count),
                        'percentage': float(missing_count / len(df) * 100)
                    }
                    
                    # Data type
                    profile_data['data_types'][column] = data_type
                
                except Exception as col_error:
                    logger.warning(f"Error profiling column {column}: {str(col_error)}")
                    profile_data['columns'][column] = {'type': 'error', 'error': str(col_error)}
            
            # Update profile status
            self.update_profile_status('completed')
            
            return profile_data
        
        except Exception as e:
            logger.error(f"Error generating profile for dataset {self.id}: {str(e)}")
            self.update_profile_status('failed')
            return None

class DataQualityAnalysis(models.Model):
    """Model for storing dataset analysis results."""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    numeric_stats = models.JSONField(null=True, blank=True)
    correlation_matrix = models.JSONField(null=True, blank=True)
    missing_value_stats = models.JSONField(null=True, blank=True)
    outlier_stats = models.JSONField(null=True, blank=True)

    class Meta:
        verbose_name_plural = 'Data quality analyses'
        ordering = ['-created_at']

    def __str__(self):
        return f'Analysis for {self.dataset} ({self.created_at})'

class DataGovernanceRule(models.Model):
    """Model for data governance rules."""
    RULE_TYPES = [
        ('format', 'Format'),
        ('range', 'Range'),
        ('required', 'Required'),
        ('unique', 'Unique'),
        ('categorical', 'Categorical'),
        ('date_format', 'Date Format'),
        ('cross_column', 'Cross Column'),
    ]
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    rule_type = models.CharField(max_length=50, choices=RULE_TYPES)
    column_name = models.CharField(max_length=255)
    parameters = models.JSONField(default=dict, blank=True)
    dataset = models.ForeignKey('Dataset', on_delete=models.CASCADE, related_name='rules')
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True, help_text="Whether the rule is currently active")
    
    class Meta:
        ordering = ['name']
        indexes = [
            models.Index(fields=['dataset', 'rule_type']),
            models.Index(fields=['column_name']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.get_rule_type_display()})"
    
    def clean(self):
        """Validate rule parameters based on rule type."""
        super().clean()
        
        if not self.parameters:
            self.parameters = {}
            
        if self.rule_type == 'format':
            if 'pattern' not in self.parameters:
                raise ValidationError({'parameters': 'Format rule must have a pattern parameter'})
            try:
                re.compile(self.parameters['pattern'])
            except re.error:
                raise ValidationError({'parameters': 'Invalid regular expression pattern'})
                
        elif self.rule_type == 'range':
            if 'min' in self.parameters:
                try:
                    float(self.parameters['min'])
                except (TypeError, ValueError):
                    raise ValidationError({'parameters': 'Min value must be numeric'})
            if 'max' in self.parameters:
                try:
                    float(self.parameters['max'])
                except (TypeError, ValueError):
                    raise ValidationError({'parameters': 'Max value must be numeric'})
                    
        elif self.rule_type == 'categorical':
            if 'allowed_values' not in self.parameters:
                raise ValidationError({'parameters': 'Categorical rule must have allowed_values parameter'})
            if not isinstance(self.parameters['allowed_values'], list):
                raise ValidationError({'parameters': 'Allowed values must be a list'})
            if not self.parameters['allowed_values']:
                raise ValidationError({'parameters': 'Allowed values list cannot be empty'})
                
        elif self.rule_type == 'date_format':
            if 'format' in self.parameters:
                try:
                    datetime.strptime('2000-01-01', self.parameters['format'])
                except ValueError:
                    raise ValidationError({'parameters': 'Invalid date format string'})
                    
        elif self.rule_type == 'cross_column':
            required_params = ['comparison_column', 'operator']
            if not all(param in self.parameters for param in required_params):
                raise ValidationError({'parameters': f'Cross-column rule must have parameters: {", ".join(required_params)}'})
            valid_operators = ['==', '>', '>=', '<', '<=']
            if self.parameters['operator'] not in valid_operators:
                raise ValidationError({'parameters': f'Invalid operator. Must be one of: {", ".join(valid_operators)}'})
    
    def get_validation_results(self, limit=10):
        """Get the most recent validation results for this rule."""
        return self.rulevalidationresult_set.all()[:limit]
    
    def get_latest_validation(self):
        """Get the most recent validation result."""
        return self.rulevalidationresult_set.first()
    
    def get_validation_success_rate(self, days=30):
        """Calculate the success rate of validations over the past N days."""
        from django.utils import timezone
        start_date = timezone.now() - timezone.timedelta(days=days)
        results = self.rulevalidationresult_set.filter(validation_date__gte=start_date)
        total = results.count()
        if total == 0:
            return 0
        passed = results.filter(passed=True).count()
        return (passed / total) * 100

class RuleValidationResult(models.Model):
    """Model to store rule validation results."""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    rule = models.ForeignKey(DataGovernanceRule, on_delete=models.CASCADE)
    validation_date = models.DateTimeField(auto_now_add=True)
    passed = models.BooleanField()
    failed_rows = models.JSONField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    execution_time = models.FloatField(null=True, blank=True)
    validated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    
    class Meta:
        ordering = ['-validation_date']
        indexes = [
            models.Index(fields=['dataset', 'rule', 'validation_date']),
            models.Index(fields=['passed']),
        ]
    
    def __str__(self):
        status = 'Passed' if self.passed else 'Failed'
        return f"{self.rule.name} validation for {self.dataset.name} - {status}"
    
    def get_failed_rows_count(self):
        """Get the number of failed rows."""
        if self.failed_rows:
            return len(self.failed_rows)
        return 0
    
    def get_failure_rate(self):
        """Calculate the failure rate as a percentage."""
        if self.failed_rows and self.dataset.total_rows:
            return (len(self.failed_rows) / self.dataset.total_rows) * 100
        return 0
    
    def get_execution_time_display(self):
        """Get a human-readable execution time."""
        if self.execution_time is None:
            return 'N/A'
        if self.execution_time < 1:
            return f"{self.execution_time * 1000:.0f}ms"
        return f"{self.execution_time:.2f}s"
    
    def save_validation_result(self, passed, failed_rows=None, error_message=None, execution_time=None):
        """Save the validation result with all related information."""
        self.passed = passed
        self.failed_rows = failed_rows
        self.error_message = error_message
        self.execution_time = execution_time
        self.save()
    
    @classmethod
    def get_latest_results(cls, dataset):
        """Get the latest validation results for a dataset."""
        return cls.objects.filter(dataset=dataset).select_related('rule').order_by('rule', '-validation_date').distinct('rule')
    
    @classmethod
    def get_validation_history(cls, dataset, rule, days=30):
        """Get validation history for a specific rule and dataset."""
        from django.utils import timezone
        start_date = timezone.now() - timezone.timedelta(days=days)
        return cls.objects.filter(
            dataset=dataset,
            rule=rule,
            validation_date__gte=start_date
        ).order_by('validation_date') 