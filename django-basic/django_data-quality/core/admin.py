from django.contrib import admin
from .models import Dataset, DataGovernanceRule, DataQualityAnalysis

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'file_type', 'created_at', 'updated_at']
    list_filter = ['file_type', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(DataGovernanceRule)
class DataGovernanceRuleAdmin(admin.ModelAdmin):
    list_display = ['name', 'rule_type', 'created_at']
    list_filter = ['rule_type', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(DataQualityAnalysis)
class DataQualityAnalysisAdmin(admin.ModelAdmin):
    """Admin configuration for DataQualityAnalysis model."""
    list_display = ['dataset', 'created_at']
    list_filter = ['created_at']
    readonly_fields = ['dataset', 'created_at', 'numeric_stats', 'correlation_matrix', 
                      'missing_value_stats', 'outlier_stats']
    
    def has_add_permission(self, request):
        return False  # Analysis should only be created through the application 