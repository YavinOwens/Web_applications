import pandas as pd
import logging
from django.conf import settings
import os
from ..models import Dataset

logger = logging.getLogger(__name__)

def generate_dataset_profile(dataset, config=None):
    """Generate profile for a dataset with error handling."""
    try:
        from ydata_profiling import ProfileReport
        
        # Read dataset
        df = pd.read_csv(dataset.file.path) if dataset.file_type == 'csv' else pd.read_excel(dataset.file.path)
        
        # Generate profile
        profile = ProfileReport(df, title=f"Profile Report for {dataset.name}", **config or {})
        
        # Save HTML report
        report_path = os.path.join(settings.MEDIA_ROOT, f'profiles/{dataset.id}_profile.html')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        profile.to_file(report_path)
        
        # Save JSON
        json_path = os.path.join(settings.MEDIA_ROOT, f'profiles/{dataset.id}_profile.json')
        profile.to_file(json_path)
        
        # Update dataset
        dataset.profile_report.name = f'profiles/{dataset.id}_profile.html'
        dataset.profile_json.name = f'profiles/{dataset.id}_profile.json'
        dataset.profile_status = 'ready'
        dataset.save()
        
        return {
            'status': 'success',
            'profile_url': dataset.profile_report.url,
            'json_url': dataset.profile_json.url
        }
        
    except Exception as e:
        logger.error(f"Error generating profile for dataset {dataset.id}: {str(e)}")
        dataset.profile_status = 'failed'
        dataset.save()
        raise

def get_profile_preview(dataset, num_rows=5):
    """Get a preview of dataset for quick viewing."""
    try:
        if not dataset.file:
            return None
            
        df = pd.read_csv(dataset.file.path) if dataset.file_type == 'csv' else pd.read_excel(dataset.file.path)
        preview = df.head(num_rows).to_dict('records')
        
        return {
            'columns': list(df.columns),
            'rows': preview,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Error getting preview for dataset {dataset.id}: {str(e)}")
        return None 