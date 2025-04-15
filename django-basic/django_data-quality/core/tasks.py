import pandas as pd
import json
import os
from django.core.files.base import ContentFile
from django.utils import timezone
from .models import Dataset, DataQualityAnalysis
from .utils import read_dataset_file, validate_profile_config, polars_to_pandas
import polars as pl
from ydata_profiling import ProfileReport
import logging
import traceback

def generate_profile_report(dataset, config=None):
    """Generate a profile report for the dataset."""
    logger = logging.getLogger('core.tasks')
    
    try:
        # Extensive logging
        logger.info(f"Starting profile generation for dataset: {dataset.name}")
        logger.info(f"Dataset file path: {dataset.file.path}")
        
        # Use Django's update method to avoid threading issues
        Dataset.objects.filter(id=dataset.id).update(
            profile_status='processing', 
            profile_last_updated=timezone.now()
        )

        # Read the dataset with Polars
        try:
            df_polars = read_dataset_file(dataset)
            logger.info(f"Dataset loaded successfully with Polars. Shape: {df_polars.shape}")
            
            # Convert Polars DataFrame to Pandas
            df = polars_to_pandas(df_polars)
            logger.info(f"Converted to Pandas DataFrame. Shape: {df.shape}")
        except Exception as read_error:
            logger.error(f"Error reading dataset: {str(read_error)}")
            logger.error(traceback.format_exc())
            
            # Update dataset status to failed
            Dataset.objects.filter(id=dataset.id).update(
                profile_status='failed', 
                profile_last_updated=timezone.now()
            )
            
            return {
                'success': False,
                'message': f'Dataset read failed: {str(read_error)}',
                'error_details': traceback.format_exc()
            }

        # Prepare profile configuration
        if not config:
            config = {
                'minimal': False,
                'sample_size': -1,
                'correlations': {'pearson': True, 'spearman': True, 'kendall': True},
                'missing_diagrams': {'matrix': True, 'bar': True, 'heatmap': True},
                'duplicates': {'head': True}
            }
        else:
            try:
                validate_profile_config(config)
            except Exception as config_error:
                logger.error(f"Invalid profile configuration: {str(config_error)}")
                
                # Update dataset status to failed
                Dataset.objects.filter(id=dataset.id).update(
                    profile_status='failed', 
                    profile_last_updated=timezone.now()
                )
                
                return {
                    'success': False,
                    'message': f'Invalid profile configuration: {str(config_error)}',
                    'error_details': traceback.format_exc()
                }

        # Apply sampling if configured
        if config['sample_size'] > 0:
            df = df.sample(n=min(config['sample_size'], len(df)))
            logger.info(f"Sampled dataset. New shape: {df.shape}")

        # Generate profile report with timeout and memory management
        try:
            profile = ProfileReport(
                df, 
                minimal=config['minimal'], 
                title=f'Profile for {dataset.name}',
                explorative=True,
                # Add memory and performance settings
                pool_size=4,  # Limit parallel processing
                samples=None  # Disable random sampling
            )
            logger.info("Profile report generated successfully")
        except MemoryError:
            logger.error("Memory error during profile generation. Trying minimal mode.")
            profile = ProfileReport(df, minimal=True)
        except Exception as profile_error:
            logger.error(f"Error generating profile: {str(profile_error)}")
            logger.error(traceback.format_exc())
            
            # Update dataset status to failed
            Dataset.objects.filter(id=dataset.id).update(
                profile_status='failed', 
                profile_last_updated=timezone.now()
            )
            
            return {
                'success': False,
                'message': f'Profile generation failed: {str(profile_error)}',
                'error_details': traceback.format_exc()
            }

        # Save HTML report
        report_content = profile.to_html()
        report_file = ContentFile(report_content.encode('utf-8'))
        report_filename = f'profile_{dataset.id}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.html'

        # Save JSON report
        json_data = profile.to_json()
        json_file = ContentFile(json_data.encode('utf-8'))
        json_filename = f'profile_{dataset.id}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.json'

        # Refresh dataset to get latest state
        dataset.refresh_from_db()

        # Update dataset with new profile
        dataset.profile_report.save(report_filename, report_file)
        dataset.profile_json.save(json_filename, json_file)
        
        # Use update method again to avoid threading issues
        Dataset.objects.filter(id=dataset.id).update(
            profile_status='completed', 
            profile_last_updated=timezone.now(),
            profile_config=config
        )

        logger.info(f"Profile generation completed for dataset: {dataset.name}")

        return {
            'success': True,
            'message': 'Profile report generated successfully',
            'report_path': dataset.profile_report.url if dataset.profile_report else None
        }

    except Exception as e:
        # Comprehensive error logging
        logger.error(f"Unexpected error in profile generation for dataset {dataset.name}")
        logger.error(f"Error details: {str(e)}")
        logger.error(traceback.format_exc())

        # Use update method to set failed status
        Dataset.objects.filter(id=dataset.id).update(
            profile_status='failed', 
            profile_last_updated=timezone.now()
        )

        return {
            'success': False,
            'message': f'Unexpected error in profile generation: {str(e)}',
            'error_details': traceback.format_exc()
        }

def analyze_dataset(dataset, include_correlations=True, include_missing_analysis=True, include_outliers=True):
    """Analyze a dataset and store the results using Polars."""
    try:
        # Read the dataset with Polars
        df_polars = read_dataset_file(dataset)
        
        # Convert to Pandas for analysis and model saving
        df = polars_to_pandas(df_polars)
        
        # Create analysis object with basic stats
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'Int64', 'float32']).columns
        numeric_stats = {}
        for col in numeric_cols:
            try:
                numeric_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
            except Exception as e:
                print(f"Error calculating stats for column {col}: {str(e)}")
                continue
        
        analysis = DataQualityAnalysis.objects.create(
            dataset=dataset,
            numeric_stats=numeric_stats
        )
        
        # Perform analysis based on configuration
        if include_correlations:
            if len(numeric_cols) > 1:
                try:
                    correlation_matrix = df[numeric_cols].corr().to_dict()
                    analysis.correlation_matrix = correlation_matrix
                except Exception as e:
                    print(f"Error calculating correlation matrix: {str(e)}")
        
        if include_missing_analysis:
            missing_analysis = {}
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_analysis[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_count / len(df) * 100)
                }
            analysis.missing_analysis = missing_analysis
        
        if include_outliers:
            outliers = {}
            for col in numeric_cols:
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                    outlier_count = outlier_mask.sum()
                    outliers[col] = {
                        'outlier_count': int(outlier_count),
                        'outlier_percentage': float(outlier_count / len(df) * 100)
                    }
                except Exception as e:
                    print(f"Error calculating outliers for column {col}: {str(e)}")
                    continue
            analysis.outliers = outliers
        
        # Save all changes
        analysis.save()
        return analysis
        
    except Exception as e:
        # Log the error and return None
        print(f"Error analyzing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 