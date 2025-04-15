from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.contrib import messages
from django.db.models import ProtectedError
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import reverse_lazy
from .models import Dataset, DataGovernanceRule, RuleValidationResult
from .forms import DatasetUploadForm
from .utils import read_dataset
import pandas as pd
import numpy as np
import logging
import json
import os
import sys
import markdown2
from django.conf import settings
from django.views.decorators.http import require_GET
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q
from django.db.models import F, Count, Case, When, Value, CharField, Max
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Add NpEncoder for JSON serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@login_required
def dataset_list(request):
    """
    List datasets with search and view mode functionality.
    
    Supports:
    - Search by dataset name, description, or file type
    - Toggle between table and card view
    - Pagination
    """
    # Get search parameters
    search_query = request.GET.get('search', '')
    sort_by = request.GET.get('sort', '-created_at')  # Default sort by newest
    view_mode = request.GET.get('view', 'table')
    
    # Base queryset
    queryset = Dataset.objects.all()
    
    # Apply search filters
    if search_query:
        queryset = queryset.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(file_type__icontains=search_query)
        )
    
    # Add annotations for better performance
    queryset = queryset.annotate(
        total_rules=Count('rules', distinct=True),
        last_validation=Max('rulevalidationresult__validation_date'),
        validation_status=Case(
            When(rulevalidationresult__isnull=True, then=Value('Not Validated')),
            When(rulevalidationresult__passed=True, then=Value('Passed')),
            default=Value('Failed'),
            output_field=CharField(),
        )
    )
    
    # Apply sorting
    if sort_by.startswith('-'):
        sort_field = sort_by[1:]
        if sort_field == 'created_at':
            queryset = queryset.order_by('-id')  # Use id as a proxy for creation time
        else:
            queryset = queryset.order_by(F(sort_field).desc(nulls_last=True))
    else:
        if sort_by == 'created_at':
            queryset = queryset.order_by('id')  # Use id as a proxy for creation time
        else:
            queryset = queryset.order_by(F(sort_by).asc(nulls_last=True))
    
    # Pagination
    paginator = Paginator(queryset, 12)  # Show 12 datasets per page
    page = request.GET.get('page')
    try:
        datasets = paginator.page(page)
    except PageNotAnInteger:
        datasets = paginator.page(1)
    except EmptyPage:
        datasets = paginator.page(paginator.num_pages)
    
    context = {
        'datasets': datasets,
        'total_datasets': queryset.count(),
        'search_query': search_query,
        'view_mode': view_mode,
        'sort_by': sort_by,
        'sort_options': [
            {'value': 'name', 'label': 'Name (A-Z)'},
            {'value': '-name', 'label': 'Name (Z-A)'},
            {'value': '-created_at', 'label': 'Newest First'},
            {'value': 'created_at', 'label': 'Oldest First'},
            {'value': '-total_rows', 'label': 'Most Rows'},
            {'value': 'total_rows', 'label': 'Least Rows'},
            {'value': '-total_columns', 'label': 'Most Columns'},
            {'value': 'total_columns', 'label': 'Least Columns'},
        ]
    }
    
    return render(request, 'core/dataset_list.html', context)

class CustomLoginView(LoginView):
    """Custom login view with enhanced functionality."""
    template_name = 'core/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        """Return the URL to redirect to after successful login."""
        next_url = self.request.GET.get('next')
        if next_url:
            return next_url
        return reverse_lazy('core:home')
    
    def form_valid(self, form):
        """Add success message on successful login."""
        response = super().form_valid(form)
        messages.success(self.request, f'Welcome back, {self.request.user.username}!')
        return response

class CustomLogoutView(LogoutView):
    """Custom logout view with enhanced functionality."""
    next_page = 'core:login'
    
    def dispatch(self, request, *args, **kwargs):
        """Add success message on logout."""
        response = super().dispatch(request, *args, **kwargs)
        messages.success(request, 'You have been successfully logged out.')
        return response

@login_required
def delete_rule(request, rule_id):
    """Delete a data governance rule."""
    try:
        # Get the rule or return 404
        rule = get_object_or_404(DataGovernanceRule, id=rule_id)
        
        # Store rule details for the success message
        rule_name = rule.name
        dataset_name = rule.dataset.name if rule.dataset else 'Global Rule'
        
        # Delete associated validation results first
        validation_results = RuleValidationResult.objects.filter(rule=rule)
        validation_results_count = validation_results.count()
        validation_results.delete()
        
        # Delete the rule
        rule.delete()
        
        # Log the deletion
        logger.info(f"Rule '{rule_name}' for dataset '{dataset_name}' deleted by user {request.user.username}")
        
        # Success message with details
        messages.success(
            request, 
            f"Rule '{rule_name}' has been deleted. "
            f"Deleted {validation_results_count} associated validation results."
        )
        
        return redirect('core:rule_list')
    
    except ProtectedError as e:
        # Handle cases where the rule might be referenced by other objects
        messages.error(
            request, 
            f"Cannot delete rule: {str(e)}. "
            "It may be referenced by other objects in the system."
        )
        return redirect('core:rule_detail', rule_id=rule_id)
    
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.error(f"Error deleting rule {rule_id}: {str(e)}")
        messages.error(request, f"Unexpected error deleting rule: {str(e)}")
        return redirect('core:rule_list')

@login_required
@require_http_methods(["GET"])
def grid_data_api(request, dataset_id):
    """API endpoint to get dataset data for AG Grid with improved data handling."""
    try:
        # Get the dataset
        dataset = get_object_or_404(Dataset, id=dataset_id)
        print(f"Loading dataset {dataset.id}: {dataset.name}")
        
        # Ensure the file exists
        if not dataset.file:
            print(f"Dataset {dataset.id} has no file")
            return JsonResponse({
                'status': 'error',
                'error': 'Dataset file not found'
            }, status=404)
        
        print(f"Reading file from: {dataset.file.path}")
        
        # Read the dataset
        df = read_dataset(dataset)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Basic column definitions
        column_defs = []
        for col in df.columns:
            col_def = {
                'field': str(col),
                'headerName': str(col).replace('_', ' ').title(),
                'sortable': True,
                'filter': True,
                'resizable': True,
                'enableRowGroup': True,
                'enableValue': True,
                'enablePivot': True,
                'minWidth': 150
            }
            
            # Get column type
            dtype = str(df[col].dtype)
            print(f"Column {col} has dtype: {dtype}")
            
            # Configure column based on data type
            if 'int' in dtype or 'float' in dtype:
                col_def.update({
                    'filter': 'agNumberColumnFilter',
                    'type': 'numericColumn',
                    'valueFormatter': 'numberFormatter'
                })
            elif 'bool' in dtype:
                col_def.update({
                    'filter': 'agSetColumnFilter',
                    'cellRenderer': 'checkboxRenderer'
                })
            elif 'datetime' in dtype:
                col_def.update({
                    'filter': 'agDateColumnFilter',
                    'type': 'dateColumn',
                    'valueFormatter': 'dateFormatter'
                })
            else:
                col_def.update({
                    'filter': 'agTextColumnFilter'
                })
            
            column_defs.append(col_def)
        
        # Convert data to records, handling special data types
        def convert_value(val):
            if pd.isna(val):
                return None
            elif isinstance(val, (np.integer, np.floating)):
                return float(val) if isinstance(val, np.floating) else int(val)
            elif isinstance(val, np.bool_):
                return bool(val)
            elif isinstance(val, pd.Timestamp):
                return val.isoformat()
            return str(val)

        row_data = []
        for _, row in df.iterrows():
            converted_row = {str(col): convert_value(val) for col, val in row.items()}
            row_data.append(converted_row)
        
        print(f"Converted {len(row_data)} rows to dict format")
        
        response_data = {
            'status': 'success',
            'columnDefs': column_defs,
            'rowData': row_data,
            'metadata': {
                'totalRows': len(df),
                'totalColumns': len(df.columns),
                'numericColumns': [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                'categoricalColumns': [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() <= 50],
                'dateColumns': [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])],
                'booleanColumns': [col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]
            }
        }
        
        print("Sending response with data")
        return JsonResponse(response_data, encoder=NpEncoder)
        
    except Exception as e:
        import traceback
        print(f"Error in grid data api: {str(e)}")
        traceback.print_exc()
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

@login_required
def dataset_upload(request):
    """Handle dataset upload."""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        form.request = request  # Pass request to form for user info
        if form.is_valid():
            dataset = form.save()
            messages.success(request, f'Dataset "{dataset.name}" uploaded successfully.')
            return redirect('core:dataset_detail', dataset_id=dataset.id)
    else:
        form = DatasetUploadForm()

    return render(request, 'core/dataset_upload.html', {
        'form': form,
        'active_page': 'upload'
    })

def documentation_view(request):
    """Render markdown documentation with extensive logging"""
    logger = logging.getLogger(__name__)
    logger.info("Entering documentation_view")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"BASE_DIR: {settings.BASE_DIR}")
    
    # Explicit path to the documentation file
    docs_path = os.path.join(settings.BASE_DIR, 'core', 'docs', 'source', 'index.md')
    
    html_content = "<h1>Documentation Error</h1>"
    
    logger.info(f"Attempting to read documentation from: {docs_path}")
    
    try:
        if os.path.exists(docs_path):
            with open(docs_path, 'r') as doc_file:
                content = doc_file.read()
                html_content = markdown2.markdown(content)
                logger.info(f"Successfully read documentation from {docs_path}")
        else:
            logger.error(f"Documentation file does not exist at {docs_path}")
            html_content = f"<h1>Documentation Not Found</h1><p>File not found: {docs_path}</p>"
    except Exception as e:
        logger.error(f"Error reading documentation: {str(e)}")
        html_content = f"<h1>Documentation Error</h1><p>{str(e)}</p>"
    
    return render(request, 'core/documentation.html', {
        'documentation_content': html_content
    })

@login_required
def dataset_detail(request, dataset_id=None):
    """Display details of a specific dataset or default behavior."""
    if dataset_id:
        # Fetch a specific dataset
        dataset = get_object_or_404(Dataset, id=dataset_id)
    else:
        # If no dataset_id is provided, show the most recent dataset
        dataset = Dataset.objects.order_by('-created_at').first()
        if not dataset:
            messages.warning(request, 'No datasets available.')
            return redirect('core:dataset_upload')

    # Get preview data
    preview_data = dataset.get_preview_data()
    
    # Get recent validation results
    validation_results = RuleValidationResult.objects.filter(dataset=dataset).order_by('-validation_date')[:5]

    return render(request, 'core/dataset_detail.html', {
        'dataset': dataset,
        'preview_data': preview_data,
        'validation_results': validation_results,
        'active_page': 'datasets'
    })

@login_required
def dataset_analyze(request, dataset_id):
    """Analyze a specific dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Perform dataset analysis
    from .tasks import analyze_dataset
    analysis = analyze_dataset(dataset)
    
    return render(request, 'core/dataset_analyze.html', {
        'dataset': dataset,
        'analysis': analysis,
        'active_page': 'analyze'
    })

@login_required
def analyze_with_grid(request, dataset_id):
    """Analyze dataset with AG Grid."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    return render(request, 'core/grid_analysis.html', {
        'dataset': dataset,
        'active_page': 'grid_analysis'
    })

@login_required
def generate_profile(request, dataset_id):
    """Generate a profile for a specific dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    # Perform profile generation
    from .tasks import generate_profile_report
    try:
        result = generate_profile_report(dataset)
        messages.success(request, 'Profile generated successfully.')
    except Exception as e:
        messages.error(request, f'Error generating profile: {str(e)}')
    
    return redirect('core:dataset_detail', dataset_id=dataset_id)

@login_required
def profile_status(request, dataset_id):
    """Check the status of dataset profile generation."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    return JsonResponse({
        'status': dataset.profile_status,
        'last_updated': dataset.profile_last_updated.isoformat() if dataset.profile_last_updated else None
    })

@login_required
def dataset_delete(request, dataset_id):
    """Delete a specific dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    try:
        dataset_name = dataset.name
        dataset.delete()
        messages.success(request, f'Dataset "{dataset_name}" has been deleted.')
        return redirect('core:dataset_list')
    except Exception as e:
        messages.error(request, f'Error deleting dataset: {str(e)}')
        return redirect('core:dataset_detail', dataset_id=dataset_id)

def about(request):
    """Render the about page."""
    return render(request, 'core/about.html', {
        'active_page': 'about'
    })

def dashboard(request):
    """Render the main dashboard."""
    total_datasets = Dataset.objects.count()
    total_rules = DataGovernanceRule.objects.count()
    recent_datasets = Dataset.objects.order_by('-created_at')[:5]
    recent_validations = RuleValidationResult.objects.order_by('-validation_date')[:5]

    return render(request, 'core/dashboard.html', {
        'total_datasets': total_datasets,
        'total_rules': total_rules,
        'recent_datasets': recent_datasets,
        'recent_validations': recent_validations,
        'active_page': 'dashboard'
    })

def generate_profile_api(request, dataset_id):
    """API endpoint to generate profile."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    try:
        from .tasks import generate_profile_report
        result = generate_profile_report(dataset)
        return JsonResponse({
            'status': 'success',
            'message': 'Profile generated successfully',
            'profile_status': dataset.profile_status
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def generate_ydata_profile(request, dataset_id):
    """Generate YData profile for a dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    try:
        from .tasks import generate_profile_report
        result = generate_profile_report(dataset)
        return JsonResponse({
            'status': 'success',
            'profile_url': dataset.profile_report.url if dataset.profile_report else None
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@login_required
def validate_rules(request, dataset_id):
    """Validate all rules for a specific dataset with enhanced error handling and feedback."""
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        rules = DataGovernanceRule.objects.filter(dataset=dataset, is_active=True)
        
        if not rules.exists():
            messages.warning(request, 'No active validation rules found for this dataset.')
            return redirect('core:dataset_detail', dataset_id=dataset_id)
        
        # Get validation history
        validation_history = {}
        for rule in rules:
            history = RuleValidationResult.objects.filter(
                dataset=dataset,
                rule=rule
            ).order_by('-validation_date')[:5]
            validation_history[rule.id] = history
        
        # Perform validation
        from .utils import validate_dataset_rules
        validation_results = validate_dataset_rules(dataset, rules)
        
        # Group results by status
        grouped_results = {
            'passed': [],
            'failed': [],
            'error': []
        }
        
        for result in validation_results:
            status = result['status']
            if status == 'pass':
                grouped_results['passed'].append(result)
            elif status == 'fail':
                grouped_results['failed'].append(result)
            else:
                grouped_results['error'].append(result)
        
        # Calculate summary statistics
        total_rules = len(validation_results)
        passed_count = len(grouped_results['passed'])
        failed_count = len(grouped_results['failed'])
        error_count = len(grouped_results['error'])
        
        success_rate = (passed_count / total_rules * 100) if total_rules > 0 else 0
        
        # Add appropriate messages
        if error_count > 0:
            messages.error(request, f'{error_count} rule(s) encountered errors during validation.')
        if failed_count > 0:
            messages.warning(request, f'{failed_count} rule(s) failed validation.')
        if passed_count > 0:
            messages.success(request, f'{passed_count} rule(s) passed validation.')
        
        return render(request, 'core/validate_rules.html', {
            'dataset': dataset,
            'validation_results': validation_results,
            'grouped_results': grouped_results,
            'validation_history': validation_history,
            'summary': {
                'total_rules': total_rules,
                'passed_count': passed_count,
                'failed_count': failed_count,
                'error_count': error_count,
                'success_rate': success_rate
            },
            'active_page': 'validate'
        })
        
    except Exception as e:
        messages.error(request, f'Error during validation: {str(e)}')
        return redirect('core:dataset_detail', dataset_id=dataset_id)

@login_required
def rule_list(request):
    """Display list of all data governance rules."""
    # By default, show only active rules
    rules = DataGovernanceRule.objects.filter(is_active=True).order_by('-created_at')
    
    # Optional: Add a query parameter to show all rules
    show_all = request.GET.get('show_all', 'false').lower() == 'true'
    if show_all:
        rules = DataGovernanceRule.objects.all().order_by('-created_at')
    
    return render(request, 'core/rule_list.html', {
        'rules': rules,
        'show_all': show_all,
        'active_page': 'rules'
    })

@login_required
def rule_create(request, dataset_id=None):
    """Create a new data governance rule."""
    from .forms import DataGovernanceRuleForm
    
    # Get dataset columns if a dataset is selected
    dataset_columns = []
    if dataset_id:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        dataset_columns = dataset.column_names or []
    
    if request.method == 'POST':
        form = DataGovernanceRuleForm(request.POST, dataset_id=dataset_id, dataset_columns=dataset_columns)
        if form.is_valid():
            rule = form.save(commit=False)
            rule.created_by = request.user
            rule.save()
            messages.success(request, f'Rule "{rule.name}" created successfully.')
            return redirect('core:rule_detail', rule_id=rule.id)
    else:
        form = DataGovernanceRuleForm(dataset_id=dataset_id, dataset_columns=dataset_columns)
    
    return render(request, 'core/rule_create.html', {
        'form': form,
        'dataset_id': dataset_id,
        'active_page': 'rules'
    })

@login_required
def rule_detail(request, rule_id):
    """Display details of a specific data governance rule."""
    rule = get_object_or_404(DataGovernanceRule, id=rule_id)
    
    # Get recent validation results
    validation_results = rule.rulevalidationresult_set.order_by('-validation_date')[:10]
    
    return render(request, 'core/rule_detail.html', {
        'rule': rule,
        'validation_results': validation_results,
        'active_page': 'rules'
    })

@login_required
def rule_delete(request, rule_id):
    """Delete a specific data governance rule."""
    rule = get_object_or_404(DataGovernanceRule, id=rule_id)
    
    try:
        rule_name = rule.name
        rule.delete()
        messages.success(request, f'Rule "{rule_name}" has been deleted.')
        return redirect('core:rule_list')
    except Exception as e:
        messages.error(request, f'Error deleting rule: {str(e)}')
        return redirect('core:rule_detail', rule_id=rule_id)

@login_required
def rule_validate(request, rule_id):
    """Validate a specific rule against its dataset with enhanced error handling."""
    try:
        rule = get_object_or_404(DataGovernanceRule, id=rule_id)
        dataset = rule.dataset

        if not dataset:
            messages.error(request, 'No dataset associated with this rule.')
            return redirect('core:rule_detail', rule_id=rule_id)

        if not dataset.file:
            messages.error(request, 'Dataset file not found.')
            return redirect('core:rule_detail', rule_id=rule_id)

        from .utils import validate_dataset_rules
        try:
            validation_results = validate_dataset_rules(dataset, [rule])
        except Exception as e:
            logger.error(f"Validation error for rule {rule_id}: {str(e)}")
            messages.error(request, f'Error during validation: {str(e)}')
            return redirect('core:rule_detail', rule_id=rule_id)

        # Process validation results
        total_rows = len(validation_results)
        failed_rows = [r for r in validation_results if not r.get('passed', False)]
        passed_rows = total_rows - len(failed_rows)
        pass_rate = (passed_rows / total_rows * 100) if total_rows > 0 else 0

        # Create validation record with detailed results
        validation_record = RuleValidationResult.objects.create(
            dataset=dataset,
            rule=rule,
            passed=len(failed_rows) == 0,
            failed_rows=failed_rows,
            validated_by=request.user,
            total_rows=total_rows,
            passed_rows=passed_rows,
            pass_rate=pass_rate,
            validation_details={
                'rule_type': rule.rule_type,
                'parameters': rule.parameters,
                'column_name': rule.column_name,
                'execution_time': None  # Will be updated in post-processing
            }
        )

        # Paginate failed rows
        paginator = Paginator(failed_rows, 10)  # Show 10 rows per page
        page = request.GET.get('page', 1)
        try:
            failed_rows_page = paginator.page(page)
        except EmptyPage:
            failed_rows_page = paginator.page(paginator.num_pages)

        context = {
            'rule': rule,
            'dataset': dataset,
            'validation_record': validation_record,
            'failed_rows': failed_rows_page,
            'total_rows': total_rows,
            'passed_rows': passed_rows,
            'failed_rows_count': len(failed_rows),
            'pass_rate': pass_rate
        }

        return render(request, 'core/rule_validate.html', context)

    except Exception as e:
        logger.error(f"Unexpected error in rule_validate view: {str(e)}")
        messages.error(request, 'An unexpected error occurred during validation.')
        return redirect('core:rule_list')

@login_required
def validation_results(request):
    """Display all validation results."""
    results = RuleValidationResult.objects.all().order_by('-validation_date')
    
    # Calculate total, passed, and failed results
    total_results = results.count()
    passed_results = results.filter(passed=True).count()
    failed_results = results.filter(passed=False).count()
    
    return render(request, 'core/validation_results.html', {
        'results': results,
        'total_results': total_results,
        'passed_count': passed_results,
        'failed_count': failed_results,
        'active_page': 'validation_results'
    })

@login_required
def validation_dashboard(request):
    """Render the validation dashboard."""
    # Get recent validation results
    recent_results = RuleValidationResult.objects.order_by('-validation_date')[:10]
    
    # Calculate overall validation statistics
    total_results = RuleValidationResult.objects.count()
    passed_results = RuleValidationResult.objects.filter(passed=True).count()
    failed_results = total_results - passed_results
    
    # Create validation stats dictionary
    validation_stats = {
        'total_validations': total_results,
        'passed_validations': passed_results,
        'failed_validations': failed_results
    }
    
    # Get rules with their latest validation status
    rules_with_status = []
    for rule in DataGovernanceRule.objects.all():
        latest_result = rule.rulevalidationresult_set.order_by('-validation_date').first()
        rules_with_status.append({
            'rule': rule,
            'latest_result': latest_result
        })
    
    return render(request, 'core/validation_dashboard.html', {
        'recent_results': recent_results,
        'total_results': total_results,
        'passed_results': passed_results,
        'failed_results': failed_results,
        'validation_stats': validation_stats,
        'rules_with_status': rules_with_status,
        'active_page': 'validation_dashboard'
    })

@login_required
def trigger_all_rules(request):
    """Trigger validation for all rules across all datasets."""
    from .utils import validate_dataset_rules
    
    # Get all datasets and rules
    datasets = Dataset.objects.all()
    rules = DataGovernanceRule.objects.all()
    
    validation_results = []
    
    for dataset in datasets:
        dataset_rules = [rule for rule in rules if rule.dataset == dataset or rule.dataset is None]
        dataset_results = validate_dataset_rules(dataset, dataset_rules)
        validation_results.extend(dataset_results)
    
    messages.success(request, f'Validated {len(datasets)} datasets with {len(rules)} rules.')
    return redirect('core:validation_dashboard')

@require_GET
def get_dataset_columns(request, dataset_id):
    """
    Fetch column names for a specific dataset.
    
    This view is used to dynamically populate column names 
    when a dataset is selected in the rule creation form.
    """
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Retrieve column names, handling potential None case
        column_names = dataset.column_names or []
        
        return JsonResponse(column_names, safe=False)
    except Dataset.DoesNotExist:
        return JsonResponse([], safe=False)

@login_required
def get_row_details(request, row_number):
    """API endpoint to get detailed information about a specific row."""
    try:
        dataset_id = request.GET.get('dataset_id')
        if not dataset_id:
            return JsonResponse({'error': 'Dataset ID is required'}, status=400)

        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Read the specific row from the dataset
        df = read_dataset(dataset)
        if row_number >= len(df):
            return JsonResponse({'error': 'Row number out of range'}, status=400)

        row_data = df.iloc[row_number].to_dict()
        
        # Add metadata about the row
        metadata = {
            'row_number': row_number,
            'total_rows': len(df),
            'column_count': len(df.columns),
            'null_count': df.iloc[row_number].isna().sum(),
            'dataset_name': dataset.name
        }

        return JsonResponse({
            'row_data': row_data,
            'metadata': metadata
        })

    except Exception as e:
        logger.error(f"Error fetching row details: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def export_validation_results(request):
    """Export validation results to CSV with error handling."""
    try:
        if request.method != 'POST':
            return JsonResponse({'error': 'Method not allowed'}, status=405)

        data = json.loads(request.body)
        rule_id = data.get('ruleId')
        validation_id = data.get('validationId')

        if not rule_id or not validation_id:
            return JsonResponse({'error': 'Rule ID and Validation ID are required'}, status=400)

        validation_record = get_object_or_404(
            RuleValidationResult, 
            id=validation_id,
            rule_id=rule_id
        )

        # Convert failed rows to DataFrame for export
        failed_rows_df = pd.DataFrame(validation_record.failed_rows)
        
        # Add metadata columns
        failed_rows_df['validation_date'] = validation_record.validation_date
        failed_rows_df['rule_name'] = validation_record.rule.name
        failed_rows_df['dataset_name'] = validation_record.dataset.name

        # Generate CSV
        csv_content = failed_rows_df.to_csv(index=False)
        
        response = JsonResponse(csv_content, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="validation_results_{validation_id}.csv"'
        return response

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        logger.error(f"Error exporting validation results: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def bulk_validate_rules(request, dataset_id):
    """Validate multiple rules for a dataset with enhanced error handling."""
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        rules = DataGovernanceRule.objects.filter(
            Q(dataset=dataset) | Q(dataset__isnull=True)
        )

        if not rules.exists():
            messages.warning(request, 'No rules found for validation.')
            return redirect('core:dataset_detail', dataset_id=dataset_id)

        validation_results = []
        failed_validations = []

        for rule in rules:
            try:
                from .utils import validate_dataset_rules
                results = validate_dataset_rules(dataset, [rule])
                
                # Process results
                total_rows = len(results)
                failed_rows = [r for r in results if not r.get('passed', False)]
                passed_rows = total_rows - len(failed_rows)
                pass_rate = (passed_rows / total_rows * 100) if total_rows > 0 else 0

                # Create validation record
                validation_record = RuleValidationResult.objects.create(
                    dataset=dataset,
                    rule=rule,
                    passed=len(failed_rows) == 0,
                    failed_rows=failed_rows,
                    validated_by=request.user,
                    total_rows=total_rows,
                    passed_rows=passed_rows,
                    pass_rate=pass_rate
                )

                validation_results.append(validation_record)

            except Exception as e:
                logger.error(f"Error validating rule {rule.id}: {str(e)}")
                failed_validations.append({
                    'rule': rule,
                    'error': str(e)
                })

        # Prepare summary message
        success_count = len(validation_results)
        failed_count = len(failed_validations)
        
        if failed_validations:
            messages.warning(
                request,
                f'Completed {success_count} validations with {failed_count} failures. '
                'Check the results for details.'
            )
        else:
            messages.success(
                request,
                f'Successfully completed {success_count} validations.'
            )

        context = {
            'dataset': dataset,
            'validation_results': validation_results,
            'failed_validations': failed_validations,
            'total_rules': rules.count(),
            'success_count': success_count,
            'failed_count': failed_count
        }

        return render(request, 'core/bulk_validation_results.html', context)

    except Exception as e:
        logger.error(f"Unexpected error in bulk_validate_rules: {str(e)}")
        messages.error(request, 'An unexpected error occurred during bulk validation.')
        return redirect('core:dataset_detail', dataset_id=dataset_id)

def test_advanced_filtering(self):
    """Test advanced filtering in dataset list view."""
    response = self.client.get(
        reverse('core:dataset_list'),
        {'filter': 'created_date', 'order': '-created_date'}
    )
    self.assertEqual(response.status_code, 200)
    self.assertTrue('dataset_list' in response.context)

def test_complete_workflow(self):
    """Test complete workflow from dataset upload to validation."""
    # Upload dataset
    dataset = self.create_test_dataset()
    
    # Create rule
    rule = self.create_test_rule(dataset)
    
    # Run validation
    response = self.client.post(
        reverse('core:validate_rules', args=[dataset.id])
    )
    self.assertEqual(response.status_code, 200)
    
    # Check results
    validation_result = RuleValidationResult.objects.get(
        dataset=dataset,
        rule=rule
    )
    self.assertIsNotNone(validation_result)

def test_api_security(self):
    """Test API endpoint security."""
    # Test unauthorized access
    self.client.logout()
    response = self.client.get(
        reverse('core:api_dataset_list'),
        HTTP_X_REQUESTED_WITH='XMLHttpRequest'
    )
    self.assertEqual(response.status_code, 403)
    
    # Test with invalid token
    response = self.client.get(
        reverse('core:api_dataset_list'),
        HTTP_X_API_TOKEN='invalid_token'
    )
    self.assertEqual(response.status_code, 401)

def test_error_handling(self):
    """Test error handling for edge cases."""
    # Test invalid file format
    response = self.client.post(
        reverse('core:dataset_upload'),
        {'file': SimpleUploadedFile('test.txt', b'invalid data')}
    )
    self.assertEqual(response.status_code, 400)
    
    # Test missing required fields
    response = self.client.post(
        reverse('core:rule_create'),
        {}
    )
    self.assertEqual(response.status_code, 400)