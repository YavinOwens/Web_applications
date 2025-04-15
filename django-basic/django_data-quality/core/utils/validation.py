from django.core.paginator import Paginator, EmptyPage
from ..models import RuleValidationResult
import logging

logger = logging.getLogger(__name__)

def process_validation_results(results, dataset, rule, user=None):
    """Process validation results and create validation record."""
    total_rows = len(results)
    failed_rows = [r for r in results if not r.get('passed', False)]
    passed_rows = total_rows - len(failed_rows)
    pass_rate = (passed_rows / total_rows * 100) if total_rows > 0 else 0

    validation_record = RuleValidationResult.objects.create(
        dataset=dataset,
        rule=rule,
        passed=len(failed_rows) == 0,
        failed_rows=failed_rows,
        validated_by=user,
        total_rows=total_rows,
        passed_rows=passed_rows,
        pass_rate=pass_rate,
        validation_details={
            'rule_type': rule.rule_type,
            'parameters': rule.parameters,
            'column_name': rule.column_name,
            'execution_time': None
        }
    )

    return {
        'validation_record': validation_record,
        'total_rows': total_rows,
        'passed_rows': passed_rows,
        'failed_rows': failed_rows,
        'pass_rate': pass_rate
    }

def paginate_failed_rows(failed_rows, page, per_page=10):
    """Paginate failed rows."""
    paginator = Paginator(failed_rows, per_page)
    try:
        return paginator.page(page)
    except EmptyPage:
        return paginator.page(paginator.num_pages)

def get_validation_statistics(queryset=None):
    """Get validation statistics from a queryset or all results."""
    if queryset is None:
        queryset = RuleValidationResult.objects.all()
    
    total_results = queryset.count()
    passed_results = queryset.filter(passed=True).count()
    failed_results = total_results - passed_results
    
    return {
        'total_validations': total_results,
        'passed_validations': passed_results,
        'failed_validations': failed_results,
        'pass_rate': (passed_results / total_results * 100) if total_results > 0 else 0
    } 