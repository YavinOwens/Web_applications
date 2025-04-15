from django import template
from django.template.defaultfilters import floatformat
from django.contrib.humanize.templatetags.humanize import intcomma
import json

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary using bracket notation."""
    return dictionary.get(key)

@register.filter
def format_value(value):
    """Template filter to format numeric values."""
    if isinstance(value, (int, float)):
        if abs(value) >= 1000000:
            return f"{value/1000000:.1f}M"
        elif abs(value) >= 1000:
            return f"{value/1000:.1f}K"
        elif isinstance(value, float):
            return f"{value:.2f}"
        else:
            return intcomma(value)
    return str(value)

@register.filter
def format_number(value):
    """Format a number with commas and handle large values."""
    try:
        value = float(value)
        if abs(value) >= 1000000000:  # Billions
            return f"{value/1000000000:.1f}B"
        elif abs(value) >= 1000000:    # Millions
            return f"{value/1000000:.1f}M"
        elif abs(value) >= 1000:       # Thousands
            return f"{value/1000:.1f}K"
        elif isinstance(value, float):
            return f"{value:.2f}"
        else:
            return intcomma(int(value))
    except (ValueError, TypeError):
        return str(value)

@register.filter
def divide(value, arg):
    """Divide the value by the argument."""
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError):
        return 0

@register.filter
def multiply(value, arg):
    """Multiply the value by the argument."""
    try:
        return float(value) * float(arg)
    except ValueError:
        return 0

@register.filter
def percentage(value):
    """Convert a decimal to a percentage."""
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "0%"

@register.filter
def format_bytes(bytes):
    """Format bytes to human readable size."""
    try:
        bytes = float(bytes)
        if bytes >= 1099511627776:
            terabytes = bytes / 1099511627776
            size = '%.2fTB' % terabytes
        elif bytes >= 1073741824:
            gigabytes = bytes / 1073741824
            size = '%.2fGB' % gigabytes
        elif bytes >= 1048576:
            megabytes = bytes / 1048576
            size = '%.2fMB' % megabytes
        elif bytes >= 1024:
            kilobytes = bytes / 1024
            size = '%.2fKB' % kilobytes
        else:
            size = '%.2fB' % bytes
        return size
    except:
        return '0B'

@register.filter
def pprint(value):
    """Pretty print a JSON object."""
    try:
        if isinstance(value, str):
            value = json.loads(value)
        return json.dumps(value, indent=2)
    except (json.JSONDecodeError, TypeError):
        return str(value) 