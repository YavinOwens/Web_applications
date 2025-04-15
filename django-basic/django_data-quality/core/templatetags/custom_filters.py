from django import template

register = template.Library()

@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Custom template filter to safely access dictionary items.
    
    Usage in template:
    {{ dictionary|get_item:key }}
    
    Args:
        dictionary: The dictionary to access
        key: The key to retrieve
    
    Returns:
        Value for the key or None if key doesn't exist
    """
    if not isinstance(dictionary, dict):
        return None
    
    # Handle nested dictionary access
    if isinstance(key, str) and '.' in key:
        keys = key.split('.')
        for k in keys:
            if isinstance(dictionary, dict) and k in dictionary:
                dictionary = dictionary[k]
            else:
                return None
        return dictionary
    
    return dictionary.get(key, None)

@register.filter(name='safe_json')
def safe_json(value):
    """
    Convert a value to a JSON-safe representation.
    
    Useful for passing complex data structures to JavaScript.
    
    Args:
        value: The value to convert
    
    Returns:
        JSON-safe representation of the value
    """
    import json
    try:
        return json.dumps(value)
    except:
        return '{}' 

@register.filter(name='multiply')
def multiply(value, arg):
    """
    Multiply two values.
    
    Usage in template:
    {{ value|multiply:arg }}
    
    Args:
        value: First value to multiply
        arg: Second value to multiply
    
    Returns:
        Product of the two values
    """
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value

@register.filter(name='divide')
def divide(value, arg):
    """
    Divide two values.
    
    Usage in template:
    {{ value|divide:arg }}
    
    Args:
        value: Numerator
        arg: Denominator
    
    Returns:
        Result of division or 0 if division by zero
    """
    try:
        return float(value) / float(arg) if float(arg) != 0 else 0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0 