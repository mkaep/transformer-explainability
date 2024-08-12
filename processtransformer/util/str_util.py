

def str_to_valid_filename(string: str):
    """Replaces all non-alphanumeric letters with underscores"""
    return ''.join([c if c.isalnum() else '_' for c in string])
