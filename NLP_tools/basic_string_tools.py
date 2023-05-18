def clean_string(input_str):
    """
    This function accepts a string and removes all newlines and tabs characters,
    and replaces them with a single space between words.
    """
    # remove all newlines and tabs characters
    clean_str = input_str.replace('\n', ' ').replace('\t', ' ')
    
    # replace multiple spaces with a single space
    clean_str = ' '.join(clean_str.split())
    
    return clean_str
