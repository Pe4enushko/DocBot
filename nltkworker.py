import nltk
from nltk.tokenize import sent_tokenize
import re

# Download required NLTK data (run once)
# nltk.download('punkt')

def shorten_technical_text(text):
    """
    Keep only sentences about fault codes and plain instructions.
    Remove everything else.
    """
    # Define patterns for fault codes and instructions
    fault_code_patterns = [
        r'\b[A-Z]{1,4}[0-9]{1,4}[A-Z]?\b',  # ABC123, AB12, A1B2, etc.
        r'\b[Ff]ault\s+[Cc]ode\s*[A-Z0-9]+\b',
        r'\b[Dd]iagnostic\s+[Tt]rouble\s+[Cc]ode\b',
        r'\bDTC\s*[A-Z0-9]+\b',
        r'\b[Ee]rror\s+[Cc]ode\s*[A-Z0-9]+\b',
        r'\b[Pp]roblem\s+[Cc]ode\s*[A-Z0-9]+\b'
    ]
    
    instruction_patterns = [
        r'\b[Ss]tep\s+\d+',
        r'\b[Ff]irst\b.*\bthen\b',
        r'\b[Mm]ust\b',
        r'\b[Ss]hould\b',
        r'\b[Cc]heck\b',
        r'\b[Vv]erify\b',
        r'\b[Ii]nspect\b',
        r'\b[Rr]eplace\b',
        r'\b[Rr]emove\b',
        r'\b[Ii]nstall\b',
        r'\b[Cc]onnect\b',
        r'\b[Dd]isconnect\b',
        r'\b[Tt]est\b',
        r'\b[Mm]easure\b',
    ]
    
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    relevant_sentences = []
    
    for sentence in sentences:
        # Check if sentence contains fault codes
        has_fault_code = any(re.search(pattern, sentence) for pattern in fault_code_patterns)
        
        # Check if sentence contains instructions
        has_instruction = any(re.search(pattern, sentence) for pattern in instruction_patterns)
        
        # Keep sentence if it contains fault codes or instructions
        if has_fault_code or has_instruction:
            relevant_sentences.append(sentence)
    
    return ' '.join(relevant_sentences)