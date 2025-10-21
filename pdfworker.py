import os, PyPDF2, logging, nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

def get_data_dirs():
    """Return list of subdirectory names inside the 'data' directory."""
    return [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))] if os.path.exists("data") else []

def find_pdf_files(directory_path = None):
    """
    Recursively find all PDF files in a directory and its subdirectories.
    
    Returns:
        list: List of absolute paths to PDF files
    """
    if directory_path == None:
        directory_path = '/home/pe4enushko/Documents/Literature/'
    
    pdf_files = []
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist")
    
    # Check if it's actually a directory
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"'{directory_path}' is not a directory")
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file has a .pdf extension (case-insensitive)
            if file.lower().endswith('.pdf'):
                # Get the absolute path and add to the list
                absolute_path = os.path.join(root, file)
                pdf_files.append(absolute_path)
    
    return pdf_files

def create_directories():
    """
    Create directories with given names in the current working directory.
    """
    directory_names = get_subdirectories()
    
    if directory_names is not str:
        for dir_name in directory_names:
            os.makedirs('data/'+dir_name.strip(), exist_ok=True)


def process_text_to_chunks(text, chunk_size=60000):
    """
    Extract text from PDF, remove stopwords, and split into chunks of ~90k words.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Number of words per chunk (default: 90,000)
    
    Returns:
        list: List of text chunks
    """

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.WordPunctTokenizer().tokenize(text)
    filtered_words = [token for token in tokens if token.lower() not in stop_words and token.isalnum()]
    
    # Split into chunks
    chunks = []
    for i in range(0, len(filtered_words), chunk_size):
        chunk = ' '.join(filtered_words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def process_pdf_to_chunks(pdf_path, chunk_size=60000):
    """
    Extract text from PDF, remove stopwords, and split into chunks of ~90k words.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Number of words per chunk (default: 90,000)
    
    Returns:
        list: List of text chunks
    """
    # Extract text from PDF
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens = nltk.WordPunctTokenizer().tokenize(text)

    filtered_words = [token for token in tokens if (token.lower() not in stop_words) and token.isalnum()]

    # Split into chunks
    chunks = []
    for i in range(0, len(filtered_words), chunk_size):
        chunk = ' '.join(filtered_words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def get_files_to_process():
    dir1 = '/home/pe4enushko/Documents/Literature/'
    dir2 = './data/'
    """Return files from dir1 that don't have same-name files in dir2 (including subdirectories)."""
    # Get all relative file paths from both directories
    files1 = {os.path.join(root, f) 
                for root, _, files in os.walk(dir1) 
                for f in files if f.lower().endswith('.pdf')}

    files2 = {os.path.basename(f) 
                for root, _, files in os.walk(dir2) 
                for f in files if f.lower().endswith('.txt')} if os.path.exists(dir2) else set()

    # Return absolute paths of unique PDF files
    return [f for f in files1 if os.path.basename(f) not in files2]


def get_subdirectories():
    directory_path = '/home/pe4enushko/Documents/Literature/'
    """Return list of subdirectory names in the given directory."""
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return []
    
    return [name for name in os.listdir(directory_path) 
            if os.path.isdir(os.path.join(directory_path, name))]