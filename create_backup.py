import os
import zipfile
from pathlib import Path

def create_backup():
    # Files and directories to exclude
    exclude = {
        'venv',
        '__pycache__',
        '.git',
        '.env',
        'backup.zip',
        'create_backup.py'
    }
    
    # File extensions to exclude
    exclude_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll'}
    
    # Create zip file
    with zipfile.ZipFile('backup.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through directory
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude]
            
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to('.')
                
                # Skip excluded files and extensions
                if (file_path.name not in exclude and 
                    file_path.suffix not in exclude_extensions):
                    print(f"Adding: {rel_path}")
                    zipf.write(file_path, rel_path)
    
    print("\nBackup created successfully as 'backup.zip'")

if __name__ == "__main__":
    create_backup() 