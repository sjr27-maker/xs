from pathlib import Path

# List of files to create based on your requirements
files_to_create = [
    "intake/__init__.py", "cognition/__init__.py", "pedagogy/__init__.py",
    "style/__init__.py", "prompt/__init__.py", "memory/__init__.py",
    "voice/__init__.py", "output/__init__.py", "onboarding/__init__.py",
    "feedback/__init__.py", "data/.gitkeep", "sessions/.gitkeep",
    "tests/__init__.py"
]

def initialize_structure():
    for file_path in files_to_create:
        path = Path(file_path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the file (touch)
        path.touch(exist_ok=True)
        print(f"Created/Verified: {file_path}")

if __name__ == "__main__":
    initialize_structure()
    print("\nFile structure initialized successfully.")