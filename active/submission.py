"""
File: submission.py
Project: Yizkor Pipeline
Author: Elijah Greenberg
Description: File submission handling for Yizkor Books.
"""


"""
Desired Methods of Upload:
 - From Computer
 - From Google Drive
 - Select pages (multiple files)
 - Full/Partial Book (one file)

I would like my own copies of these submissions stored maybe
on my computer or google drive or a persistent database. 
"""

# Prompt user for file submission method
import os


ACCEPTED_FILE_FORMATS = ['.pdf']  # consider adding more formats



# ---------------------------------------------------------------------------
# Submission prompts
# ---------------------------------------------------------------------------
def prompt_submission_method():
    print("Select submission method:")
    print("1. Single File From Computer")
    print("2. Multiple Files From Computer")
    print("3. Single File From Google Drive")
    print("4. Multiple Files From Google Drive")

    choice = input("Enter your choice: ")
    return choice

def _validate_submission_choice(choice):
    return choice in ['1', '2', '3', '4']

def _prompt_for_path(choice):
    if choice == '1':
        print("Please enter the path to the file:")
        return input("File path: ")
    elif choice == '2':
        print("Please enter the path to the folder containing the files:")
        return input("Folder path: ")
    elif choice == '3':
        print("Please enter the path to the file in Google Drive:")
        return input("File path: ")
    elif choice == '4':
        print("Please enter the path to the folder in Google Drive:")
        return input("Folder path: ")

def get_submission_path():
    choice = prompt_submission_method()
    while not _validate_submission_choice(choice):
        print("Invalid choice. Please select a valid submission method.")
        choice = prompt_submission_method()

    return _prompt_for_path(choice)


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------
def verify_file_path(path):
    if not os.path.isfile(path):
        raise ValueError("Invalid file path")

    _, ext = os.path.splitext(path)
    if ext.lower() not in ACCEPTED_FILE_FORMATS:
        raise ValueError("File format not accepted")

    return path

def verify_folder_path(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("Invalid folder path")
    return folder_path

def normalize_path(path):
    if not isinstance(path, str):
        return path

    return path.strip().strip('"').strip("'")

def verify_submission_path(path):
    normalized_path = normalize_path(path)

    # Determine if the path is a file or folder
    # ASSUME NO GOOGLE DRIVE (for now)
    if os.path.isfile(normalized_path):
        return verify_file_path(normalized_path)
    elif os.path.isdir(normalized_path):
        return verify_folder_path(normalized_path)
    else:
        raise ValueError("Invalid path")


# ---------------------------------------------------------------------------
# Submission flow
# ---------------------------------------------------------------------------
def handle_submission_request():
    while True:
        path = get_submission_path()

        try:
            validated_path = verify_submission_path(path)
            print("Submission successful. Path validated:", validated_path)
            return validated_path
        except ValueError as error:
            print(f"Submission error: {error}")
            print("Please try a different path.")

    

    


def main():
    handle_submission_request()

if __name__ == "__main__":
    main()