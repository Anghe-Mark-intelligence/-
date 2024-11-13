import os

def find_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

# Directory to search
log_directory = r'C:\Users\Administrator\Desktop\heangcomputervision\第七次实验\log'

# Find all files
all_files = find_all_files(log_directory)

# Print out all files found
if not all_files:
    print("No files found in the specified directory.")
else:
    print("Files found:")
    for file in all_files:
        print(file)