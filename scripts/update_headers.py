import os
import glob
import sys
import os
import glob

def update_faa_headers(output_folder):
    # Get all .faa files in the specified directory
    faa_files = glob.glob(os.path.join(output_folder,'*.faa'))
    
    for file_path in faa_files:
        # Get the filename without the directory
        filename = os.path.basename(file_path)
        
        # Read the content of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Update the header lines
        updated_lines = []
        for line in lines:
            if line.startswith('>'):
                updated_lines.append(f'>{filename}_{line[1:]}')
            else:
                updated_lines.append(line)
        
        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)


# Example usage
output_folder=sys.argv[1]
update_faa_headers(output_folder)

