import os
import sys

def process_fna_files(input_folder, output_file):
    with open(output_file, 'w') as combined_file:
        for filename in os.listdir(input_folder):
            if filename.endswith(".fna"):
                file_path = os.path.join(input_folder, filename)
                base_filename = os.path.splitext(filename)[0]
                new_header = f">{base_filename}"
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    lines[0] = new_header + '\n'
                    combined_file.writelines(lines)
                    combined_file.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_file>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory.")
        sys.exit(1)
    
    process_fna_files(input_folder, output_file)

