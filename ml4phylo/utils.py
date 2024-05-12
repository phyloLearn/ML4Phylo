def println(txt, param):
    print(txt + "\n", param, "\n")

def split_typing_data(input_file, output_prefix, number_of_files, lines_per_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    header = lines[0]  # Save the header line
    lines = lines[1:]  # Remove the header from the lines

    for i in range(0, number_of_files*lines_per_file, lines_per_file):
        output_file = f"{output_prefix}_{i//lines_per_file + 1}.txt"
        with open(output_file, 'w') as f:
            f.write(header)  # Write the header to each output file
            f.writelines(lines[i:i+lines_per_file])
            f.close()