def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    same_lines = 0
    same_chars = 0
    total_lines = 0
    total_chars = 0

    for line1, line2 in zip(lines1, lines2):
        total_lines += 1
        if line1.strip() == line2.strip():
            same_lines += 1
        for char1, char2 in zip(line1.strip(), line2.strip()):
            total_chars += 1
            if char1 == char2:
                same_chars += 1
    
    return same_lines, same_chars, total_lines, total_chars

if __name__ == '__main__':
    same_lines, same_chars, total_lines, total_chars = compare_files('data/output.txt', 'data/std_output.txt')
    print(f'Number of lines that are the same: {same_lines}/{total_lines}, {same_lines/total_lines*100}%')
    print(f'Number of characters that are the same: {same_chars}/{total_chars}, {same_chars/total_chars*100}%')