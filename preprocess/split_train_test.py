import os
import random
import argparse

def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def split_files(file_paths, train_ratio=0.9):
    random.shuffle(file_paths)
    split_index = int(len(file_paths) * train_ratio)
    train_files = file_paths[:split_index]
    test_files = file_paths[split_index:]
    return train_files, test_files

def write_to_file(file_list, file_name):
    with open(file_name, 'w') as f:
        for file_path in file_list:
            f.write(f"{file_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split files in a directory into train and test sets.")
    parser.add_argument("--directory", type=str, default="/home/yxlin/backup/cv-corpus-18.0-2024-06-14/zh-TW/clips", help="The directory containing files to split")
    parser.add_argument("--train_file", type=str, default="/home/yxlin/backup/cv-corpus-18.0-2024-06-14/zh-TW/my_train.txt", help="Output file for training set paths")
    parser.add_argument("--test_file", type=str, default="/home/yxlin/backup/cv-corpus-18.0-2024-06-14/zh-TW/my_test.txt", help="Output file for test set paths")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of files to use for training set")

    args = parser.parse_args()

    all_files = get_all_files(args.directory)
    train_files, test_files = split_files(all_files, args.train_ratio)

    write_to_file(train_files, args.train_file)
    write_to_file(test_files, args.test_file)

    print(f"Train files written to: {args.train_file}")
    print(f"Test files written to: {args.test_file}")
