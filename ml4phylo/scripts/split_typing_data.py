import argparse
from utils import split_typing_data

def main():
    parser = argparse.ArgumentParser(description="Split Typing testdata file")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="/path/ to input file containing the typing data",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="/path/ to output file that'll contain the splitted typing data",
    )
    parser.add_argument(
        "-f",
        "--files",
        required=True,
        type=int,
        help="number of files to split",
    )
    parser.add_argument(
        "-l",
        "--lines",
        required=True,
        type=int,
        help="number of lines splited by each file",
    )

    args = parser.parse_args()

    split_typing_data(args.input, args.output, args.files, args.lines)

if __name__ == "__main__":
    main()
