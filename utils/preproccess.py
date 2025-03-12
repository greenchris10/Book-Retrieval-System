import os
import json
import pandas as pd

def convert_book_to_json( file_path, author="Frank Herbert"):
        """
        Converts a text file containing a book into a structured JSON format.

        Args:
            file_path (str): Path to the book text file.
            author (str): Name of the book's author.

        Returns:
            dict: A structured JSON dictionary.
        """

        book_title = ' '.join(os.path.splitext(os.path.basename(file_path))[0].split('_'))  # Extract filename as book title

        with open(file_path, "r", errors="ignore") as f:
            text = f.read()

        return {
            "author": author,
            "book_title": book_title,
            "text": text
        }

def convert_books(input_dir="./data/books_txt/Frank_Herbert", output_dir="./data/books_txt/Frank_Herbert/books_json"):
    """
    Converts all text books in the input directory to structured JSON files.

    Args:
        input_dir (str): Directory containing raw book text files.
        output_dir (str): Directory to save the converted JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            
            book_json = convert_book_to_json(file_path)

            output_file = os.path.join(output_dir, f"{ '_'.join(book_json['book_title'].split(' ')) }.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(book_json, f, indent=4)

            print(f"Converted {filename} → {output_file}")
    
if __name__ == "__main__":
     convert_books()