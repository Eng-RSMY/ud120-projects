#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split(b"X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        # This is the best way to remove punctuation from a string since it's performing raw string operations in C with
        # a lookup table.
        text_string = content[1].translate(bytes.maketrans(b"", b""), bytearray(string.punctuation, encoding='UTF=8'))

        ### project part 2: comment out the line below
        # words = text_string.decode('UTF-8')

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        from nltk.stem.snowball import SnowballStemmer

        snstmr = SnowballStemmer('english')
        text_string = text_string.decode('UTF-8')
        text_string = text_string.split()

        text_string = [snstmr.stem(item) for item in text_string]
        words = " ".join(text_string)

        # words = set(snstmr.stem(item) for item in text_string)
    return words

    

def main():
    with open(r"../text_learning/test_email.txt", "rb") as fin:
        text = parseOutText(fin)

    print(text)


if __name__ == '__main__':
    main()

