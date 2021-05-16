import pandas as pd
from nltk.tokenize import RegexpTokenizer
from pathlib import Path


def tokenize_code(text):
    """Gets filtered fucntion tokens"""

    # Remove decorators and function signatures till the def token
    keyword = 'def '
    before_keyword, keyword, after_keyword = text.partition(keyword)
    words = RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(after_keyword)

    # Convert function tokens to lowercase and remove single alphabet variables
    new_words = [word.lower() for word in words if (word.isalpha() and len(word) > 1) or (word.isnumeric())]
    return new_words


def tokenize_docstring(text):
    """Gets filtered docstring tokens which help describe the function"""

    # Remove decorators and other parameter signatures in the docstring
    before_keyword, keyword, after_keyword = text.partition(':')
    before_keyword, keyword, after_keyword = before_keyword.partition('@param')
    before_keyword, keyword, after_keyword = before_keyword.partition('param')
    before_keyword, keyword, after_keyword = before_keyword.partition('@brief')

    if after_keyword:
        words = RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(after_keyword)
    else:
        before_keyword, keyword, after_keyword = before_keyword.partition('@')
        words = RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(before_keyword)

    # Convert all docstrings to lowercase
    new_words = [word.lower() for word in words if word.isalnum()]

    return new_words


def jsonl_list_to_dataframe(files):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat([pd.read_json(f,
                                   orient='records',
                                   compression='gzip',
                                   lines=True)
                      for f in files], sort=False)


def preprocess_data(data_type):
    data_frame = jsonl_list_to_dataframe(
        sorted(Path('resources/data/python/final/jsonl/' + data_type + '/').glob('**/*.gz')))
    data_frame['docstring_filtered'] = data_frame['docstring'].map(tokenize_docstring)
    data_frame['code_filtered'] = data_frame['code'].map(tokenize_code)
    data_frame['docstring_filtered'] = [' '.join(map(str, l)) for l in data_frame['docstring_filtered']]
    data_frame['code_filtered'] = [' '.join(map(str, l)) for l in data_frame['code_filtered']]
    return data_frame


if __name__ == '__main__':
    train_dataframe = preprocess_data('train')
    valid_dataframe = preprocess_data('valid')
    test_dataframe = preprocess_data('test')
    train_dataframe.to_csv('generated_resources/train_data.csv')
    valid_dataframe.to_csv('generated_resources/valid_data.csv')
    test_dataframe.to_csv('generated_resources/test_data.csv')