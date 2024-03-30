import collections
from wordcloud import STOPWORDS



def get_most_common_words(text, num_words=10):
    """
    Function to get the most common words from a text.

    Parameters:
        text (str): The input text.
        num_words (int): Number of most common words to return. Default is 10.

    Returns:
        list: A list containing the most common words.
    """
    stopwords = set(STOPWORDS)
    filtered_words = [word for word in text.split() if word.lower() not in stopwords]
    counted_words = collections.Counter(filtered_words)
    most_common_words = counted_words.most_common(num_words)
    return [word for word, _ in most_common_words]