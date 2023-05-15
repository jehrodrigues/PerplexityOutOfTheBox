def convert_labels(labels):
    """Convert labels into integer format."""
    # to do: automatize
    return {'female': 0,
            'male': 1,
            'neutral': 2
            }
    """
    return {'body_type': 0,
            'characteristics': 1,
            'ability': 2,
            'age': 3,
            'cultural': 4,
            'gender_and_sex': 5,
            'nationality': 6,
            'nonce': 7,
            'political_ideologies': 8,
            'race_ethnicity': 9,
            'religion': 10,
            'sexual_orientation': 11,
            'socioeconomic_class': 12
            }
    """


class TextPreprocessing(object):
    """
    Handles text pre-processing
    """

    def __init__(self, sentence: str):
        self._sentence = sentence

    def remove_html(self) -> str:
        """Take a sentence and remove the html tags."""
        clean_sentence = self._sentence.replace("[\"","")
        clean_sentence = clean_sentence.replace("\"]", "")
        clean_sentence = clean_sentence.replace("[\'", "")
        clean_sentence = clean_sentence.replace("\']", "")
        return clean_sentence