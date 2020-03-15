import string
import spacy 
import en_core_web_sm
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from normalise import normalise
from nltk.corpus import stopwords
import re

nlp = en_core_web_sm.load()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,variety="BrE",user_abbrevs={},n_jobs=1):
        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs
        return
    
    def transform(self, text,*_):
        text = self._preprocess_text(text)
        return text

    def _preprocess_text(self, text):
        lower_text = self._lower_case(text)
        
        html_remove_text = self._html_tags_removal(lower_text)

        url_remove_text = self._remove_url(html_remove_text) 
        decontracted_text = self._decontracted(url_remove_text)

        normalized_text = self._normalize(decontracted_text)
  
        doc = self._tokenize_text(normalized_text)

        removed_punct = self._remove_punct(doc)
        removed_stop_words = self._remove_stop_words(removed_punct)
        return  re.sub(' +', ' ',' '.join(removed_stop_words))

    def _lower_case(self,text):
        return text.lower()

    def _html_tags_removal(self,text): #function to clean the word of any html-tags and make it lower Cases
        #removing code part from the text
        compiler = re.compile('<code>.*?</code>')
        text = re.sub(compiler, '', text)

        #removing content between 'a' tags 
        compiler = re.compile('<a.*?>.*?</a>')
        text = re.sub(compiler, '', text)

        #removing content between 'img' tags
        compiler = re.compile('<img.*?>.*?</img>')
        text = re.sub(compiler, '', text)

        #removing all tags
        compiler = re.compile('<.*?>')
        text = re.sub(compiler, ' ', text)

        #removing html special symbols
        compiler = re.compile('&.*;')
        text = re.sub(compiler, ' ', text)

        return text

    def _remove_url(self,text):
        #https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
        compiler = re.compile("^https?:\/\/.*[\r\n]*")
        text = re.sub(compiler, '', text)
        return text

    def _decontracted(self,text):
        # specific
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)

        # general
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\n", "", text)
        return text

    def _tokenize_text(self,text):
        tokens = nlp.tokenizer(text)
        return [token.text.lower() for token in tokens if not token.is_space]

    def _normalize(self, text):
        # some issues in normalise package
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text

    def _remove_stop_words(self, doc):
        new_words = []
        for word in doc:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words

    def _remove_punct(self, doc):
        new_words = []
        for word in doc:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words