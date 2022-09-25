from abc import ABC, abstractmethod
import unicodedata

from g2p_en import G2p
from phones.convert import Converter


class G2P(ABC):
    def __init__(self, lexicon_path):
        self.lexicon_path = lexicon_path
        self.lexicon = self.load_lexicon()

    @abstractmethod
    def __call__(self, text):
        raise NotImplementedError

    @abstractmethod
    def load_lexicon(self):
        raise NotImplementedError


class EnglishG2P(G2P):
    def __init__(self, lexicon_path=None):
        super().__init__(lexicon_path)
        self.g2p = G2p()
        self.converter = Converter()

    def __call__(self, text):
        text = unicodedata.normalize("NFKD", text)
        text = text.lower()
        words = text.split(" ")
        phones = []
        for word in words:
            if word[-1] in [".", ",", "!", "?"]:
                punctuation = word[-1]
                word = word[:-1]
            else:
                punctuation = ""
            if word in self.lexicon:
                add_phones = self.lexicon[word]
            else:
                add_phones = self.g2p(word)
            for phone in add_phones:
                phone.replace("ˌ", "")
                phone = phone.replace("0", "").replace("1", "")
                phone = self.converter(phone, "arpabet", lang=None)
                phones += phone
            if punctuation != "":
                phones.append("[" + unicodedata.name(punctuation) + "]")
            else:
                phones.append("[SILENCE]")
        return phones

    def load_lexicon(self):
        lexicon = {}
        if self.lexicon_path is not None:
            with open(self.lexicon_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    word, phones = line.split("\t")
                    phones = phones.split(" ")
                    lexicon[word.lower()] = phones
        return lexicon
