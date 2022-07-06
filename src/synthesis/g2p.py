from abc import ABC, abstractmethod
import unicodedata

from phones.convert import Converter

class G2P(ABC):
    @abstractmethod
    def __call__(self, text):
        raise NotImplementedError

class EnglishG2P(G2P):
    def __init__(self):
        from g2p_en import G2p
        self.g2p = G2p()
        self.converter = Converter()
    
    def __call__(self, text):
        phones = self.g2p(text)
        result = []
        for i, phone in enumerate(phones):
            if phone in ["!", "?", ".", ",", ";"]:
                phone_name = "[" + unicodedata.name(phone) + "]"
                if "[" in result[-1]:
                    result[-1] = phone_name
                else:
                    result.append(phone_name)
                continue
            if phone == " ":
                if i == 0 or phones[i - 1] in ["!", "?", ".", ",", ";"]:
                    continue
                else:
                    result.append("[SILENCE]")
                    continue
            phone.replace("ˌ", "")
            phone = phone.replace("0", "").replace("1", "")
            phone = self.converter(phone, "arpabet", lang=None)
            if len(phone) > 0:
                result.append(phone[0])
        return result