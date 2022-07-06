from synthesis.generator import SpeechGenerator
from synthesis.g2p import EnglishG2P

if __name__ == "__main__":
    SpeechGenerator(None, EnglishG2P()).generate_sample_from_text("Hello, this is an important test!")      