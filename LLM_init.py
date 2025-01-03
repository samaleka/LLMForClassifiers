# import libs
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Base class for different classifiers to inherit from
class Classifier:
    def __init__(self, promptFile):

        model_name = "google/flan-t5-large"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, trust_remote_code=True)

        file = open(promptFile, "r")
        self.promptTemplate = file.read()
        file.close()
    
    def classify_internal(self, input):

        prompt = self.promptTemplate.format(input=input)
        tokens = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)

        # Ask LLM to classify
        outputs = self.model.generate(**tokens, max_length=100)
        classification = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return classification

# Binary Classifier | IMDB Movie Ratings Sentiment Classifier
class IMDB_BinaryClassifier(Classifier):
    def __init__(self):
        promptFile = "./Prompts/IMDB_Sentiment_Prompt.txt"
        Classifier.__init__(self, promptFile)

    def classify(self, input):
        result = self.classify_internal(input)

        try:
            return int(result)
        except:
            print(f"failed to parse {result}")
            raise Exception(f"failed to parse {result}")


# Multi Class Classifier | 20 News Group Classifier
class NewsGroup_MultiClassClassifier(Classifier):
    def __init__(self):
        promptFile = "./Prompts/20NewsGroup_Prompt.txt"
        Classifier.__init__(self, promptFile)
        self.classEnum = {'alt.atheism': 0, 'comp.graphics': 1, 'comp.os.ms-windows.misc': 2,
                          'comp.sys.ibm.pc.hardware': 3, 'comp.sys.mac.hardware': 4, 'comp.windows.x': 5,
                          'misc.forsale': 6, 'rec.autos': 7, 'rec.motorcycles': 8, 'rec.sport.baseball': 9,
                          'rec.sport.hockey': 10, 'sci.crypt': 11, 'sci.electronics': 12, 'sci.med': 13,
                          'sci.space': 14, 'soc.religion.christian': 15, 'talk.politics.guns': 16,
                          'talk.politics.mideast': 17, 'talk.politics.misc': 18, 'talk.religion.misc': 19}

    def classify(self, input):
        result = self.classify_internal(input)

        try:
            # return enum for predicted class
            return self.classEnum[result]
        except:
            print(f"failed to parse {result}")
            raise Exception(f"failed to parse {result}")


# Multi Class Classifier | Emotions Classifier
class Emotions_MultiClassClassifier(Classifier):
    def __init__(self):
        promptFile = "./Prompts/Emotions_Prompt.txt"
        Classifier.__init__(self, promptFile)
        self.classTypes = ('anger', 'fear', 'joy', 'love', 'sadness', 'surprise')

    def classify(self, input):
        result = self.classify_internal(input)

        if result in self.classTypes:
            return result
        else:
            print(f"failed to parse {result}")
            raise Exception(f"failed to parse {result}")