Following is the code repository structure:

Dataset/ -> Dir which hosts datasets for different tasks
Prompts/ -> Dir which hosts prompt templates for different dataset tasks
LLM_init.py -> Python module which hosts LLM based classifiers for different datasets
LLM_eval.py -> Python module to evaluate LLM based classifiers on different datasets (calculates F1, Recall, Precision)
Project_Documentation.txt -> Project overview and Documentation 

Steps to run Demo:

# move to project root directory
cd <../LLM_Classifiers>

# Install necessary Python package dependencies
pip install -r requirements.txt

# Run end to end eval for different classifiers
python 'LLM_eval.py'

Steps to evaluate individual classifiers:

# import required classifier eval objects
from LLM_eval import IMDBBinaryClassifier_Evaluator, NewsGroupMultiClassClassifier_Evaluator, EmotionsMultiClassClassifier_Evaluator

# evaluate classifiers individually (provide percentOfDataset param to eval subset of dataset)
eval_obj = IMDBBinaryClassifier_Evaluator()
eval_obj.evaluate(percentOfDataset=0.125)

eval_obj = NewsGroupMultiClassClassifier_Evaluator()
eval_obj.evaluate()

eval_obj = EmotionsMultiClassClassifier_Evaluator()
eval_obj.evaluate()