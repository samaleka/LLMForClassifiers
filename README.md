Following is the code repository structure:

1. Dataset/ -> Dir which hosts datasets for different tasks
2. Prompts/ -> Dir which hosts prompt templates for different dataset tasks
3. LLM_init.py -> Python module which hosts LLM based classifiers for different datasets
4. LLM_eval.py -> Python module to evaluate LLM based classifiers on different datasets (calculates F1, Recall, Precision)
5. Project_Documentation.txt -> Project overview and Documentation 

## Steps to run Demo:

### Move to project root directory
```bash
cd <../LLM_Classifiers>
```
### Install necessary Python package dependencies
```bash
pip install -r requirements.txt
```
### Run end to end eval for different classifiers
```bash
python 'LLM_eval.py'
```

## Steps to evaluate individual classifiers:

### Import required classifier eval objects
```bash
from LLM_eval import IMDBBinaryClassifier_Evaluator, NewsGroupMultiClassClassifier_Evaluator, EmotionsMultiClassClassifier_Evaluator
```
### Evaluate classifiers individually (provide percentOfDataset param to eval subset of dataset)
```bash
eval_obj = IMDBBinaryClassifier_Evaluator()
eval_obj.evaluate(percentOfDataset=0.125)

eval_obj = NewsGroupMultiClassClassifier_Evaluator()
eval_obj.evaluate()

eval_obj = EmotionsMultiClassClassifier_Evaluator()
eval_obj.evaluate()
```
