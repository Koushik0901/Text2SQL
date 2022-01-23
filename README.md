# ***Text2SQL***
***A Transformer model trained on WikiSQL dataset that accepts natural language as input and returns SQL Query as output.***

## **Demo**
### Try it yourself [here](https://share.streamlit.io/koushik0901/text2sql/ui.py)
<p align="center"> <img src="./utils/examples.png" width="1200" height="300"  /> </p>

## **Project Organization**
------------

    ├── LICENSE
    │
    ├── README.md               <- Documentation to get more information about the project.
    │
    ├── saved_models            <- Trained and serialized models.
    │
    ├── requirements.txt        <- The requirements file for reproducing the environment.
    │
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module.
    |   |
    |   ├── train_tokenizer.py  <- Script to train a sentencepiece tokenizer on the dataset.
    |   |
    |   ├── dataset.py          <- Script to load and preprocess the dataset.
    |   |
    |   ├── model.py            <- Script that defines the transformer model.
    |   |
    |   ├── config.py           <- Contains all the basic parameters for training.
    │   │
    |   └── train.py            <- Script to train the model. 
    │
    ├── utils
    |   ├── examples.csv        <- CSV files with few example predictions.
    │   │
    |   └── examples.png        <- An image with few example predictions.
    |
    ├── engine.py               <- Script to perform inference on the trained model.
    |
    └── ui.py                   <- Script to build the streamlit web application.

## **Running on native machine**
### *dependencies*
* python3
### *pip packages*
```bash
pip install -r requirements.txt
```
## **Steps to train your own model**
 ### *Scripts*
 `src/train.py` - is used to train the model \
 `engine.py` - is used to perform inference \
 `ui.py` - is used to build the streamlit web application
