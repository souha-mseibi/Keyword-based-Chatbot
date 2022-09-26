# Keyword-based-Chatbot
Keyword-based chatbot use customizable keywords and NLP to detect action triggers in the conversation to understand how to respond appropriately to the consumer. 


[Demo.webm](https://user-images.githubusercontent.com/81240719/191527938-ee62fe0c-335e-4143-b9e1-6b006ca54d60.webm)


# Installation 
Clone the repo : 

``` 
https://github.com/souha-mseibi/Keyword-based-Chatbot.git
``` 

Install dependencies

``` 
pip install Flask tensorflow keras NLTK
``` 
Install nltk package

``` 
python 
>>> import nltk
>>> nltk.download('punkt')
``` 
```intents.json``` contain different intents and responses , you can customize it

To train the model and save it , you can run :
```
python train.py
```
Run the following command to test it in the console
```
python chat.py
```


