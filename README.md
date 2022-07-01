#Natural Language Processing 

Spam Filtering Project

## About the project
Spam is any kind of unwanted, unsolicited digital communication that gets sent out in bulk. Often
spam is sent via email, but it can also be distributed via text messages, phone calls, or social media.
Natural Language Processing has been very promising in detecting spam messages in recent years.
Nowadays all bid email service providers use NLP to automatically detect spam emails.


### Features

Currently, the Scripts contained in this folder allow to:

- Preprocess the email texts and get insights of the data.
- Train a neural network to detect spam emails based on different vectorization models
- Train a model by vectorizing words and finding the most similar words of 20 random chosen ones.


Note:
The instructions are written for Linux, specifically Ubuntu 20.04.4 LTS 

### Development
The scripts are currently under development. The development is done using *Python 3.8.10*. The rest of the requirements can be found in the [requirements file (requirements.txt)](requirements.txt).

## Set up

### Starting 🚀

_
These instructions will allow you to get a copy of the project running on your local machine for development and testing purposes_

Look at **Deployment** to know how to deploy this project.


### Requirements 📋

_**Git**_

```
sudo apt update
sudo apt install git
```

_**Python 3.8.10**_
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3
```

_**Install Pip**_
```
sudo apt install python3-pip
```


**virtualenv**

_install python3-venv with_
```
sudo apt install python3.8-venv
```

_create a directory to save virtual environments and enter it_
```
mkdir virtualenvs
cd virtualenvs
```

_Set the environment_
```
python -m venv env
```

_To activate use_
```
source env/bin/activate
```

_To deactivate it use_
```
source env/bin/deactivate
```

### Installation 🔧

_Follow these steps once done with the **Requirements**:_

_**NOTE: Keep your virtual environment activated for the installation.**



_clone the github repository_

```
git clone https://git.tu-berlin.de/juliop1996/NLP_project1_4.git
```

_enter repository_

```
cd NLP_project1_4/
```




_A requirements file was generated that will allow the automatic installation of the modules with_

```
sudo pip install -r requirements.txt
```

## Run and coding ⌨️

### Structure of the code

_The features are divided into three different scripts which follow this order by logic: 1st  **insights_data.py**, 2nd **bayes.py**, 3rd **feed_forward_neural_network.py**, and 4rd **pmi.py**._

_The first script preprocesses the data and provides some insights about the emails while the second and third scripts provide models to predict if an email is spam or not, by using different vectorization models._
_The last script convert words into vectors and trains a model in order to later choose 20 random words of the data and find the most similar words for each one of them._
_A filesystem is created automatically by the scripts in order to organize the results of the scripts and have a log of what is been done._

#### insights_of_data
_In this folder lay the box-plot made to compare the lenghts of spam emails and non spam emails along with a text file which provide some interesting meassurements such as the most common words and the number of appearances in each type of email._


#### Neural_Netowork_results
_In this folder lay all the results of the neural network model based on accuracy and f1-score for each of the vectorization model used._
_There is a png file for each vectorization model that shows the progress of the training done and a text file with the name of the model which shows the f1 score and accuracy._


### Important commands

_to run the scripts_

They take around 5 minutes each.

```
python insights_data.py
```
```
python bayes.py
```

```
 python feed_forward_neural_network.py <vectorization_type>
```

```
python pmi.py
```

### Available vectorization types

Count_Vectorizer = "count"
TF-IDF = "TF-IDF"
2-gram = "2-gram"
3-gram = "3-gram"
4-gram = "4-gram"

## Authors

Group 22

-Anne Laure Olga Tettoni\
-Julio Cesar Perez Duran


