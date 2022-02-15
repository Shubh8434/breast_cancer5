# Breast_Cancer_Prediction
# Website Link: https://breast-cancer-123.herokuapp.com/
# This is the (Dataset)[https://drive.google.com/file/d/1cwxAcAnf8UqRdrfE6bcNIVvz_wJsqaWQ/view?usp=sharing]
## Breast Cancer
###Breast cancer is a malignant tumor that grows in or around the breast tissue, mainly in the milk ducts and glands. A tumor usually starts as a lump or calcium deposit that develops as a result of abnormal cell growth.
It’s important to understand that most breast lumps are benign and not cancer (malignant).

![image](https://user-images.githubusercontent.com/81500352/153710185-52a774be-5232-4f59-a0e0-84d62e4c0766.png)         
Malingnat-->0 && Benign-->1 for computation

## About Project:-
### Can We Predict that a person has whether a benign or malingnat tumor?
Answer is Yess!!! 
This is a project on predicting whether a person has a benign or malingnat tumor in breast_cancer_sklearn_dataset using Machine learning Algorithm.This project is tested over lot of ml models. Out of these models Logistic Regression performed very well giving an 91% Score. . To increase Accuracy of algotithm, I have done hyperparameter tuning using GridSearchCV modeule and get score of 94%.

# Testing Values:-
## Case:- Benign
![benign](https://user-images.githubusercontent.com/81500352/153711489-6bc9eb99-6a71-42fb-83cd-3b7c0b996a8d.jpg)

## Case:- Malignant
![malignant](https://user-images.githubusercontent.com/81500352/153711727-c24406b5-c99d-44fd-b278-a3d1a2976e06.jpg)

# Tech Stack:-
- UI/UX: Streamlit 
- IDE: Jupyter notebook, VsCode
- Deployment: Heroku

# How to run this app:-
- First create a virtual environment by using this command:
- conda create -n myenv python=3.6
- Activate the environment using the below command:
- conda activate myenv
- Then install all the packages by using the following command
- pip install -r requirements.txt
- Now for the final step. Run the app
- streamlit run app.py

# Dashboard:-
![Image 2](https://user-images.githubusercontent.com/81500352/154002784-7e849ed6-b8d2-4397-ae0e-e654883477bc.jpg)

