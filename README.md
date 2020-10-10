# isml
My study of ISML (Introduction to Statistical Machine Learning)  
Wuli Zuo  
a1785343  

## a1_svm:

### Introduction

This practice aims to implement soft margin svm classifiers with cvxopt  
by solving the primal problem and the dual problem,  
and compare the results with a third-party library scikit-learn.

### How to run  

* Mode 0: To completely run the code with all the experiment  
Console command: python main.py or python3 main.py  
    >note: This mode would take hours to complete.  
* Mode 1: To skip cross validation, use primal C to train the model, predict and compare  
Console command: python main.py 1 or python3 main.py 1  
    >note: Recommended! You can see main steps, and this mode would take about 13 minutes.    
* Mode 2: To predict and compare using saved models 
Console command: python main.py 2 or python3 main.py 2  
    >note: This is the fast way to see the results.  
> Please ensure that the data files 'train.csv' and 'test.csv' are at directory: data/  
> The output files are at: output/  
> The results of w and b may vary a little because of shuffle, 
> but have no significant effect on performance.  

### Documents list:

#### Report

* Intro_to_Statis_Machine_learning-Assignment_1_report-Wuli_Zuo-a1785343.pdf:  Report to my A1  
including:  
    1. Understanding of SVM  
    2. A brief explanation of how I implemented my code,  
    3. Experiments and result analysis;  
    4. Appendix: The cvxopt forms and construction of coefficient matrix
    5. Reference

#### Code

* main.py: The program entry, control the experiment process.  
* load.py: The python script to load data from data files.  
* primal.py: Implementation of the primal SVM by solving the primal optimisation problem.  
* dual.py: Implementation of the dual SVM by solving the dual optimisation problem.  
* predict.py: The python script to predict test data with a given SVM model.  
* sk.py: The use of third-party library scikit-learn, to compare results with my SVMs.  
* validate.py: Implementation of k-fold cross validation to decide the optimal C.  
* draw.py: The python script to generate figures to show the accuracy of different C in cross validation.  
* compare.py: The python script to compare w and b of two different SVMs.  

#### Output

* out.txt: The complete output of the process, including:  
    1. The process and result of cross validation;      
    2. Values of w and b;  
    3. Accuracy of SVMs on training and test data; 
    4. Comparison of w and b between primal and dual SVM;
    5. Comparison of w and b between primal and sklearn SVM;
* svm_model_primal: Saved primal SVM, can be used to predict and compare the solution directly
* svm_model_dual: Saved dual SVM, can be used to predict and compare the solution directly

## a2_adaboost:

### Introduction

This practice aims to implement AdaBoost  
using decision stump as weak learners,  
and compare the results with an third-party scikit-learn AdaBoost and SVM.

### How to run  
 
Console command: python main.py or python3 main.py  
>note: This would take 3 minutes to complete.    
> Please ensure that the data files 'wdbc_data.csv' are at directory: data/  
> The output files are at: output/  
 

### Documents list:

#### Report

* Intro_to_Statis_Machine_learning-Assignment_2_report-Wuli_Zuo-a1785343.pdf:  Report to my A2  
including:  
    1. Description of the AdaBoost method  
    2. Understanding of AdaBoost  
    3. Introduction of how to run my code  
    4. Experiments and result analysis  
    5. Reference

#### Code

* main.py: The program entry, control the experiment process.  
* load.py: The python script to load data from the data file.  
* stump.py: Implementation of decision stump.  
* adaboost.py: Implementation of AdaBoost algorithm.  
* predict.py: The python script to predict test data with my AdaBoost model.  
* sk.py: The use of third-party library scikit-learn, to compare results with my model.  
* plot.py: The python script to generate figures to show the relationship between errors and iterations.  

#### Output

* out.txt: The complete output of the process, including:  
    1. The weak learner of each iteration;
    2. My AdaBoost model;
    3. The predicting result of my AdaBoost with different number of iterations; 
    4. The predicting result of scikit-learn AdaBoost with default parameters;
    5. The predicting result of scikit-learn AdaBoost with best parameters;
    6. The analysis of scikit-learn AdaBoost parameters;
    7. The analysis of depth of weak learners;
    8. The predicting result of scikit-learn SVM.
