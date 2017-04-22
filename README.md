# Machine learning interview

This repository covers how to prepare for machine learning interviews, mainly
in the format of questions & answers. Asides from machine learning knowledge,
other crucial aspects include:

* [Explain your resume](#explain-your-resume)
* [SQL](#sql)

Go directly to [machine learning](#machine-learning)


## Explain your resume

Your resume should specify interesting ML projects you got involved in the past,
and **quantitatively** show your contribution. Consider the following comparison:

> Trained a machine learning system

vs.

> Trained a deep vision system (SqueezeNet) that has 1/30 model size, 1/3 training
> time, 1/5 inference time, and 2x faster convergence compared with traditional
> ConvNet (e.g., ResNet)

We all can tell which one is gonna catch interviewer's eyeballs and better show
case your ability.

In the interview, be sure to explain what you've done well. Spend some time going
over your resume before the interview.


## SQL

Although you don't have to be a SQL expert for most machine learning positions,
the interviews might ask you some SQL related questions so it helps to refresh
your memory beforehand. Some good SQL resources are:

* [W3schools SQL](https://www.w3schools.com/sql/)
* [SQLZOO](http://sqlzoo.net/)


## Machine learning

* [How decision tree works](#how-decision-tree-works)
* [How random forest works](#how-random-forest-works)



### How decision tree works

* Non-parametric, supervised learning algorithms
* Given the training data, a decision tree algorithm divides the feature space into
regions. For inference, we first see which
region does the test data point fall in, and take the mean label values (regression)
or the majority label value (classification).
* **Construction**: top-down, chooses a variable to split the data such that the 
target variables within each region are as homogeneous as possible. Two common
metrics: gini impurity or information gain, won't matter much in practice.
* Advantage: simply to understand & interpret, mirrors human decision making
* Disadvantage: 
    - can overfit easily (and generalize poorly)if we don't limit the depth of the tree
    - can be non-robust: A small change in the training data can lead to a totally different tree

![decision-tree](https://docs.microsoft.com/en-us/azure/machine-learning/media/machine-learning-algorithm-choice/image5.png)

[back to top](#machine-learning)


### How random forest works

* Random forest is an ensemble of decision tree algorithms.