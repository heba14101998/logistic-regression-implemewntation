## BACKGROUND 
</center>

Exaplination in depth the logistic regression algorithm

### Description of Binary Classification

A large part of data analysis boils down to a simple question: is something "A" or "B?" Is it "positive" or "negative?" Is this person a "potential customer" or "not a potential customer?" Machine learning accommodates such questions through logistic equations, and specifically through what is known as the sigmoid function[1].

The sigmoid function produces an S-shaped curve that can convert any number and map it into a numerical value between 0 and 1, but it does so without ever reaching those exact limits[1].

Note:
> Logistic regression is typically used for binary classification to predict two discrete classes [1].

### Hypothesis Representation
A key difference from linear regression is that the output value being modeled is a binary value (0 or 1) rather than a numeric value. if the value returned by the model for input x is closer to 0, then we assign a negative label to x; otherwise, the example is labeled as positive. One function that has such a property is the standard logistic function (also known as the sigmoid function)[2].

### Objective Function 
Now, how do we find optimal w and b? we **maximize the likelihood** of our training set according to the model. The optimization criterion in logistic regression is called maximum likelihood. Instead of minimizing the average loss, like in linear regression, we now maximize the likelihood of the training data according to our model. We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.
[2].
Note
> Probability corresponds to finding the chance of something given a sample distribution of the data, while on the other hand, Likelihood refers to finding the best distribution of the data given a particular value of some feature or some situation in the data.

### Tools

* Python 3.
* NumPy Library (1.18.5).
* Matplotlib Library (3.3.2).
* Visual studio.
* Spyder IDE.

## RRFREMCES
[1] [Machine Learning For Absolute Beginners](https://www.amazon.com/Machine-Learning-Absolute-Beginners-Introduction-ebook/dp/B07335JNW1)

[2] [The Hundred-Page Machine Learning Book](https://www.amazon.com/Hundred-Page-Machine-Learning-Book/dp/199957950X)

[3] All equations from [androw course](https://www.coursera.org/learn/machine-learning)

[4] [Multi class classification medium artical](https://wadhwatanya1234.medium.com/multi-class-classification-one-vs-all-one-vs-one-993dd23ae7ca)

[5] You can find the equations explination from [here]()
