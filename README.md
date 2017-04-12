## Machine Learning Toolkit

This repository contains python examples and notes around the development and implementations of various machine learning algorithms & data preparation techniques. It is a compilation of various book notes, bootcamps & research put together for myself - and others, to make references to quick implementations easily. Please feel free to contribute to the repository! 

-- Jayson

![](img/machinelearning201.png)

---

### Python Machine Learning 

_Training Machine Learning Algorithms for Classification_
- [Perceptron](src/perceptron.ipynb) - Step by step implementation in Python 
- [Adaline](src/adaline.ipynb) - Working with the basics of optimizations, stochastic/batch gradient descent

_Training Artificial Neural Networks for Image Recognition_
- [Multi-Layer Perceptron MLP - Part 1](src/MLP_part1.ipynb) - Implementation of MLP algorithm & analysis  


_Training Machine Learning Algorithms for Regression_
- [Overview Regression](src/regression.ipynb) - Examples on linear regression, polynomial regression, random forest regressor  

_Machine Learning Classifiers Using Scikit-learn_ 
- [Perceptron with sklearn](src/perceptron-sklearn.ipynb) - Demonstrating the iris dataset with perceptron algorithm in sklearn ml kit
- [Logistic Regression with sklearn](src/logisticregression-sklearn.ipynb) - Working with logistic regression in sklearn & visualizing regularization
- [Support Vector Machines (SVM) with sklearn](src/supportvectormachines.ipynb) - Using the linear and kernal SVM,
- [Decision tree learning, Random Forest](src/decisiontree.ipynb) - Impurity measures, such as Gini, Entropy & Classification Error
- [K-nearest neighbors classifier (KNN)](src/knearestneighbors.ipynb) - Visualizing the lazy learning algorithm

_Clustering Analaysis_  
- [Clustering Analysis Part 1](src/clustering-part1.ipynb) - K-means/++, Elbow method, Silhouette plots  
- [Clustering Analysis Part 2](src/clustering-part2.ipynb) - Hierarchical trees, Distance matrix, Dendrograms  

_Building good datasets_
- [Data Preprocessing 1](src/datapreprocessing-part1.ipynb) - Handling missing, nominal and ordinal values
- [Data Preprocessing 2](src/datapreprocessing-part2.ipynb) - Dataset partitioning, feature scaling & selecting
- [Data Preprocessing 3](src/datapreprocessing-part3.ipynb) - Sequential feature selection (SBS), Feature importance with Random Forests  

_Compressing Data via Dimensionality Reduction_
- [Principal component anaysis (PCA)](src/pca.ipynb) - Unsupervised data compression 
- [Linear discriminant analysis (LDA)](src/lda.ipynb) - Supervised dimensionality reduction
- [Kernel Principal component analysis (K-PCA)](src/kernel-pca.ipynb) - Nonlinear dimensionality

_Model Evaluation and Hyper-parameter Tuning_
- [Model Evaluation 1](src/modelevaluation-part1.ipynb) - Using pipelines and cross validation techniques
- [Model Evaluation 2](src/modelevaluation-part2.ipynb) - Learning curves, grid search, nested cross-validation
- [Model Evaluation 3](src/modelevaluation-part3.ipynb) - Precision, recall, F1-scores, ROC curves  

---

### Additional Machine Learning with Scikit-Learn

_Introduction_
- [The Classifier Interface](scikit/Chapter_1/Classification.ipynb) - LinearSVC, RandomForest, Classifier Comparison 
- [The Regressor Interface](scikit/Chapter_1/Regression.ipynb) - Ridge, RandomForestRegressor
- [The Transformer Interface](scikit/Chapter_1/Transformers.ipynb) - StandardScaler, PCA, Dimensionality Reduction
- [The Cluster Interface](scikit/Chapter_1/Clustering.ipynb) - KMeans, SpectralClustering, Overview of visuals
- [The Manifold Interface](scikit/Chapter_1/Manifold.ipynb) - Unsupervised fitting with PCA, Isomap. Non-linear dimensionality reduction for use of visuals.
- [Using Cross Validation](scikit/Chapter_1/CrossValidation.ipynb) - Splitting training/test and using cross validation to iterate scoring of classifiers
- [Grid Searches](scikit/Chapter_1/GridSearches.ipynb) - Recommend hyper-parameters (i.e, C, kernel, gamma) to be passed when building an estimator. 
- [Scikit Interface Summary](scikit/Chapter_1/API_Overview.ipynb) - Quick recap on scikit-learns interface  


_Model Complexity, Overfitting and Underfitting_  
- [Model Complexity](scikit/Chapter_2/ModelComplexity.ipynb) - Overfitting, Underfitting visuals
- [Linear models with Scikit](scikit/Chapter_2/Linearmodels.ipynb) - Linear regression, linear classification, regularization  
- [Kernel SVMs with Scikit](scikit/Chapter_2/SupportVectorMachines.ipynb) - Support vector machines, kernel SVMs, hyperparameters
- [Random Forests Preview](scikit/Chapter_2/TreesandForests.ipynb) - Decision tree classification, random forest classifier
- [Learning Curves](scikit/Chapter_2/LearningCurves.ipynb) - Learning curves for analyzing model complexity
- [Validation Curves](scikit/Chapter_2/ValidationCurves.ipynb) -  For Analyzing Model Parameters  
- [Hyperparameter CV Objects](scikit/Chapter_2/EstimatorCVObjects.ipynb) - Efficient Parameter Search with EstimatorCV Objects

_Using Pipelines in Scikit Learn_ 
- [Motivation of using pipelines](scikit/Chapter_3/PipelinesmMtivation.ipynb) - Why pipelines, how not to do grid-searches.    
- [Defining a pipeline and basic usage](scikit/Chapter_3/PipelineBasics.ipynb) - Examples of using pipelines and without pipelines.  
- [Cross-validation with pipelines](scikit/Chapter_3/Cross_Validation_with_Pipelines.ipynb) - Cross-validation with/without pipelines
- [Parameter selection with pipelines](scikit/Chapter_3/Parameter_Selection_with_Pipelines.ipynb) -  Feature selection, grid-search using pipelines


---

### Readings
- [Python Machine Learning, Sebastian Raschka](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka-ebook/dp/B00YSILNL0#navbar)  
- [Applied Predictive Analytics, Dean Abott](https://www.amazon.com/Applied-Predictive-Analytics-Principles-Professional/dp/1118727967)  
- [Advanced Machine Learning with scikit-learn, Andreas Mueller](https://www.amazon.com/Advanced-Machine-Learning-scikit-learn-Training/dp/B015WPK674)

---

### Other Notes

- [Machine Learning, Stanford University (Andrew Ng) - Using octave](https://github.com/jaysonfrancis/coursera/tree/master/machinelearning-stanford)  
- [140 Machine Learning Formulas](docs/140 Machine Learning Formulas.pdf) 