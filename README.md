### 简介
我写了一篇关于多标签分类的综述论文"A Review on Multi-Label Learning Algorithms"的总结与反思，[详见](A Summary and Reflection on 'A Review of Multi-Label Learning Algorithms'.md)

### 数据集
酵母数据集包含2417个条目，涵盖微阵列表达数据和系统发育档案。每个基因有103个描述性特征，并与多个功能类别相关联。在这个数据集的版本中，共有14个不同的功能类别，一个基因可以属于多个功能类别。数据集以CSV格式存储，位于数据存储库的“data”文件夹中。
### Binary Relevance Classifier
实现一个二元关联分类器，其中为每个标签实现了独立的基础分类器。方法包括：Decision Tree,Random Forest,Logistic Regression,GaussianNB,kNN,SVM
### Classifier Chains
简单二元关联分类器方法不足是，它没有利用多标签分类情景中标签之间的关联。通过将标签之间的关联性纳入模型来提升分类效果。在分类器链中，每个标签都有一个独立的二元分类器，这些分类器按照特定的顺序进行排列。每个分类器在预测时不仅使用原始输入特征，还会使用前面分类器的预测结果作为附加特征。这样，每个分类器的输出可以影响后续分类器的预测，从而允许模型捕捉和利用标签之间的潜在关联。
### References
M. -L. Zhang and Z. -H. Zhou, "A Review on Multi-Label Learning Algorithms," in IEEE Transactions on Knowledge and Data Engineering, vol. 26, no. 8, pp. 1819-1837, Aug. 2014, doi: 10.1109/TKDE.2013.39. 
https://ieeexplore.ieee.org/document/6471714
