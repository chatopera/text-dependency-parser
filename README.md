# text-dependency-parser
依存关系分析

![](https://camo.githubusercontent.com/ae91a5698ad80d3fe8e0eb5a4c6ee7170e088a7d/687474703a2f2f37786b6571692e636f6d312e7a302e676c622e636c6f7564646e2e636f6d2f61692f53637265656e25323053686f74253230323031372d30342d30342532306174253230382e32302e3437253230504d2e706e67)

## Data
format: [CoNLL-2009 Shared Task](http://ufal.mff.cuni.cz/conll2009-st/task-description.html)

### Universal Dependencies
http://universaldependencies.org/

### 采用清华大学语义依存网络语料的20000句作为训练集。
http://www.hankcs.com/nlp/corpus/chinese-treebank.html#h3-6

### 汉语树库
http://www.hankcs.com/nlp/corpus/chinese-treebank.html

## Run

Train model.

```
admin/train.sh
```

Test model.

```
admin/test.sh
```

# Give credits to

[Transition Based Dependency Parsers](https://www.cs.bgu.ac.il/~yoavg/software/transitionparser/)

References:
~~~~~~~~~~~
[1] Liang Huang, Wenbin Jiang and Qun Liu. 2009.
    Bilingually-Constrained (Monolingual) Shift-Reduce Parsing.
    