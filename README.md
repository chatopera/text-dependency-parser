# text-dependency-parser
依存关系分析

![](https://camo.githubusercontent.com/ae91a5698ad80d3fe8e0eb5a4c6ee7170e088a7d/687474703a2f2f37786b6571692e636f6d312e7a302e676c622e636c6f7564646e2e636f6d2f61692f53637265656e25323053686f74253230323031372d30342d30342532306174253230382e32302e3437253230504d2e706e67)

## 数据
格式说明: [CoNLL-2009 Shared Task](http://ufal.mff.cuni.cz/conll2009-st/task-description.html)

### Universal Dependencies
http://universaldependencies.org/

### 采用清华大学语义依存网络语料的20000句作为训练集。
http://www.hankcs.com/nlp/corpus/chinese-treebank.html#h3-6

### 汉语树库
http://www.hankcs.com/nlp/corpus/chinese-treebank.html

## 执行

### 安装

依赖: **py2.7**

```
pip install -r requirements.txt
```

### 训练模型

```
admin/train.sh
```

### 测试模型

```
admin/test.sh
```

结果:

```
I0316 22:09:48.193829 140736085984064 eager.py:181] accuracy: 0.833049127081
I0316 22:09:48.194097 140736085984064 eager.py:182] complete: 0.525758305248
I0316 22:09:48.478684 140736085984064 eager.py:185] recall: 0.817580490915
I0316 22:09:48.484018 140736085984064 eager.py:187] precision: 0.833049127081
I0316 22:09:48.484204 140736085984064 eager.py:188] assigned: 0.981431303793
```

### 浏览依存关系

使用 conllu.js 浏览依存关系：打开[网页](http://samurais.github.io/conllu.js/)，点击"edit"按钮，然后粘贴CoNLL-U 格式内容到编辑器中。比如粘贴下面的内容到[conllu.js](http://samurais.github.io/conllu.js/) 网页中。

```
1 He _ PRP PRP _ 4 nsubj _ _
2 was _ VBD VBD _ 4 cop _ _
3 very _ RB RB _ 4 advmod _ _
4 clean _ JJ JJ _ 0 root _ SpaceAfter=No
5 , _ , , _ 7 punct _ _
6 very _ RB RB _ 7 advmod _ _
7 nice _ JJ JJ _ 4 conj _ _
8 to _ TO TO _ 9 mark _ _
9 work _ VB VB _ 7 advcl _ _
10 with _ IN IN _ 9 obl _ _
11 and _ CC CC _ 12 cc _ _
12 gave _ VBD VBD _ 4 conj _ _
13 a _ DT DT _ 16 det _ _
14 very _ RB RB _ 15 advmod _ _
15 reasonable _ JJ JJ _ 16 amod _ _
16 price _ NN NN _ 12 obj _ SpaceAfter=No
17 . _ . . _ 4 punct _ _

```

> 注意：粘贴时包括17行下面的空行，因为空白行作为句子之间的标志。

得到如下的依存关系树：

<img width="750" alt="1" src="https://user-images.githubusercontent.com/3538629/37527699-9c425aa0-296d-11e8-8054-67e865757af3.png">

# 感谢

[Transition Based Dependency Parsers](https://www.cs.bgu.ac.il/~yoavg/software/transitionparser/)

[conllu.js](https://github.com/spyysalo/conllu.js)

References:
~~~~~~~~~~~
[1] Liang Huang, Wenbin Jiang and Qun Liu. 2009.
    Bilingually-Constrained (Monolingual) Shift-Reduce Parsing.
    