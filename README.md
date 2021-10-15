# text-dependency-parser
依存关系分析
[![chatoper banner][co-banner-image]][co-url]

[co-banner-image]: https://user-images.githubusercontent.com/3538629/42383104-da925942-8168-11e8-8195-868d5fcec170.png
[co-url]: https://www.chatopera.com

# 目录

* [数据](https://github.com/chatopera/text-dependency-parser#%E6%95%B0%E6%8D%AE)

* [运行程序：训练模型和评测](https://github.com/chatopera/text-dependency-parser#%E6%89%A7%E8%A1%8C)

* [浏览依存关系](https://github.com/chatopera/text-dependency-parser#%E6%B5%8F%E8%A7%88%E4%BE%9D%E5%AD%98%E5%85%B3%E7%B3%BB)

* [附录：词性解释](https://github.com/chatopera/text-dependency-parser#%E8%AF%8D%E6%80%A7%E8%A7%A3%E9%87%8A)

* [附录：句法分析（句法树）](https://github.com/chatopera/text-dependency-parser#%E5%8F%A5%E6%B3%95%E5%88%86%E6%9E%90%E5%8F%A5%E6%B3%95%E6%A0%91)

* [附录：关系表示](https://github.com/chatopera/text-dependency-parser#%E5%85%B3%E7%B3%BB%E8%A1%A8%E7%A4%BA)

# 数据
格式说明: [CoNLL-U Format](http://universaldependencies.org/docs/format.html)

在本程序中，至少需要该格式的前10列数据：

| 列 | 名称 | 含义 |
| --- | --- | --- |
| 1 | ID |  从1开始的单词ID |
| 2 | FORM | 单词 |  
| 3 | LEMMA | 英文词根，中文[义原](http://www.keenage.com/zhiwang/c_zhiwang_r.html) |  
| 4 | UPOSTAG | 词性（跨语言抽象出来的一套词性） |  
| 5 | XPOSTAG | 非通用词性（该语言特有的）  |  
| 6 | FEATS | 形态特点 |  
| 7 | HEAD | 这个词所属的父亲节点 |  
| 8 | DEPREL | 与父亲节点的[关系](http://universaldependencies.org/docs/u/dep/index.html) |  
| 9 | DEPS | 次要的关联节点 |  
| 10 | MISC | 其他补充信息 |  

* 在上述各列中，值是 '\_' 时代表不可用。

* 本程序中，允许 3，5，6，9，10列为 '\_'，其它列为必须为有效值。

## data 目录下的数据
* 中文数据集: UD_Chinese-GSD

https://github.com/UniversalDependencies/UD_Chinese-GSD

* 英文数据集: UD_English-EWT

https://github.com/UniversalDependencies/UD_English-EWT


* 第二届自然语言处理与中文计算会议（NLP&CC 2013）

[清华和哈工大提供的训练集和开发集](./data/evsam05)

查看[其他数据集](https://github.com/chatopera/text-dependency-parser/issues/2)。

# 算法

<img width="750" alt="screen shot 2018-03-24 at 11 38 57 am" src="https://user-images.githubusercontent.com/3538629/37860014-59795f7a-2f58-11e8-85fc-854f0160ae79.png">

[详细介绍: Dependency Parsing](https://web.stanford.edu/~jurafsky/slp3/14.pdf)

在具体更新句子的依存树的时候，有两个思路：standard（从底到顶） 和 eager（从上到下）。

# 执行

### 安装

依赖: **py2.7**

```
pip install -r requirements.txt
```

## 训练模型

* Nivre's Arc-Standard

```
admin/standard.thu.train.sh # 训练中文模型
admin/standard.thu.test.sh  # 测试中文模型

admin/standard.ewt.train.sh # 训练英文模型
admin/standard.ewt.test.sh  # 测试英文模型
```

* Nivre's Arc-Eager
```
admin/eager.thu.train.sh # 训练中文模型
admin/eager.thu.test.sh  # 测试中文模型

admin/eager.ewt.train.sh # 训练英文模型
admin/eager.ewt.test.sh  # 测试英文模型
```

针对 *UD_Chinese-GSD* 的结果:

```
I0316 23:19:25.249176 140736085984064 eager.py:152] accuracy: 0.760666326704
I0316 23:19:25.249367 140736085984064 eager.py:153] complete: 0.206
I0316 23:19:25.389566 140736085984064 eager.py:156] recall: 0.745088245088
I0316 23:19:25.391751 140736085984064 eager.py:158] precision: 0.760666326704
I0316 23:19:25.391916 140736085984064 eager.py:159] assigned: 0.97952047952
```

## 浏览依存关系

使用 conllu.js 浏览依存关系：打开[网页](http://hailiang-wang.github.io/conllu.js/)，点击"edit"按钮，然后粘贴CoNLL-U 格式内容到编辑器中。比如粘贴下面的内容到[conllu.js](http://samurais.github.io/conllu.js/) 网页中。

```
1 就 _ RB RB _ 7 mark _ SpaceAfter=No
2 像 _ IN IN _ 6 case _ SpaceAfter=No
3 所有 _ DT DT _ 6 det _ SpaceAfter=No
4 的 _ DEC DEC _ 3 case:dec _ SpaceAfter=No
5 大 _ PFA PFA _ 6 case:pref _ SpaceAfter=No
6 賣場 _ NN NN _ 7 nmod _ SpaceAfter=No
7 一樣 _ JJ JJ _ 15 acl _ SpaceAfter=No
8 , _ , , _ 15 punct _ SpaceAfter=No
9 宜家 _ NNP NNP _ 10 nmod _ SpaceAfter=No
10 家居 _ NN NN _ 11 nsubj _ SpaceAfter=No
11 吸引 _ VV VV _ 14 acl:relcl _ SpaceAfter=No
12 的 _ DEC DEC _ 11 mark:relcl _ SpaceAfter=No
13 消費 _ VV VV _ 14 case:suff _ SpaceAfter=No
14 者 _ SFN SFN _ 15 nsubj _ SpaceAfter=No
15 來 _ VV VV _ 0 root _ SpaceAfter=No
16 自 _ VV VV _ 15 mark _ SpaceAfter=No
17 於 _ VV VV _ 15 mark _ SpaceAfter=No
18 範圍 _ NN NN _ 20 nsubj _ SpaceAfter=No
19 非常 _ RB RB _ 20 advmod _ SpaceAfter=No
20 廣大 _ JJ JJ _ 22 acl:relcl _ SpaceAfter=No
21 的 _ DEC DEC _ 20 mark:relcl _ SpaceAfter=No
22 地區 _ NN NN _ 15 obj _ SpaceAfter=No
23 . _ . . _ 15 punct _ SpaceAfter=No

```

> 注意：粘贴时包括17行下面的空行，因为空白行作为句子之间的标志。

得到如下的依存关系树：

<img width="750" alt="screen shot 2018-03-16 at 11 21 25 pm" src="https://user-images.githubusercontent.com/3538629/37528966-e488e9e8-2970-11e8-8ac0-f4dd7b783e99.png">


# 代码结构

*app/standard.py*  和 *app/eager.py* 是训练代码，**transition parser**的核心实现在*app/transitionparser.py*中。

## parser

* 父类：TransitionBasedParser
* 子类：ArcStandardParser2, ArcEagerParser

## configuration

* 父类：Configuration
* 子类：ArcStandardConfiguration, ArcEagerConfiguration

<img src="https://user-images.githubusercontent.com/3538629/37859794-980085e2-2f54-11e8-9f85-f050213cb2e8.png" width="600">

依赖：ArcEagerConfiguration --> ArcEagerParser, ArcStandardConfiguration --> ArcStandardParser2


## oracle

<img src="https://user-images.githubusercontent.com/3538629/37859795-985e457e-2f54-11e8-8a7f-98992b059922.png" width="600">

## decider

<img src="https://user-images.githubusercontent.com/3538629/37859799-9a353592-2f54-11e8-813c-569fdefc7227.png" width="600">


## feature extractor

<img src="https://user-images.githubusercontent.com/3538629/37859923-a6cc8a06-2f56-11e8-9efa-f0a0f5252c7c.png" width="600">

# 训练

## standard

<img src="https://user-images.githubusercontent.com/3538629/37859987-b8e51dc4-2f57-11e8-84df-c806359c9e90.png" width="600">

代码：
<img width="600" alt="screen shot 2018-03-24 at 11 31 31 am" src="https://user-images.githubusercontent.com/3538629/37859960-3f28b1c6-2f57-11e8-809b-c1a0742f731b.png">

## eager

<img src="https://user-images.githubusercontent.com/3538629/37859988-b94589d4-2f57-11e8-985b-8790b51a3b95.png" width="600">

代码：
<img width="600" alt="screen shot 2018-03-24 at 11 32 15 am" src="https://user-images.githubusercontent.com/3538629/37859961-3f9f85bc-2f57-11e8-9099-a0e7e7e2853e.png">

# 更多内容

[介绍句法分析](https://github.com/Samurais/text-dependency-parser/issues/1)

# 感谢

[CoNLL-2009 Shared Task: Syntactic and Semantic Dependencies in Multiple Languages](http://ufal.mff.cuni.cz/conll2009-st/task-description.html)

[Transition Based Dependency Parsers](https://www.cs.bgu.ac.il/~yoavg/software/transitionparser/)

[conllu.js](https://github.com/spyysalo/conllu.js)

[python︱六款中文分词模块尝试:jieba、THULAC、SnowNLP、pynlpir、CoreNLP、pyLTP](http://blog.csdn.net/sinat_26917383/article/details/77067515?%3E)

# References

[1] Liang Huang, Wenbin Jiang and Qun Liu. 2009. Bilingually-Constrained (Monolingual) Shift-Reduce Parsing.
    
# 附录

## 词性、句法分析、依存关系的符号解释

### 词性解释

```
CC: conjunction, coordinatin 表示连词 
CD: numeral, cardinal 表示基数词 
DT: determiner 表示限定词 
EX: existential there 存在句 
FW: foreign word 外来词 
IN: preposition or conjunction, subordinating 介词或从属连词 
JJ: adjective or numeral, ordinal 形容词或序数词 
JJR: adjective, comparative 形容词比较级 
JJS: adjective, superlative 形容词最高级 
LS: list item marker 列表标识 
MD: modal auxiliary 情态助动词 
NN: noun, common, singular or mass 
NNS: noun, common, plural 
NNP: noun, proper, singular 
NNPS: noun, proper, plural 
PDT: pre-determiner 前位限定词 
POS: genitive marker 所有格标记 
PRP: pronoun, personal 人称代词 
PRP:pronoun,possessive所有格代词RB:adverb副词RBR:adverb,comparative副词比较级RBS:adverb,superlative副词最高级RP:particle小品词SYM:symbol符号TO:”to”asprepositionorinfinitivemarker作为介词或不定式标记UH:interjection插入语VB:verb,baseformVBD:verb,pasttenseVBG:verb,presentparticipleorgerundVBN:verb,pastparticipleVBP:verb,presenttense,not3rdpersonsingularVBZ:verb,presenttense,3rdpersonsingularWDT:WH−determinerWH限定词WP:WH−pronounWH代词WP: WH-pronoun, possessive WH所有格代词 
WRB:Wh-adverb WH副词
```

[中文词性标注标准：ICTPOS3.0词性标记集](http://hailiang-wang.github.io/development/2017/04/28/chinese-pos-tagging/)

### 句法分析（句法树）

```
ROOT：要处理文本的语句 
IP：简单从句 
NP：名词短语 
VP：动词短语 
PU：断句符，通常是句号、问号、感叹号等标点符号 
LCP：方位词短语 
PP：介词短语 
CP：由‘的’构成的表示修饰性关系的短语 
DNP：由‘的’构成的表示所属关系的短语 
ADVP：副词短语 
ADJP：形容词短语 
DP：限定词短语 
QP：量词短语 
NN：常用名词 
NR：固有名词：表示仅适用于该项事物的名词，含地名，人名，国名，书名，团体名称以及一事件的名称等。 
NT：时间名词 
PN：代词 
VV：动词 
VC：是 
CC：表示连词 
VE：有 
VA：表语形容词 
AS：内容标记（如：了） 
VRD：动补复合词 
CD: 表示基数词 
DT: determiner 表示限定词 
EX: existential there 存在句 
FW: foreign word 外来词 
IN: preposition or conjunction, subordinating 介词或从属连词 
JJ: adjective or numeral, ordinal 形容词或序数词 
JJR: adjective, comparative 形容词比较级 
JJS: adjective, superlative 形容词最高级 
LS: list item marker 列表标识 
MD: modal auxiliary 情态助动词 
PDT: pre-determiner 前位限定词 
POS: genitive marker 所有格标记 
PRP: pronoun, personal 人称代词 
RB: adverb 副词 
RBR: adverb, comparative 副词比较级 
RBS: adverb, superlative 副词最高级 
RP: particle 小品词 
SYM: symbol 符号 
TO:”to” as preposition or infinitive marker 作为介词或不定式标记 
WDT: WH-determiner WH限定词 
WP: WH-pronoun WH代词 
WP$: WH-pronoun, possessive WH所有格代词 
WRB:Wh-adverb WH副词
```

### 关系表示

```
abbrev: abbreviation modifier，缩写 
acomp: adjectival complement，形容词的补充； 
advcl : adverbial clause modifier，状语从句修饰词 
advmod: adverbial modifier状语 
agent: agent，代理，一般有by的时候会出现这个 
amod: adjectival modifier形容词 
appos: appositional modifier,同位词 
attr: attributive，属性 
aux: auxiliary，非主要动词和助词，如BE,HAVE SHOULD/COULD等到 
auxpass: passive auxiliary 被动词 
cc: coordination，并列关系，一般取第一个词 
ccomp: clausal complement从句补充 
complm: complementizer，引导从句的词好重聚中的主要动词 
conj : conjunct，连接两个并列的词。 
cop: copula。系动词（如be,seem,appear等），（命题主词与谓词间的）连系 
csubj : clausal subject，从主关系 
csubjpass: clausal passive subject 主从被动关系 
dep: dependent依赖关系 
det: determiner决定词，如冠词等 
dobj : direct object直接宾语 
expl: expletive，主要是抓取there 
infmod: infinitival modifier，动词不定式 
iobj : indirect object，非直接宾语，也就是所以的间接宾语； 
mark: marker，主要出现在有“that” or “whether”“because”, “when”, 
mwe: multi-word expression，多个词的表示 
neg: negation modifier否定词 
nn: noun compound modifier名词组合形式 
npadvmod: noun phrase as adverbial modifier名词作状语 
nsubj : nominal subject，名词主语 
nsubjpass: passive nominal subject，被动的名词主语 
num: numeric modifier，数值修饰 
number: element of compound number，组合数字 
parataxis: parataxis: parataxis，并列关系 
partmod: participial modifier动词形式的修饰 
pcomp: prepositional complement，介词补充 
pobj : object of a preposition，介词的宾语 
poss: possession modifier，所有形式，所有格，所属 
possessive: possessive modifier，这个表示所有者和那个’S的关系 
preconj : preconjunct，常常是出现在 “either”, “both”, “neither”的情况下 
predet: predeterminer，前缀决定，常常是表示所有 
prep: prepositional modifier 
prepc: prepositional clausal modifier 
prt: phrasal verb particle，动词短语 
punct: punctuation，这个很少见，但是保留下来了，结果当中不会出现这个 
purpcl : purpose clause modifier，目的从句 
quantmod: quantifier phrase modifier，数量短语 
rcmod: relative clause modifier相关关系 
ref : referent，指示物，指代 
rel : relative 
root: root，最重要的词，从它开始，根节点 
tmod: temporal modifier 
xcomp: open clausal complement 
xsubj : controlling subject 掌控者
```


## Chatopera 云服务

[https://bot.chatopera.com/](https://bot.chatopera.com/)

[Chatopera 云服务](https://bot.chatopera.com)是一站式实现聊天机器人的云服务，按接口调用次数计费。Chatopera 云服务是 [Chatopera 机器人平台](https://docs.chatopera.com/products/chatbot-platform/index.html)的软件即服务实例。在云计算基础上，Chatopera 云服务属于**聊天机器人即服务**的云服务。

Chatopera 机器人平台包括知识库、多轮对话、意图识别和语音识别等组件，标准化聊天机器人开发，支持企业 OA 智能问答、HR 智能问答、智能客服和网络营销等场景。企业 IT 部门、业务部门借助 Chatopera 云服务快速让聊天机器人上线！

<details>
<summary>展开查看 Chatopera 云服务的产品截图</summary>
<p>

<p align="center">
  <b>自定义词典</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530072-da92d600-d33e-11e9-8656-01c26caff4f9.png" width="800">
</p>

<p align="center">
  <b>自定义词条</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530091-e41c3e00-d33e-11e9-9704-c07a2a02b84e.png" width="800">
</p>

<p align="center">
  <b>创建意图</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530169-12018280-d33f-11e9-93b4-9db881cf4dd5.png" width="800">
</p>

<p align="center">
  <b>添加说法和槽位</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530187-20e83500-d33f-11e9-87ec-a0241e3dac4d.png" width="800">
</p>

<p align="center">
  <b>训练模型</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530235-33626e80-d33f-11e9-8d07-fa3ae417fd5d.png" width="800">
</p>

<p align="center">
  <b>测试对话</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530253-3d846d00-d33f-11e9-81ea-86e6d47020d8.png" width="800">
</p>

<p align="center">
  <b>机器人画像</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530312-6442a380-d33f-11e9-869c-85fb6a835a97.png" width="800">
</p>

<p align="center">
  <b>系统集成</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530281-4ecd7980-d33f-11e9-8def-c53251f30138.png" width="800">
</p>

<p align="center">
  <b>聊天历史</b><br>
  <img src="https://static-public.chatopera.com/assets/images/64530295-5856e180-d33f-11e9-94d4-db50481b2d8e.png" width="800">
</p>

</p>
</details>


<p align="center">
  <b>立即使用</b><br>
  <a href="https://bot.chatopera.com" target="_blank">
      <img src="https://static-public.chatopera.com/assets/images/64531083-3199aa80-d341-11e9-86cd-3a3ed860b14b.png" width="800">
  </a>
</p>
