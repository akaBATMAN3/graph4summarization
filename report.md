# Graph4Summarization

## 1	建图过程

- word_emb_size = 300


- dataset.py中解析json格式数据

  ```python
  data_item = Text2TextDataItem(
      input_text=input, # article中每行句子排列成一行
      output_text=output, # highlight中每行句子用<t></t>分隔并排列成一行
      tokenizer=self.tokenizer, # nltk推荐tokenizer
      share_vocab=self.share_vocab, # True
  )
  ```

  train/test/val目前为包含以上Text2TextDataItem实例的list

- **dependency/constituency/ie都为static graph**

### 1.1	ie_graph_construction.py中corenlp部分解析结果

1. 一句句子中分词结果

   ```python
   'tokens': [{'index': 1, 'word': '-lrb-', 'originalText': '-lrb-', 'lemma': '-lrb-', 'characterOffsetBegin': 0, 'characterOffsetEnd': 5, 'pos': 'JJ', 'ner': 'O', 'speaker': 'PER0', 'before': '', 'after': ' '}, {'index': 2, 'word': 'cnn', 'originalText': 'cnn', 'lemma': 'cnn', 'characterOffsetBegin': 6, 'characterOffsetEnd': 9, 'pos': 'NN', 'ner': 'O', 'speaker': 'PER0', 'before': ' ', 'after': ' '}]
   ```

2. 共指解析结果

   ```python
   [{'id': 67, 'text': 'the only major drawback to the nexus 4', 'type': 'NOMINAL', 'number': 'SINGULAR', 'gender': 'NEUTRAL', 'animacy': 'INANIMATE', 'startIndex': 1, 'endIndex': 9, 'headIndex': 4, 'sentNum': 10, 'position': [10, 2], 'isRepresentativeMention': True}, {'id': 69, 'text': 'it', 'type': 'PRONOMINAL', 'number': 'SINGULAR', 'gender': 'NEUTRAL', 'animacy': 'INANIMATE', 'startIndex': 11, 'endIndex': 12, 'headIndex': 11, 'sentNum': 10, 'position': [10, 4], 'isRepresentativeMention': False}]
   ```

   根据打印结果可知共指解析有错，会对下游造成影响

3. **将所有第二次及以后出现的实体用第一次出现的实体替换**

4. 一组由openIE提取的triple

   ```python
   {'subject': 'splash', 'subjectSpan': [16, 17], 'relation': 'was', 'relationSpan': [17, 18], 'object': 'well built', 'objectSpan': [18, 21]}
   ```

5. **获得triples[sbj, rel, obj]组成的list，并去除相似的triples，这一步达到了去冗余的效果**

6. parsed_results包含三个key

   - **graph_content的一个样例**

     ```python
     [{'edge_tokens': [], 'src': {'tokens': 'google', 'id': 0, 'type': 0}, 'tgt': {'tokens': 'has', 'id': 1, 'type': 3}}, {'edge_tokens': [], 'src': {'tokens': 'has', 'id': 1, 'type': 3}, 'tgt': {'tokens': 'nexus 4 for arrival of cheaper iphone', 'id': 2, 'type': 0}}]
     ```

   - node_num：结点数量，包括边也是结点

   - graph_nodes：包括实体结点和边结点

7. 为每篇article构建图结构，其中graph node_attributes

   - token:所代表单词
   - type:0为sub/obj（结点），3为rel（边）

8. 将sub连向rel，rel连向obj，单向无权边

### 1.2	dependency_graph_construction.py中corenlp部分解析结果

1. 整篇article作为一整个句子处理（上限400个token）
2. graph node_attributes
   - type:全0
   - token:所代表单词
   - position_id:在整篇article中的索引
   - sentence_id:在所有article中位于第几篇article
   - head:全True
   - tail:全True
3. 根据解析得到的依赖关系直接连接token，忽略边的属性，单向无权边

### 1.3	constituency_graph_construction.py中corenlp部分解析结果

1. 对分句后的每句句子建立静态图

2. 将每句句子的pos解析结果去掉重复括号，去掉root，倒数第二个.换成period，各部分存储在list中

   ```python
   ['(', 'S', '(', 'S', '(', 'NP', '(', 'JJ', '-lrb-', ')', '(', 'NN', 'cnn', ')', ')', '(', 'VP', '(', 'VBZ', '-rrb-', ')', ')', ')', '(', ':', '--', ')', '(', 'S', '(', 'S', '(', 'NP', '(', 'NP', '(', 'NP', '(', 'NNP', 'google', ')', '(', 'POS', "'s", ')', ')', '(', 'NN', 'nexus', ')', ')', '(', 'NP-TMP', '(', 'CD', '4', ')', ')', ')', '(', 'VP', '(', 'VBD', 'made', ')', '(', 'NP', '(', 'DT', 'a', ')', '(', 'NN', 'splash', ')', ')', '(', 'NP-TMP', '(', 'JJ', 'last', ')', '(', 'NN', 'fall', ')', ')', '(', 'SBAR', '(', 'ADVP', '(', 'RB', 'simply', ')', ')', '(', 'IN', 'because', ')', '(', 'S', '(', 'NP', '(', 'PRP', 'it', ')', ')', '(', 'VP', '(', 'VBD', 'was', ')', '(', 'ADJP', '(', 'JJ', 'well-built', ')', '(', 'CC', 'and', ')', '(', 'JJ', 'inexpensive', ')', ')', ')', ')', ')', ')', ')', '(', ',', ',', ')', '(', 'CC', 'and', ')', '(', 'S', '(', 'CC', 'yet', ')', '(', 'NP', '(', 'PRP', 'it', ')', ')', '(', 'VP', '(', 'VBD', 'did', ')', '(', 'RB', "n't", ')', '(', 'VP', '(', 'VB', 'require', ')', '(', 'NP', '(', 'DT', 'a', ')', '(', 'JJ', 'two-year', ')', '(', 'NN', 'contract', ')', ')', '(', 'PP', '(', 'IN', 'with', ')', '(', 'NP', '(', 'DT', 'a', ')', '(', 'JJ', 'wireless', ')', '(', 'NN', 'carrier', ')', ')', ')', ')', ')', ')', ')', '(', 'period', '.', ')', ')']
   ```

  3. node_attribudes

     - token:单词，单词词性

     - type:单词为0，单词词性为1

     - position_id:单词为单词索引，单词词性为None

     - sentence_id:该token所在句子在article中的索引，从1开始

     - tail:句子中最后一个单词的尾为True，其余为false

     - false:句子中第一个单词的头为True，其余为false，应该是为了图的合并而设置

       **注意：处理过程中先生成词性结点，然后为词性结点间添加边，之后会用单词结点代替单词对应的词性结点，注意只删除单词对应词性，保留其余词性**

       ```python
       {'Edges': [(1, 0), (2, 1), (3, 2), (4, 2), (5, 1), (6, 5), (7, 0), (8, 0), (9, 8), (10, 9), (11, 10), (12, 11), (13, 12), (14, 12), (15, 11), (16, 10), (17, 16), (18, 9), (19, 18), (20, 18), (21, 20), (22, 20), (23, 18), (24, 23), (25, 23), (26, 18), (27, 26), (28, 27), (29, 26), (30, 26), (31, 30), (32, 31), (33, 30), (34, 33), (35, 33), (36, 35), (37, 35), (38, 35), (39, 8), (40, 8), (41, 8), (42, 41), (43, 41), (44, 43), (45, 41), (46, 45), (47, 45), (48, 45), (49, 48), (50, 48), (51, 50), (52, 50), (53, 50), (54, 48), (55, 54), (56, 54), (57, 56), (58, 56), (59, 56), (60, 0)]}
       ```

  4. 删除线性排列的结点重新建图

  5. 为所有句子（子图）间添加双向边，并令单词结点排在前面，词性结点排在后面，形成大图返回

     大图中一句句子各结点连接情况

     ```
     S 	---	 S|NP 	---	 S|-lrb- 	---	 NP|cnn 	---	 NP|-- 	---	 S|S 	---	 S|S 	---	 S
     NP 	---	 S|NP 	---	 NP|NP 	---	 NP|google 	---	 NP|'s 	---	 NP|nexus 	---	 NP|VP 	---	 S
     made 	---	 VP|NP 	---	 VP|a 	---	 NP|splash 	---	 NP|NP-TMP 	---	 VP|last 	---	 NP-TMP
     fall 	---	 NP-TMP|SBAR 	---	 VP|because 	---	 SBAR|S 	---	 SBAR|VP 	---	 S
     was 	---	 VP|ADJP 	---	 VP|well-built 	---	 ADJP|and 	---	 ADJP|inexpensive 	---	 ADJP
     , 	---	 S|and 	---	 S|S 	---	 S|yet 	---	 S|VP 	---	 S|did 	---	 VP|n't 	---	 VP
     VP 	---	 VP|require 	---	 VP|NP 	---	 VP|a 	---	 NP|two-year 	---	 NP|contract 	---	 NP
     PP 	---	 VP|with 	---	 PP|NP 	---	 PP|a 	---	 NP|wireless 	---	 NP|carrier 	---	 NP
     . 	---	 S|-rrb- 	---	 S
     ```

### 1.4	建完图的后处理

- 每篇article即data item增加新的属性graph（根据不同建图方法建得的图），重新收集data_item

  ```python
  data_item = Text2TextDataItem(
      input_text=input, # article中每行句子排列成一行
      output_text=output, # highlight中每行句子用<t></t>分隔并排列成一行
      tokenizer=self.tokenizer, # nltk推荐tokenizer
      share_vocab=self.share_vocab, # True
      graph # 根据不同建图方法得到的图结构，包含结点和边
  )
  ```

  此时trian/test/val为由以上data_item构成，每个data_item代表一篇article

- 利用训练集建立vocabulary模型

- train/test/val向量化

  为所有结点新增属性token_id：应该是结点token在vocab中的索引

  为graph新增==node_features==：一个data_item中所有结点的token_id组成的矩阵，大小为结点数*1（ie图除外）

  data_item新增属性output_np：output_text转换为单词索引并加上终止符

  ```python
  data_item = Text2TextDataItem(
      input_text=input, # article中每行句子排列成一行
      output_text=output, # highlight中每行句子用<t></t>分隔并排列成一行
      tokenizer=self.tokenizer, # nltk推荐tokenizer
      share_vocab=self.share_vocab, # True
      graph # 根据不同建图方法得到的图结构，包含结点和边
      output_np # output_text转换为单词索引并加上终止符
  )
  ```

- 将{train,test,val}保存为data.pt，将建立的vocabulary模型保存为vocab.pt

- 重新加载data(包含train,test,val)和vocab

## 2	模型结构

```python
SumModel(
  (g2s): Graph2Seq(

    (graph_initializer): GraphEmbeddingInitialization(
      (embedding_layer): EmbeddingConstruction(
        (word_emb_layers): ModuleDict(
          (w2v): WordEmbedding(
            (word_emb_layer): Embedding(70004, 300, padding_idx=0)
          )
        )
        (seq_info_encode_layer): RNNEmbedding(
          (model): LSTM(300, 150, batch_first=True, bidirectional=True)
        )
      )
    )#GraphEmbeddingInitialization end

    (enc_word_emb): WordEmbedding(
      (word_emb_layer): Embedding(70004, 300, padding_idx=0)
    )

    (gnn_encoder): GCN(
      (gcn_layers): ModuleList(
        (0): GCNLayer(
          (model): BiFuseGCNLayerConv(
            (_feat_drop): Dropout(p=0.0, inplace=False)
            (fuse_linear): Linear(in_features=1200, out_features=300, bias=True)
            (res_fc): Identity()
          )
        )#GCNLayer0 end
        (1): GCNLayer(
          (model): BiFuseGCNLayerConv(
            (_feat_drop): Dropout(p=0.0, inplace=False)
            (fuse_linear): Linear(in_features=1200, out_features=300, bias=True)
            (res_fc): Identity()
          )
        )
        (2): GCNLayer(
          (model): BiFuseGCNLayerConv(
            (_feat_drop): Dropout(p=0.0, inplace=False)
            (fuse_linear): Linear(in_features=1200, out_features=300, bias=True)
            (res_fc): Identity()
          )
        )
      )
    )#GCN end

    (dec_word_emb): Embedding(70004, 300, padding_idx=0)

    (seq_decoder): StdRNNDecoder(
      (dropout): Dropout(p=0.3, inplace=False)
      (tgt_emb): Embedding(70004, 300, padding_idx=0)
      (rnn): LSTM(600, 512)
      (enc_attention): Attention(
        (dropout): Dropout(p=0.2, inplace=False)
        (query_in): Linear(in_features=1024, out_features=512, bias=True)
        (memory_in): Linear(in_features=300, out_features=512, bias=False)
        (out): Linear(in_features=512, out_features=1, bias=False)
      )
      (rnn_attention): Attention(
        (dropout): Dropout(p=0.2, inplace=False)
        (query_in): Linear(in_features=1024, out_features=512, bias=True)
        (memory_in): Linear(in_features=300, out_features=512, bias=False)
        (out): Linear(in_features=512, out_features=1, bias=False)
      )
      (encoder_decoder_adapter): ModuleList(
        (0): Linear(in_features=300, out_features=512, bias=True)
        (1): Linear(in_features=300, out_features=512, bias=True)
      )
      (memory_to_feed): Sequential(
        (0): Linear(in_features=1112, out_features=300, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.3, inplace=False)
      )
      (pre_out): Linear(in_features=300, out_features=300, bias=True)
      (out_project): Linear(in_features=300, out_features=70004, bias=False)
      (ptr): Linear(in_features=2224, out_features=1, bias=True)
    )#StdRNNDecoder end
  )#graph2seq end
  (word_emb): Embedding(70004, 300, padding_idx=0)

  (loss_calc): Graph2SeqLoss(
    (loss_func): SeqGenerationLoss(
      (loss_ce): GeneralLoss(
        (loss_function): NLLLoss()
      )
      (loss_coverage): CoverageLoss()
    )
  )#Graph2SeqLoss end
)
```

## 3	前向传播过程

- 在collate_fn中将一个batch中的token_id填充至等长（取决于最大结点数，不一定全部一样长，最后一个可能达不到最大结点数那么大），一个batch中的图组成一个大图，一个batch中的output_np填充至等长作为tgt_seq，最终得到组成一个batch的data

  ```python
  data={
  	graph_data # 多个graph组成的大图
  	tgt_seq # 目标文本的单词索引序列
  	output_str # 目标文本
  }
  ```

- 利用复制机制构建oov_dict并重新构建tgt_seq

### 3.1	graph_initializer

- token_id大小为[4,400,1]经过词嵌入层得到对应词嵌入word_feat大小为[4,400,300]，这里4应该为batch_size，400应为填充后一致的token数量，300为每个词嵌入的大小，融合为大小[400,300]的new_feat送入seq_info_encode_layer
- 使用seq_info_encode_layer（BiLSTM）计算后得到的隐藏状态更新图的==node_features["node_feat"]==，这一步算是提取了时序、位置信息

### 3.2	gnn_encoder

#### 3.2.1	gcn

1. 输入为node_feat，大小为[1489,300]，[一个batch中结点数之和，词嵌入大小]，经过GCN卷积层，每次输出得到的隐藏状态h大小为[一个batch中结点数之和，词嵌入大小]
2. GCN最后一层输出更新==node_features["node_emb"]==，大小仍为[一个batch中结点数之和，词嵌入大小]
2. node_feat送入GCN提取特征，使用最后一层输出更新结点特征node_emb

#### 3.2.2	gat

- 与GCN类似

- 使用node_features["node_feat"]更新==node_features["rnn_emb"]==，即由rnn_emb存储模型在seq_info_encode_layer（BiLSTM）中提取的时序、位置信息

### 3.3	seq_decoder

- 当前==node_features=['node_feat', 'node_emb', 'token_id', 'token_id_oov', 'rnn_emb']==

  一个（dependency）结点的node_attributes={'node_attr': None, 'type': 0, 'token': '-lrb-', 'position_id': 0, 'sentence_id': 0, 'head': True, 'tail': False, 'token_id': 39}

- 抽取参数

  ```python
  params={
  	"graph_node_embedding": graph_node_emb, # gcn更新后的node embedding
  	# [batch size, node num, word emb size]
  	"graph_node_mask": graph_node_mask, # 压缩后的node_features["token_id"]
  	# [batch size, node num]
  	"rnn_node_embedding": rnn_node_emb, # 一开始BiLSTM提取的信息
  	"graph_level_embedding": graph_level_emb, # none，图级别的embedding，可用于初始化解码状态
  	"graph_edge_embedding": None,
  	"graph_edge_mask": None,
  	"src_seq": src_seq_ret.long() if self.use_copy else None, # node_features["token_id_oov"]
      "tgt_seq": data["tgt_seq"] # 由oov_dict构建的tgt_seq
      "teacher_forcing_rate": # 1.0
      "oov_dict":
  }
  ```
  
- 目标句子的长度取决于min(max_decoder_step, tgt seq len)

- 在每个解码step中

  ```python
  for i in range(target_len): # 单步解码
      (
          decoder_output,
          input_feed,
          decoder_state,
          dec_attn_scores,
          coverage_vec,
      ) = self.decode_step(
          decoder_input=decoder_input, # 采用真值而非上一步预测结果作为当前时刻解码器输入，初始化为全1
          input_feed=input_feed, # 输入，初始化为全0
          rnn_state=decoder_state,
          dec_input_mask=graph_node_mask, # token_id
          encoder_out=graph_node_embedding, # gcn更新后的node embedding
          rnn_emb=rnn_node_embedding, # 一开始BiLSTM提取的时序信息
          enc_attn_weights_average=enc_attn_weights_average,
          src_seq=src_seq, # token_id_oov
          oov_dict=oov_dict,
      )
  ```

  1. 使用真值作为解码器输入（decoder_input），经过一层由out word vocab构建的word embedding得到dec_emb
  1. 将enc_attn_weights_average求和更新覆盖向量coverage_vec
  1. 上一步解码状态decoder_state经过rnn得到dec_out以及rnn_state
  1. 使用coverage_vec和coverage_weight更新coverage_repr
  5. 分别对gcn更新后的node embedding和一开始由BiLSTM提取的序列信息rnn_emb使用注意力机制，分别收集结果存储在attn_collect和score_collect
  1. 将对node_emb和rnn_emb的注意力结果融合得到attn_total，并融合attn_total和dec_out得到解码器输出decoder_output，并利用score_collect计算dec_attn_scores
  1. decoder_output经过一层线性分类器得到输出out
  1. out再经过一层pre_out线性层和dropout得到out_emb，再经过一层线性out_project得到新的decoder_output
  1. 使用复制机制更新decoder_output
  1. 输出decoder_output, out, rnn_state, dec_attn_scores, coverage_vec

- 每一步decoder_output组成outputs list，融合后得到ret，最终输出ret，由dec_attn_scores组成的enc_attn_weights_average list和coverage_vec组成的coverage_vectors list

## 4	改进方案

1. GCN最后一层输出再加上node_feat以增强时序信息
2. dependency graph是单词级别的，constituency是短语级别的
3. graph_pooling_strategy未启用，是否可以考虑生成graph_level_embedding加入解码过程，可以指导解码器生成结构性强的摘要
4. GCN多层堆叠后使用残差
5. teacher forcing rate解决曝光偏差

## 5	实验

### 5.1	数据集

​	采用CNN/DailyMail数据集，CNN/DailyMail是最广泛使用的标准文档摘要数据集，其中包含了网络新闻文章（平均781个单词，）和多句摘要（平均3.75句句子或56个单词）。标准的CNN/DailyMail数据集包含287227个训练样例，13368个验证样例和11490个测试样例。在实验中，采用较小的数据量，并且由于需要使用CoreNLP进行在线分词解析，因此可能导致丢失部分数据。

### 5.2	实验结果及分析

​	首先，对比三种建图方法，对三种图构建方法都使用相同的节点嵌入初始化方法，使用相同的GCN编码器，都不生成图嵌入，并且都采用分离注意力机制和覆盖机制而不采用复制机制，保持其余条件相同的实验结果如下。

| Compare graph construction methods | ROUGE-L(%) |
| ---------------------------------- | :--------: |
| **dependency**-GCN-attn-cov        |   29.23    |
| **constituency**-GCN-attn-cov      | **29.98**  |
| **IE**-GCN-attn-cov                |   27.51    |

共同点是三种图都产生了不同程度的冗余，或重复当前句子、上一句句子中的单词、词组，或直接复制上一句句子（这一点在选区图中尤为明显），并且这是在增加了覆盖机制后依旧导致的问题，这说明采用的覆盖机制仍有很大的提升空间。此外，三种图还出现了生成摘要不准确、无法很好概括原文的共同点，这一点主要体现在原文中出现人物对话的情形下，生成的摘要可能简单的从原文中复制人物说的话，这样的摘要没有重点且人物角色含糊不清，会对阅读者造成困惑，为解决这样的问题，可能需要在数据预处理时就提取人物对话文本并对人物对话逻辑关系进行处理。在这三种图中依赖图和选区图表现相近，这是因为二者都从原文中提取了不同角度的语法、句法关系，从预测摘要中可以发现二者生成的摘要大多都有着语义连贯、可读性强和较简洁的特点；反观IE图发现其生成的摘要部分存在语序混乱现象，甚至部分摘要无法组成一句句子或词组，这一点也是在原文中出现人物对话时较为明显，但是IE图形成的部分摘要能够高度概括一段话，有着什么人做了什么事或者什么地方发生了什么事这样一种语序关系的特点，这种特点的摘要非常适合新闻类型的数据，并且IE图提取摘要的角度比依赖图和选区图更多，能够多角度概括原文，一定程度上也减轻了冗余现象。

​	接着，对比了四种图神经网络编码器对生成摘要质量的影响。对于GCN，使用每个节点的出度与入度和来进行特征归一化；对于GAT，设置多头注意力数量为3，激活函数采用ReLU，负斜率角度为0.2；对于GraphSage，使用LSTM作为信息聚集函数，归一化方法采用批量归一化，激活函数采用ReLU；对于GGNN，设置边的类型数量为1。图构建方法统一采用IE图，解码强化机制采用分离注意力和覆盖机制。保持其余条件相同的情况如下。

| Compare graph encoders    | ROUGE-L(%) |
| ------------------------- | :--------: |
| IE-**GCN**-attn-cov       | **27.51**  |
| IE-**GAT**-attn-cov       |   27.10    |
| IE-**GraphSAGE**-attn-cov |   27.10    |
| IE-**GGNN**-attn-cov      |   27.40    |

由于在实验中仅堆叠三层图神经网络，并且每个节点更新自身表示时仅考虑相邻一个单位的节点作为邻点，而在图中一个节点可能不光需要从其相连的某个节点中聚集信息，还可能需要从相连的节点连接的另一个节点中聚集信息（即两跳节点），鉴于以上分析，一方面，可能因没有最大程度挖掘图神经网络聚集信息的能力而导致四种图神经网络在数据集上表现相近，另一方面，也可能是由于采用的图构建方法（IE图）限制了图神经网络的表现能力，这一点是由于观察了几组实验结果发现在四种图神经网络生成的摘要中再现了上一组实验中IE图出现的一些问题，这也充分说明了一开始所采用的图构建方法对模型后续表现具有决定性作用。

​	接着验证不同图嵌入生成方法，设定图构建方法为IE图，编码器为GraphSage，解码强化机制采用分离注意力和覆盖机制，在保持其余条件完全相同的情况下，得到的实验结果如下。

| Compare graph embedding            | ROUGE-L(%) |
| ---------------------------------- | :--------: |
| IE-**GraphSAGE**-attn-cov          |   27.10    |
| IE-**GraphSAGE**-**mean**-attn-cov |   27.21    |
| IE-**GraphSAGE**-**max**-attn-cov  | **27.54**  |

使用最大池化生成图嵌入加入解码过程表现最佳，并且使用平均池化生成图嵌入加入解码过程也会比完全不使用图嵌入的表现要好。产生这样的原因是因为图嵌入通过学习节点嵌入表示而获得了整个图的结构信息，使用图嵌入初始化解码器可以令模型在解码刚开始时就得到指导而洞察输入图的结构，经由这种指导生成的摘要更富含语义语法结构特征，产生的句子语序也更连贯。

​	最后，为了探究三种解码强化机制存在的必要性，设定图构建方法为选区图，编码器为GCN，都不采用图嵌入，并且在保持其余条件完全相同的情况下，得到的实验结果如下。

| Compare decoder mechanism      | ROUGE-L(%) |
| ------------------------------ | :--------: |
| constituency-GCN-attn-cov-copy | **30.94**  |
| constituency-GCN-attn-cov      |   29.98    |
| constituency-GCN-attn          |   30.11    |
| constituency-GCN               |   29.92    |

根据在测试集上的表现可知在解码过程中同时采用分离注意力机制、覆盖机制和复制机制的表现最佳。当不采用复制机制时，可能导致生成摘要中的OOV单词过多，导致摘要准确度下降，如果能从源文本中直接复制单词可以很好地缓解该问题。当不采用覆盖机制时，模型会反复为文章中某句话或段落生成合适的摘要，具体表现为预测摘要中出现了两句一模一样的句子，或者仅有极个别单词不一样的两个句子，显然在解码过程中，当已经生成了能够概括文章中某一部分的摘要后，模型后续不应该再关注该部分内容，而是应当将注意力转向文章中的其余部分，以争求生成多角度的、更能涵盖整篇文章的摘要。当不采用分离注意力机制时，只关注节点嵌入生成的固定中间向量表示只蕴含了节点与节点间的交互信息及各节点聚集的信息，而忽略了初始序列文本转化为图节点前蕴含的时序关系和上下文信息，但是这一部分信息对于辅助生成语序正确且连贯的摘要同样非常重要。

## 6	命令汇总

### 6.1	本地启动corenlp

```python
cd stanford-corenlp-4.4.0
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

#### 6.2	建图方法比较

```python
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/graph_construction_cp/gcn_dependency.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/graph_construction_cp/gcn_constituency.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml && shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/graph_construction_cp/gcn_ie.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml && shutdown # 这个没采用覆盖和复制，采用了图嵌入，暂且保留实验结果
```

#### 6.3	图编码器比较

```c
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/graph_construction_cp/gcn_ie.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/gnn_encoder_cp/gat_ie.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/gnn_encoder_cp/graphsage_ie.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/gnn_encoder_cp/ggnn_ie.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
```

#### 6.4	图嵌入比较

```
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/graph_emb_cp/graphsage_ie_mean.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/graph_emb_cp/graphsage_ie_max.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/graph_emb_cp/graphsage_ie_min.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
```

#### 6.5	解码器比较

```
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/decoder_cp/gcn_constituency_attn_cvg_non-copy.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/decoder_cp/gcn_constituency_attn_non-cvg-copy.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
python examples/pytorch/summarization/cnn/main.py -g2s_config examples/pytorch/summarization/cnn/config/decoder_cp/gcn_constituency_non-attn-cvg-copy.yaml -task_config examples/pytorch/summarization/cnn/config/cnn.yaml; shutdown
```

### 6.6	AutoDL中打印gpu状态

```python
watch -n 0.5 nvidia-smi
```

## 参考

[1] [Graph Neural Networks for Natural Language Processing: A Survey](https://arxiv.org/abs/2106.06090)

[2] [Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks](https://arxiv.org/abs/1804.00823)

[3] [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

[4] [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

[5] [Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)

[6] [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
