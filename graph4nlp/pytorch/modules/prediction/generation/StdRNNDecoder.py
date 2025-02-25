import random
from functools import reduce
import torch
import torch.nn as nn

from graph4nlp.pytorch.modules.prediction.generation.attention import Attention
from graph4nlp.pytorch.modules.prediction.generation.base import RNNDecoderBase
from graph4nlp.pytorch.modules.utils.vocab_utils import Vocab


def extract_mask(mask, token):
    mask_ret = torch.zeros(*(mask.shape)).to(mask.device)
    mask_ret.fill_(0)
    mask_ret[mask == token] = 1
    return mask_ret


class StdRNNDecoder(RNNDecoderBase):
    """
        The standard rnn for sequence decoder.
    Parameters
    ----------
    max_decoder_step: int
        The maximal decoding step.
    input_size: int
        The dimension for standard rnn decoder's input.
    hidden_size: int
        The dimension for standard rnn decoder's hidden representation during calculation.
    word_emb: torch.nn.Embedding
        The target's embedding matrix.
    vocab: Any
        The target's vocabulary
    rnn_type: str, option=["lstm", "gru"], default="lstm"
        The rnn's type. We support ``lstm`` and ``gru`` here.
    use_attention: bool, default=True
        Whether use attention during decoding.
    attention_type: str, option=["uniform", "sep_diff_encoder_type", sep_diff_node_type], default="uniform" # noqa
        The attention strategy choice.
        "``uniform``": uniform attention. We will attend on the nodes uniformly.
        "``sep_diff_encoder_type``": separate attention.
            We will attend on graph encoder and rnn encoder's results separately.
        "``sep_diff_node_type``": separate attention.
            We will attend on different node type separately.

    attention_function: str, option=["general", "mlp"], default="mlp"
        Different attention function.
    node_type_num: int, default=None
        When we choose "``sep_diff_node_type``", we must set this parameter.
        This parameter indicate the the amount of node type.
    fuse_strategy: str, option=["average", "concatenate"], default=average
        The strategy to fuse attention results generated by separate attention.
        "``average``": We will take an average on all results.
        "``concatenate``": We will concatenate all results to one.
    use_copy: bool, default=False
        Whether use ``copy`` mechanism. See pointer network. Note that you must use attention first.
    use_coverage: bool, default=False
        Whether use ``coverage`` mechanism. Note that you must use attention first.
    coverage_strategy: str, option=["sum", "max"], default="sum"
        The coverage strategy when calculating the coverage vector.
    tgt_emb_as_output_layer: bool, default=False
        When this option is set ``True``, the output projection layer(It is used to project RNN encoded # noqa
        representation to target sequence)'s weight will be shared with the target vocabulary's embedding. # noqa
    dropout: float, default=0.3
    """

    def __init__(
        self,
        max_decoder_step,
        input_size,
        hidden_size,  # decoder config
        word_emb,
        vocab: Vocab,  # word embedding & vocabulary TODO: add our vocabulary when building pipeline
        rnn_type="lstm",
        graph_pooling_strategy=None,  # RNN config
        use_attention=True,
        attention_type="uniform",
        rnn_emb_input_size=None,  # attention config
        attention_function="mlp",
        node_type_num=None,
        fuse_strategy="average",
        use_copy=False,
        use_coverage=False,
        coverage_strategy="sum",
        tgt_emb_as_output_layer=False,  # share label projection with word embedding
        dropout=0.3,
    ):
        super(StdRNNDecoder, self).__init__(
            use_attention=use_attention,
            use_copy=use_copy,
            use_coverage=use_coverage,
            attention_type=attention_type,
            fuse_strategy=fuse_strategy,
        )
        self.max_decoder_step = max_decoder_step
        self.word_emb_size = word_emb.embedding_dim
        self.decoder_input_size = input_size
        self.graph_pooling_strategy = graph_pooling_strategy
        self.dropout = nn.Dropout(p=dropout)
        self.tgt_emb = word_emb
        self.tgt_emb_as_output_layer = tgt_emb_as_output_layer # true
        if self.tgt_emb_as_output_layer:
            self.input_feed_size = self.word_emb_size # 300
        else:
            self.input_feed_size = hidden_size

        self.rnn = self._build_rnn(
            rnn_type=rnn_type,
            input_size=self.word_emb_size + self.input_feed_size,
            hidden_size=hidden_size,
        )
        self.decoder_hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = 1
        self.out_logits_size = self.decoder_hidden_size

        # attention builder
        if self.use_attention:
            if self.rnn_type == "lstm":
                query_size = 2 * self.decoder_hidden_size
            elif self.rnn_type == "gru":
                query_size = self.decoder_hidden_size
            else:
                raise NotImplementedError()
            if attention_type == "uniform":
                self.enc_attention = Attention(
                    hidden_size=self.decoder_hidden_size,
                    query_size=query_size,
                    memory_size=input_size,
                    has_bias=True,
                    attention_funtion=attention_function,
                )
                self.out_logits_size += self.decoder_input_size
            elif attention_type == "sep_diff_encoder_type":
                assert isinstance(rnn_emb_input_size, int)
                self.rnn_emb_input_size = rnn_emb_input_size
                self.enc_attention = Attention(
                    hidden_size=self.decoder_hidden_size,
                    query_size=query_size,
                    memory_size=input_size,
                    has_bias=True,
                    attention_funtion=attention_function,
                )
                self.out_logits_size += self.decoder_input_size
                self.rnn_attention = Attention(
                    hidden_size=self.decoder_hidden_size,
                    query_size=query_size,
                    memory_size=rnn_emb_input_size,
                    has_bias=True,
                    attention_funtion=attention_function,
                )
                if self.fuse_strategy == "concatenate":
                    self.out_logits_size += self.rnn_emb_input_size
                else:
                    if rnn_emb_input_size != input_size:
                        raise ValueError(
                            "input RNN embedding size is not equal to graph embedding size"
                        )
            elif attention_type == "sep_diff_node_type":
                assert node_type_num >= 1
                attn_modules = [
                    Attention(
                        hidden_size=self.decoder_hidden_size,
                        query_size=query_size,
                        memory_size=input_size,
                        has_bias=True,
                        attention_funtion=attention_function,
                    )
                    for _ in range(node_type_num)
                ]
                self.node_type_num = node_type_num
                self.attn_modules = nn.ModuleList(attn_modules)
                if self.fuse_strategy == "concatenate":
                    self.out_logits_size += self.decoder_input_size * self.node_type_num
                elif self.fuse_strategy == "average":
                    self.out_logits_size += self.decoder_input_size
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

        self.attention_type = attention_type

        if self.rnn_type == "lstm":
            self.encoder_decoder_adapter = nn.ModuleList(
                [nn.Linear(self.decoder_input_size, self.decoder_hidden_size) for _ in range(2)]
            )
        elif self.rnn_type == "gru":
            self.encoder_decoder_adapter = nn.Linear(
                self.decoder_input_size, self.decoder_hidden_size
            )
        else:
            raise NotImplementedError()

        # input_feed

        self.memory_to_feed = nn.Sequential(
            nn.Linear(self.out_logits_size, self.input_feed_size), nn.ReLU(), nn.Dropout(dropout)
        )

        # project logits to labels
        # self.tgt_emb_as_output_layer = tgt_emb_as_output_layer
        if self.tgt_emb_as_output_layer:  # use pre_out layer
            self.out_embed_size = self.word_emb_size
            self.pre_out = nn.Linear(self.input_feed_size, self.out_embed_size)
            size_before_output = self.out_embed_size
        else:  # don't use pre_out layer
            size_before_output = self.input_feed_size

        # size_before_output = self.input_feed_size
        self.vocab = vocab
        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        self.out_project = nn.Linear(size_before_output, vocab_size, bias=False)
        if self.tgt_emb_as_output_layer:
            # self.out_project.weight = self.tgt_emb.word_emb_layer.weight
            self.out_project.weight = self.tgt_emb.weight

        # coverage strategy
        if self.use_coverage:
            if not self.use_attention:
                raise ValueError("You should use attention when you use coverage strategy.")

            self.coverage_strategy = coverage_strategy

            self.coverage_weight = torch.Tensor(1, 1, self.decoder_hidden_size)
            self.coverage_weight = nn.Parameter(nn.init.xavier_uniform_(self.coverage_weight))

        # copy: pointer network
        if self.use_copy:
            ptr_size = self.input_feed_size + self.word_emb_size
            if self.rnn_type == "lstm":
                ptr_size += self.decoder_hidden_size * 2
            elif self.rnn_type == "gru":
                ptr_size += self.decoder_hidden_size
            else:
                raise NotImplementedError()
            if self.use_attention:
                if self.attention_type == "uniform":
                    ptr_size += input_size
                elif self.attention_type == "sep_diff_encoder_type":
                    ptr_size += input_size + rnn_emb_input_size
                elif self.attention_type == "sep_diff_node_type":
                    ptr_size += input_size * node_type_num
            self.ptr = nn.Linear(ptr_size, 1)

    def _build_rnn(self, rnn_type, **kwargs):
        """
            The rnn factory.
        Parameters
        ----------
        rnn_type: str, option=["lstm", "gru"], default="lstm"
            The rnn type.
        """
        if rnn_type == "lstm":
            return nn.LSTM(**kwargs)
        elif rnn_type == "gru":
            return nn.GRU(**kwargs)
        else:
            raise NotImplementedError("RNN type: {} is not supported.".format(rnn_type))

    def coverage_function(self, enc_attn_weights):
        if self.coverage_strategy == "max":
            coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
        elif self.coverage_strategy == "sum":
            coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
        else:
            raise ValueError("Unrecognized cover_func: " + self.cover_func)
        return coverage_vector

    def _run_forward_pass(
        self,
        graph_node_embedding,
        graph_node_mask=None,
        rnn_node_embedding=None,
        graph_level_embedding=None,
        graph_edge_embedding=None,
        graph_edge_mask=None,
        tgt_seq=None,
        src_seq=None,
        oov_dict=None,
        teacher_forcing_rate=1.0,
    ):
        """
            The forward function for RNN.
        Parameters
        ----------
        graph_node_embedding: torch.Tensor
            shape=[B, N, D]
        graph_node_mask: torch.Tensor
            shape=[B, N]
            -1 indicating dummy node. 0-``node_type_num`` are valid node type.
        rnn_node_embedding: torch.Tensor
            shape=[B, N, D]
        graph_level_embedding: torch.Tensor
            shape=[B, D]
        graph_edge_embedding: torch.Tensor
            shape=[B, E, D]
            Not implemented yet.
        graph_edge_mask: torch.Tensor
            shape=[B, E]
            Not implemented yet.
        tgt_seq: torch.Tensor
            shape=[B, T]
            The target sequence's index.
        src_seq: torch.Tensor
            shape=[B, S]
            The source sequence's index. It is used for ``use_copy``.
            Note that it can be encoded by target word embedding.
        oov_dict: Vocab
        teacher_forcing_rate: float, default=1.0
            The teacher forcing rate.

        Returns
        -------
        logits: torch.Tensor
            shape=[B, tgt_len, vocab_size]
            The probability for predicted target sequence. It is processed by softmax function.
        enc_attn_weights_average: torch.Tensor
            It is used for calculating coverage loss.
            The averaged attention scores.
        coverage_vectors: torch.Tensor
            It is used for calculating coverage loss.
            The coverage vector.
        """
        #print(graph_node_embedding.size())
        #print(graph_node_mask.size())
        #print(tgt_seq.size())
        #print(src_seq.size())

        target_len = self.max_decoder_step # 100
        if tgt_seq is not None:
            target_len = tgt_seq.shape[1] # 修改为tgt seq len

        batch_size = graph_node_embedding.shape[0]
        decoder_input = torch.tensor([self.vocab.SOS] * batch_size).to(graph_node_embedding.device) # gcn更新后的node embedding作为输入，大小[4]

        decoder_state = self.get_decoder_init_state( # 初始化权重参数
            rnn_type=self.rnn_type, batch_size=batch_size, content=graph_level_embedding
        )

        input_feed = torch.zeros(batch_size, self.input_feed_size).to(graph_node_embedding.device) # [batch size, word emb size]

        outputs = []
        enc_attn_weights_average = []
        coverage_vectors = []

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
            if self.use_coverage:
                enc_attn_weights_average.append(dec_attn_scores.unsqueeze(0))
                coverage_vectors.append(coverage_vec)

            outputs.append(decoder_output.unsqueeze(1))

            # teacher_forcing
            if tgt_seq is not None and random.random() < teacher_forcing_rate: # 每次都不用上一步解码结果而用ground truth作为当前解码输入
                decoder_input = tgt_seq[:, i]
            else:
                # sampling
                # TODO: now argmax sampling
                decoder_input = decoder_output.squeeze(1).argmax(dim=-1)
            decoder_input = self._filter_oov(decoder_input)
        ret = torch.cat(outputs, dim=1)

        return ret, enc_attn_weights_average, coverage_vectors

    def decode_step(
        self,
        decoder_input,
        input_feed,
        rnn_state,
        encoder_out,
        dec_input_mask,
        rnn_emb=None,
        enc_attn_weights_average=None,
        src_seq=None,
        oov_dict=None,
    ):
        batch_size = decoder_input.shape[0]
        dec_emb = self.tgt_emb(decoder_input) # 由out word vocab构建的word embedding，大小[batch size, word emb size]
        dec_emb = torch.cat((dec_emb, input_feed), dim=1) # 大小[batch size, word emb sie+input feed size=2*word emb size]
        dec_emb = self.dropout(dec_emb)
        if self.use_coverage and enc_attn_weights_average: # true
            coverage_vec = self.coverage_function(enc_attn_weights_average) # 简单求和
        else:
            coverage_vec = None

        dec_out, rnn_state = self.rnn(dec_emb.unsqueeze(0), rnn_state) # 上一步decoder_state经过rnn得到dec_out
        dec_out = dec_out.squeeze(0) # [batch size, hidden size]

        if self.rnn_type == "lstm":
            rnn_state = tuple([self.dropout(x) for x in rnn_state])
            hidden = torch.cat(rnn_state, -1).squeeze(0) # [batch size, 2*hidden size]
        elif self.rnn_type == "gru":
            rnn_state = self.dropout(rnn_state)
            hidden = rnn_state.squeeze(0)
        else:
            raise NotImplementedError()
        attn_collect = []
        score_collect = []

        if self.use_attention:
            if self.use_coverage and coverage_vec is not None:
                coverage_repr = coverage_vec
            else:
                coverage_repr = None
            if coverage_repr is not None:
                coverage_repr = coverage_repr.unsqueeze(-1) * self.coverage_weight
            if self.attention_type == "uniform" or self.attention_type == "sep_diff_encoder_type":
                enc_mask = extract_mask(dec_input_mask, token=-1)
                enc_mask = 1 - enc_mask

                attn_res, scores = self.enc_attention( # 对gcn更新后的node embedding使用attention，加入coverage
                    query=hidden, memory=encoder_out, memory_mask=enc_mask, coverage=coverage_repr
                )
                attn_collect.append(attn_res)
                score_collect.append(scores)
                if self.attention_type == "sep_diff_encoder_type":
                    rnn_attn_res, rnn_scores = self.rnn_attention( # 对rnn_emb使用attention，加入coverage
                        query=hidden, memory=rnn_emb, memory_mask=enc_mask, coverage=coverage_repr
                    )
                    score_collect.append(rnn_scores)
                    attn_collect.append(rnn_attn_res)
            elif self.attention_type == "sep_diff_node_type":
                for i in range(self.node_type_num):
                    node_mask = extract_mask(dec_input_mask, token=i)
                    attn, scores = self.attn_modules[i](
                        query=hidden,
                        memory=encoder_out,
                        memory_mask=node_mask,
                        coverage=coverage_repr,
                    )
                    attn_collect.append(attn)
                    score_collect.append(scores)

        if self.use_attention:
            if self.attention_type == "uniform":
                assert len(attn_collect) == 1
                assert len(score_collect) == 1
                attn_total = attn_collect[0]
            elif (
                self.attention_type == "sep_diff_encoder_type"
                or self.attention_type == "sep_diff_node_type"
            ):
                if self.fuse_strategy == "average":
                    attn_total = reduce(lambda x, y: x + y, attn_collect) / len(attn_collect)
                elif self.fuse_strategy == "concatenate":
                    attn_total = torch.cat(attn_collect, dim=-1) # 对node_emb和rnn_emb的注意力结果进行融合
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            decoder_output = torch.cat((dec_out, attn_total), dim=-1)
            dec_attn_scores = reduce(lambda x, y: x + y, score_collect) / len(score_collect) # 对注意力分数求和平均
        else:
            decoder_output = dec_out

        out = self.memory_to_feed(decoder_output)

        # project
        if self.tgt_emb_as_output_layer:
            out_embed = torch.tanh(self.pre_out(out))
            # out_embed = torch.tanh(out)
            out_embed = self.dropout(out_embed)
        else:
            out_embed = out

        decoder_output = self.out_project(out_embed)  # [B, S, Vocab]

        if self.use_copy:
            assert src_seq is not None
            assert oov_dict is not None
            # output = torch.zeros(batch_size, oov_dict.get_vocab_size()).to(decoder_input.device)
            attn_ptr = torch.cat(attn_collect, dim=-1) # 综合注意力向量c
            pgen_collect = [dec_emb, hidden, attn_ptr]

            prob_ptr = torch.sigmoid(self.ptr(torch.cat(pgen_collect, -1)))
            prob_gen = 1 - prob_ptr
            gen_output = torch.softmax(decoder_output, dim=-1)
            #print(decoder_output)
            ret = prob_gen * gen_output
            need_pad_length = oov_dict.get_vocab_size() - self.vocab.get_vocab_size()
            output = torch.cat((ret, ret.new_zeros((batch_size, need_pad_length))), dim=1)
            # output[:, :self.vocab.get_vocab_size()] = ret

            ptr_output = dec_attn_scores
            # p_gen * softmax(decoder_output) + (1 - p_gen) *
            output.scatter_add_(1, src_seq, prob_ptr * ptr_output) # 加法
            decoder_output = output
        else:
            decoder_output = torch.softmax(decoder_output, dim=-1)

        return decoder_output, out, rnn_state, dec_attn_scores, coverage_vec

    def get_decoder_init_state(self, rnn_type, batch_size, content=None):
        if rnn_type == "lstm":
            if content is not None:
                assert len(content.shape) == 2
                assert content.shape[0] == batch_size
                ret = tuple(
                    [
                        self.encoder_decoder_adapter[i](content)
                        .view(1, batch_size, self.decoder_hidden_size)
                        .expand(self.num_layers, -1, -1)
                        for i in range(2)
                    ]
                )
            else:
                weight = next(self.parameters()).data
                ret = (
                    weight.new(self.num_layers, batch_size, self.decoder_hidden_size).zero_(),
                    weight.new(self.num_layers, batch_size, self.decoder_hidden_size).zero_(),
                )
        elif rnn_type == "gru":
            if content is not None:
                ret = (
                    self.encoder_decoder_adapter(content)
                    .view(1, batch_size, self.decoder_hidden_size)
                    .expand(self.num_layers, -1, -1)
                )
            else:
                weight = next(self.parameters()).data
                ret = weight.new(self.num_layers, batch_size, self.decoder_hidden_size).zero_()
        else:
            raise NotImplementedError()
        return ret

    def _filter_oov(self, tokens):
        ret = tokens.clone()
        ret[tokens >= self.vocab_size] = self.vocab.UNK
        return ret

    def forward(self, batch_graph, tgt_seq=None, oov_dict=None, teacher_forcing_rate=1.0):
        """
            The forward function of ``StdRNNDecoder``
        Parameters
        ----------
        batch_graph: GraphData
            The graph input
        tgt_seq: torch.Tensor
            shape=[B, T]
            The target sequence's index.
        oov_dict: VocabModel, default=None
            The vocabulary for copy mechanism.
        teacher_forcing_rate: float, default=1.0
            The teacher forcing rate.

        Returns
        -------
        logits: torch.Tensor
            shape=[B, tgt_len, vocab_size]
            The probability for predicted target sequence. It is processed by softmax function.
        enc_attn_weights_average: torch.Tensor
            It is used for calculating coverage loss.
            The averaged attention scores.
        coverage_vectors: torch.Tensor
            It is used for calculating coverage loss.
            The coverage vector.
        """
        params = self.extract_params(batch_graph)
        params["tgt_seq"] = tgt_seq
        params["teacher_forcing_rate"] = teacher_forcing_rate
        params["oov_dict"] = oov_dict
        return self._run_forward_pass(**params)

    def extract_params(self, batch_graph):
        """
            Extract parameters from ``batch_graph`` for _run_forward_pass() function.
        Parameters
        ----------
        batch_graph: GraphData

        Returns
        -------
        params: dict
        """
        batch_data_dict = batch_graph.batch_node_features
        graph_node_emb = batch_data_dict["node_emb"] # GCN更新后的node embedding

        rnn_node_emb = batch_data_dict["rnn_emb"] # BiLSTM提取的信息
        if len(batch_data_dict["token_id"].shape) == 3: #[4,400,1]
            graph_node_mask = (torch.sum(batch_data_dict["token_id"], dim=-1) != 0).squeeze( # [4,400]
                -1
            ).float() - 1
        else:
            graph_node_mask = (batch_data_dict["token_id"] != 0).squeeze(-1).float() - 1
        if self.use_copy:
            src_seq_ret = batch_graph.batch_node_features["token_id_oov"]
        else:
            src_seq_ret = None

        graph_level_emb = self.graph_pooling(graph_node_emb) # none，整个图级别的embedding，说不定对解码过程有益

        return {
            "graph_node_embedding": graph_node_emb, # gcn更新后的node embedding
            "graph_node_mask": graph_node_mask, # 压缩后的node_features["token_id"]，大小[batch size, node num]
            "rnn_node_embedding": rnn_node_emb, # 一开始BiLSTM提取的信息
            "graph_level_embedding": graph_level_emb, # none
            "graph_edge_embedding": None,
            "graph_edge_mask": None,
            "src_seq": src_seq_ret.long() if self.use_copy else None, # node_features["token_id_oov"]
        }

    def graph_pooling(self, graph_node):
        if self.graph_pooling_strategy is None: # true
            pooled_vec = None
        elif self.graph_pooling_strategy == "mean":
            pooled_vec = torch.mean(graph_node, dim=1)
        elif self.graph_pooling_strategy == "max":
            pooled_vec, _ = torch.max(graph_node, dim=1)
        elif self.graph_pooling_strategy == "min":
            pooled_vec, _ = torch.mean(graph_node, dim=1)
        else:
            raise NotImplementedError()
        return pooled_vec