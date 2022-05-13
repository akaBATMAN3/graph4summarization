import json

from ...data.data import GraphData
from .base import StaticGraphConstructionBase


class IEBasedGraphConstruction(StaticGraphConstructionBase):
    """
        Information Extraction based graph construction class

    Parameters
    ----------
    embedding_style: dict
        Specify embedding styles including ``single_token_item``,
        ``emb_strategy``, ``num_rnn_layers``, ``bert_model_name``
        and ``bert_lower_case``.
    vocab: VocabModel
        Vocabulary including all words appeared in graphs.
    """

    def __init__(self, vocab):
        super(IEBasedGraphConstruction, self).__init__()
        self.vocab = vocab
        self.verbose = 1

    def add_vocab(self, g):
        """
            Add node tokens appeared in graph g to vocabulary.

        Parameters
        ----------
        g: GraphData
            Graph data-structure.

        """
        for i in range(g.get_node_num()):
            attr = g.get_node_attrs(i)[i]
            self.vocab.word_vocab._add_words([attr["token"]])

    @classmethod
    def parsing(cls, all_sent_triples_list, edge_strategy):
        """
        Parameters
        ----------
        all_sent_triples_list: list
        edge_strategy: str

        Returns
        -------
        parsed_results: dict
            parsed_results is an intermediate dict that contains all the information of
            the constructed IE graph for a piece of raw text input.

            `parsed_results['graph_content']` is a list of dict.

            Each dict in `parsed_results['graph_content']` contains information about a
            triple (src_ent, rel, tgt_ent).

            `parsed_results['graph_nodes']` contains all nodes in the KG graph.

            `parsed_results['node_num']` is the number of nodes in the KG graph.
        """

        parsed_results = {}
        parsed_results["graph_content"] = []
        graph_nodes = []
        for triple in all_sent_triples_list:
            if edge_strategy is None:
                if triple[0] not in graph_nodes:
                    graph_nodes.append(triple[0])

                if triple[2] not in graph_nodes:
                    graph_nodes.append(triple[2])

                triple_info = {
                    "edge_tokens": triple[1],
                    "src": {"tokens": triple[0], "id": graph_nodes.index(triple[0])},
                    "tgt": {"tokens": triple[2], "id": graph_nodes.index(triple[2])},
                }
                if triple_info not in parsed_results["graph_content"]:
                    parsed_results["graph_content"].append(triple_info)
            elif edge_strategy == "as_node":
                if triple[0] not in graph_nodes:
                    graph_nodes.append(triple[0])

                if triple[1] not in graph_nodes: # 边也看作结点
                    graph_nodes.append(triple[1])

                if triple[2] not in graph_nodes:
                    graph_nodes.append(triple[2])

                triple_info_0_1 = {
                    "edge_tokens": [],
                    "src": {
                        "tokens": triple[0],
                        "id": graph_nodes.index(triple[0]),
                        "type": 0,  # 'ent_node'
                    },
                    "tgt": {
                        "tokens": triple[1],
                        "id": graph_nodes.index(triple[1]),
                        "type": 3,  # 'edge_node'
                    },
                }

                triple_info_1_2 = {
                    "edge_tokens": [],
                    "src": {
                        "tokens": triple[1],
                        "id": graph_nodes.index(triple[1]),
                        "type": 3,  # 'edge_node'
                    },
                    "tgt": {
                        "tokens": triple[2],
                        "id": graph_nodes.index(triple[2]),
                        "type": 0,  # 'ent_node'
                    },
                }

                if triple_info_0_1 not in parsed_results["graph_content"]:
                    parsed_results["graph_content"].append(triple_info_0_1)
                if triple_info_1_2 not in parsed_results["graph_content"]:
                    parsed_results["graph_content"].append(triple_info_1_2)
                #print(parsed_results["graph_content"])
            else:
                raise NotImplementedError(
                    "Not Implemented Edge Strategy: {}.".format(edge_strategy)
                )

        parsed_results["node_num"] = len(graph_nodes)
        parsed_results["graph_nodes"] = graph_nodes

        return parsed_results

    @classmethod
    def static_topology(
        cls,
        raw_text_data,
        nlp_processor,
        processor_args,
        merge_strategy, # null
        edge_strategy, # as_node
        verbose=True, # False
    ):
        """
            Graph building method.

        Parameters
        ----------
        raw_text_data: str
            Raw text data, it can be multi-sentences.

        nlp_processor: StanfordCoreNLP
            NLP parsing tools

        merge_strategy: None or str, option=[None, "global", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``:  Do not add additional nodes and edges.

            ``global``: All subjects in extracted triples are connected by a "GLOBAL_NODE" # like supernode?
                        using a "global" edge

            ``"user_define"``: We will give this option to the user.
            User can override this method to define your merge strategy.

        edge_strategy: None or str, option=[None, "as_node"]
            Strategy to process edge.
            ``None``: It will be the default option.
                      Edge information will be preserved in GraphDate.edge_attributes.
            ``as_node``: We will view the edge as a graph node.
                         If there is an edge whose type is ``k`` between node ``i`` and node ``j``,
                         we will insert a node ``k`` into the graph and link node
                         (``i``, ``k``) and (``k``, ``j``).
                         The ``type`` of original nodes will be set as ``ent_node``,
                         while the ``type`` of edge nodes is ``edge_node`.`

        Returns
        -------
        graph: GraphData
            The merged graph data-structure.
        """
        cls.verbose = verbose

        if isinstance(processor_args, list):
            props_coref = processor_args[0]
            props_openie = processor_args[1]
        else:
            raise RuntimeError(
                "processor_args for IEBasedGraphConstruction shouble be a list of dict."
            )

        # Do coreference resolution on the whole 'raw_text_data'
        coref_json = nlp_processor.annotate(raw_text_data.strip(), properties=props_coref)
        from .utils import CORENLP_TIMEOUT_SIGNATURE

        if CORENLP_TIMEOUT_SIGNATURE in coref_json:
            raise TimeoutError(
                "Coref-CoreNLP timed out at input: \n{}\n This item will be skipped. "
                "Please check the input or change the timeout threshold.".format(raw_text_data)
            )

        coref_dict = json.loads(coref_json)

        # Extract and preserve necessary parsing results from coref_dict['sentences']
        # sent_dict['tokenWords']: list of tokens in a sentence
        sentences = []
        for sent in coref_dict["sentences"]: # 对article分句后对每句句子处理
            #print(sentences)
            #print(sent)
            sent_dict = {}
            sent_dict["sentNum"] = sent["index"]  # start from 0 # 句子索引
            sent_dict["tokens"] = sent["tokens"] # 从1开始，所有token的信息组成的list
            sent_dict["tokenWords"] = [token["word"] for token in sent["tokens"]] # token所代表的单词组成的list
            sent_dict["sentText"] = " ".join(sent_dict["tokenWords"]) # 所有单词用空格分隔组成句子
            sentences.append(sent_dict) # 句子list
            #print(sent_dict["tokenWords"])
        #print(sentences)

        for _, v in coref_dict["corefs"].items(): # 对article中所有共指处理
            # v is a list of dict, each dict contains a str
            # v[0] contains 'original entity str'
            # v[1:] contain 'pron strs' refers to 'original entity str'
            #print(v)
            ent_text = v[0]["text"]  # 'original entity str' # 获取第一个共指实体
            if "," in ent_text:
                # cut the 'original entity str' if it is too long
                ent_text = ent_text.split(",")[0].strip()

            for pron in v[1:]: # 对第一个共指实体之后的实体处理
                pron_text = pron["text"]  # 'pron strs'
                if ent_text == pron_text or v[0]["text"] == pron_text: # 一模一样的实体不做处理
                    continue
                pron_sentNum = pron["sentNum"] - 1  # the sentNum 'pron str' appears in
                pron_startIndex = pron["startIndex"] - 1
                pron_endIndex = pron["endIndex"] - 1

                # replace 'pron str' with 'original entity str'
                sentences[pron_sentNum]["tokenWords"][pron_startIndex] = ent_text # 所有第一次出现的实体后的实体都被替换为第一个实体
                for rm_idx in range(pron_startIndex + 1, pron_endIndex):
                    sentences[pron_sentNum]["tokenWords"][rm_idx] = ""

        # build resolved text
        for sent_id, _ in enumerate(sentences):
            sentences[sent_id]["tokenWords"] = list(
                filter(lambda a: a != "", sentences[sent_id]["tokenWords"])
            )
            sentences[sent_id]["resolvedText"] = " ".join(sentences[sent_id]["tokenWords"])
        #至此，所有第二次及以后出现的实体都被第一次出现的实体替换
        # use OpenIE to extract triples from resolvedText
        all_sent_triples = {}
        for sent in sentences:
            resolved_sent = sent["resolvedText"]
            #print(resolved_sent)
            openie_json = nlp_processor.annotate(resolved_sent.strip(), properties=props_openie)
            if CORENLP_TIMEOUT_SIGNATURE in openie_json:
                raise TimeoutError(
                    "OpenIE-CoreNLP timed out at input: \n{}\n This item will be skipped. "
                    "Please check the input or change the timeout threshold.".format(raw_text_data)
                )
            openie_dict = json.loads(openie_json)
            #print(openie_dict)
            for triple_dict in openie_dict["sentences"][0]["openie"]:
                #print(triple_dict)
                sbj = triple_dict["subject"]
                rel = triple_dict["relation"]
                if rel in ["was", "is", "were", "are"]:
                    continue
                obj = triple_dict["object"]

                # If two triples have the same subject and relation,
                # only preserve the one has longer object
                if sbj + "<TSEP>" + rel not in all_sent_triples.keys():
                    all_sent_triples[sbj + "<TSEP>" + rel] = [sbj, rel, obj]
                else:
                    if len(obj) > len(all_sent_triples[sbj + "<TSEP>" + rel][2]):
                        all_sent_triples[sbj + "<TSEP>" + rel] = [sbj, rel, obj]

        all_sent_triples_list = list( # 所有triples[sbj, rel, obj]组成的list
            all_sent_triples.values()
        )  # triples extracted from all sentences

        # remove similar triples
        triples_rm_list = []
        for i, lst_i in enumerate(all_sent_triples_list[:-1]):
            for lst_j in all_sent_triples_list[i + 1 :]:
                str_i = " ".join(lst_i)
                str_j = " ".join(lst_j)
                if (
                    str_i in str_j
                    or str_j in str_i
                    or lst_i[0] + lst_i[2] == lst_j[0] + lst_j[2]
                    or lst_i[1] + lst_i[2] == lst_j[1] + lst_j[2]
                ):
                    if len(lst_i[1]) > len(lst_j[1]):
                        triples_rm_list.append(lst_j)
                    else:
                        triples_rm_list.append(lst_i)

        for lst in triples_rm_list:
            if lst in all_sent_triples_list:
                all_sent_triples_list.remove(lst)

        global_triples = cls._graph_connect(all_sent_triples_list, merge_strategy) # 未采用连接所有实体
        all_sent_triples_list.extend(global_triples) # 若采用global则添加[entitie, global, GLOBAL_NODE]的三元组
        # 算是最终IE解析结果，实质就是强化后（合并共指实体；去掉相似三元组，这一步也是去冗余；）的[sub,rel,obj]
        parsed_results = cls.parsing(all_sent_triples_list, edge_strategy)

        graph = cls._construct_static_graph(parsed_results, edge_strategy=edge_strategy) # 为整个article构建图结构

        if cls.verbose:
            for info in parsed_results["graph_content"]:
                print(info)

        return graph

    @classmethod
    def _construct_static_graph(cls, parsed_object, edge_strategy=None):
        """

        Parameters
        ----------
        parsed_object: dict
            ``parsed_object`` contains all triples extracted from the raw_text_data.

        edge_strategy: None or str, option=[None, "as_node"]
            Strategy to process edge.
            ``None``: It will be the default option.
                      Edge information will be preserved in GraphDate.edge_attributes.
            ``as_node``: We will view the edge as a graph node.
                         If there is an edge whose type is ``k`` between node ``i`` and node ``j``,
                         we will insert a node ``k`` into the graph and link node
                         (``i``, ``k``) and (``k``, ``j``).
                         The ``type`` of original nodes will be set as ``ent_node``,
                         while the ``type`` of edge nodes is ``edge_node`.`

        Returns
        -------
        graph: GraphData
            graph structure for single sentence
        """
        ret_graph = GraphData() # 定义空图
        node_num = parsed_object["node_num"]
        if node_num == 0:
            raise RuntimeError(
                '"The number of nodes to be added should be greater than 0. (Got {})"'.format(
                    node_num
                )
            )
        ret_graph.add_nodes(node_num)
        for triple_info in parsed_object["graph_content"]:
            if edge_strategy is None:
                ret_graph.add_edge(triple_info["src"]["id"], triple_info["tgt"]["id"])
                eids = ret_graph.edge_ids(triple_info["src"]["id"], triple_info["tgt"]["id"])
                for eid in eids:
                    ret_graph.edge_attributes[eid]["token"] = triple_info["edge_tokens"]
            elif edge_strategy == "as_node":
                ret_graph.add_edge(triple_info["src"]["id"], triple_info["tgt"]["id"]) # sub连向rel，rel连向obj
            else:
                raise NotImplementedError()

            ret_graph.node_attributes[triple_info["src"]["id"]]["token"] = triple_info["src"][
                "tokens" # 所代表单词
            ]
            ret_graph.node_attributes[triple_info["tgt"]["id"]]["token"] = triple_info["tgt"][
                "tokens"
            ]
            if edge_strategy == "as_node":
                ret_graph.node_attributes[triple_info["src"]["id"]]["type"] = triple_info["src"][
                    "type" # type为0代表sub/obj
                ]
                ret_graph.node_attributes[triple_info["tgt"]["id"]]["type"] = triple_info["tgt"][
                    "type" # type为3代表edge
                ]

        return ret_graph

    @classmethod
    def _graph_connect(cls, triple_list, merge_strategy=None):
        """
            This method will connect entities in the ``triple_list`` to ensure the graph
            is connected.

        Parameters
        ----------
        triple_list: list of [subject, relation, object]
            A list of all triples extracted from ``raw_text_data`` using coref and openie.

        merge_strategy: None or str, option=[None, "global", "user_define"]
            Strategy to merge sub-graphs into one graph
            ``None``:  Do not add additional nodes and edges.

            ``global``: All subjects in extracted triples are connected by a "GLOBAL_NODE"
                        using a "global" edge

            ``"user_define"``: We will give this option to the user.
            User can override this method to define your merge strategy.

        Returns
        -------
        global_triples: list of [subject, relation, object]
            The added triples using merge_strategy.
        """

        if merge_strategy == "global":
            graph_nodes = []
            global_triples = []
            for triple in triple_list:
                if triple[0] not in graph_nodes:
                    graph_nodes.append(triple[0])
                    global_triples.append([triple[0], "global", "GLOBAL_NODE"]) # 只需要和subject相连就能和object相连

                if triple[2] not in graph_nodes:
                    graph_nodes.append(triple[2])

            return global_triples
        elif merge_strategy is None:
            return []
        else:
            raise NotImplementedError("Not Implemented Merge Strategy: {}.".format(merge_strategy))

    def forward(self, batch_graphdata: list):
        raise RuntimeError("This interface is removed.")
