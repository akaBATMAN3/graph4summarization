import json
import torch
from nltk.tokenize import word_tokenize

from graph4nlp.pytorch.data.dataset import Text2TextDataItem, Text2TextDataset
from graph4nlp.pytorch.modules.utils.padding_utils import pad_2d_vals, pad_2d_vals_no_size


class CNNDataset(Text2TextDataset):
    def __init__(
        self,
        root_dir,
        topology_subdir,
        graph_name,
        static_or_dynamic="static",
        tokenizer=word_tokenize,
        lower_case=True,
        pretrained_word_emb_name="840B",
        pretrained_word_emb_url=None,
        target_pretrained_word_emb_name=None,
        target_pretrained_word_emb_url=None,
        pretrained_word_emb_cache_dir=".vector_cache/",
        use_val_for_vocab=False,
        seed=1234,
        thread_number=4,
        port=9000,
        timeout=15000,
        edge_strategy=None,
        share_vocab=True,
        word_emb_size=300,
        dynamic_init_graph_name=None,
        dynamic_init_topology_builder=None,
        dynamic_init_topology_aux_args=None,
        for_inference=False,
        reused_vocab_model=None,
        **kwargs
    ):
        super(CNNDataset, self).__init__(
            root_dir=root_dir,
            topology_subdir=topology_subdir,
            graph_name=graph_name,
            static_or_dynamic=static_or_dynamic, # static
            tokenizer=tokenizer,
            lower_case=lower_case,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            target_pretrained_word_emb_name=target_pretrained_word_emb_name,
            target_pretrained_word_emb_url=target_pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            use_val_for_vocab=use_val_for_vocab,
            seed=seed,
            thread_number=thread_number,
            port=port,
            timeout=timeout,
            edge_strategy=edge_strategy,
            share_vocab=share_vocab,
            word_emb_size=word_emb_size,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            for_inference=for_inference,
            reused_vocab_model=reused_vocab_model,
            **kwargs
        )

    @property
    def raw_file_names(self):
        """
        3 reserved keys: 'train', 'val' (optional), 'test'.
        Represent the split of dataset.
        """
        return {"train": "train_1w.json", "val": "val.json", "test": "test.json"}

    @property
    def processed_file_names(self):
        """At least 2 reserved keys should be fiiled: 'vocab' and 'data'."""
        return {"vocab": "vocab.pt", "data": "data.pt"}

    def download(self):
        return

    def parse_file(self, file_path):
        """
        Read and parse the file specified by `file_path`. The file format is
        specified by each individual task-specific base class. Returns all
        the indices of data items in this file w.r.t. the whole dataset.
        For Text2TextDataset, the format of the input file should contain
        lines of input, each line representing one record of data. The input
        and output is separated by a tab(\t).

        Parameters
        ----------
        file_path: str
            The path of the input file.
        Returns
        -------
        list
            The indices of data items in the file w.r.t. the whole dataset.
        """
        data = []
        #print(file_path)
        with open(file_path, "r") as f:
            examples = json.load(f)
            for example_dict in examples:
                input = " ".join(" ".join(example_dict["article"]).split()[:400]).lower() # 只截400个单词会导致有些文本丢失信息，不过对于新闻（主要信息出现在前面）摘要够用
                output = " ".join(
                    " ".join(
                        ["<t> " + sent[0] + " . </t>" for sent in example_dict["highlight"]]
                    ).split()[:99]
                ).lower()
                if input == "" or output == "":
                    continue
                data_item = Text2TextDataItem( # 构成Text2TextDataItem实例
                    input_text=input, # article中每行句子排列成一行
                    output_text=output, # highlight中每行句子用<t></t>分隔并排列成一行
                    tokenizer=self.tokenizer, # nltk推荐tokenizer
                    share_vocab=self.share_vocab, # True
                )
                data.append(data_item)
        return data

    @staticmethod
    def collate_fn(data_list: [Text2TextDataItem]):
        graph_data = [item.graph for item in data_list]
        max_node_len = 0
        for graph_item in graph_data:
            max_node_len = max(max_node_len, graph_item.node_features["token_id"].size()[1])
        for graph_item in graph_data:
            token_id_numpy = graph_item.node_features["token_id"].numpy()
            token_id_pad = pad_2d_vals(token_id_numpy, token_id_numpy.shape[0], max_node_len) # 一个batch中的token_id填充至最大结点大小
            graph_item.node_features["token_id"] = torch.from_numpy(token_id_pad).long()
            #print(graph_item.node_features["token_id"].size())

        from graph4nlp.pytorch.data.data import to_batch

        big_graph = to_batch(graph_data) # 一个batch中的图整合成一个大图

        output_numpy = [item.output_np for item in data_list]
        output_str = [item.output_text.lower().strip() for item in data_list]
        output_pad = pad_2d_vals_no_size(output_numpy) # 一个batch中的output_np填充至等长
        tgt_seq = torch.from_numpy(output_pad).long()

        return {"graph_data": big_graph, "tgt_seq": tgt_seq, "output_str": output_str}
