import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os


def draw_single(result, out_dir):
    assert len(result["train_loss"]) == len(result["val_rouge"])
    x = list(range(1, len(result["train_loss"]) + 1))
    
    plt.subplot()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel("Train oss")
    plt.plot(x, result["train_loss"])
    plt.savefig(os.path.join(
        out_dir, "train_loss.png"
    ))
    plt.cla()


    plt.subplot()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel("Val ROUGE-L")
    plt.plot(x, result["val_rouge"])
    plt.savefig(os.path.join(
        out_dir, "val_rouge.png"
    ))

def draw_comparison(result, out_dir):
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.ylabel("ROUGE-L")
    for i in range(len(result["graph_construction_cp"]["x"])):
        plt.bar(i, result["graph_construction_cp"]["y"][i], label=result["graph_construction_cp"]["x"][i])
        plt.text(i, result["graph_construction_cp"]["y"][i], '%.2f' %result["graph_construction_cp"]["y"][i], ha='center', va='bottom')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(
        out_dir, "graph_construction_cp.png"
    ))
    plt.cla()

    ax.axes.xaxis.set_visible(False)
    plt.ylabel("ROUGE-L")
    for i in range(len(result["gnn_encoder_cp"]["x"])):
        plt.bar(i, result["gnn_encoder_cp"]["y"][i], label=result["gnn_encoder_cp"]["x"][i])
        plt.text(i, result["gnn_encoder_cp"]["y"][i], '%.2f' %result["gnn_encoder_cp"]["y"][i], ha='center', va='bottom')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(
        out_dir, "gnn_encoder_cp.png"
    ))
    plt.legend(loc='lower right')
    plt.cla()

    ax.axes.xaxis.set_visible(False)
    plt.ylabel("ROUGE-L")
    for i in range(len(result["graph_emb_cp"]["x"])):
        plt.bar(i, result["graph_emb_cp"]["y"][i], label=result["graph_emb_cp"]["x"][i])
        plt.text(i, result["graph_emb_cp"]["y"][i], '%.2f' %result["graph_emb_cp"]["y"][i], ha='center', va='bottom')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(
        out_dir, "graph_emb_cp.png"
    ))
    plt.legend(loc='lower right')
    plt.cla()

    ax.axes.xaxis.set_visible(False)
    plt.ylabel("ROUGE-L")
    for i in range(len(result["decoder_cp"]["x"])):
        plt.bar(i, result["decoder_cp"]["y"][i], label=result["decoder_cp"]["x"][i])
        plt.text(i, result["decoder_cp"]["y"][i], '%.2f' %result["decoder_cp"]["y"][i], ha='center', va='bottom')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(
        out_dir, "decoder_cp.png"
    ))
    plt.legend(loc='lower right')
    plt.cla()


def main(results):
    out_dir = "examples/pytorch/summarization/cnn"
    #draw_single(results["constituency-gcn-attn-cov-copy"], out_dir)
    draw_comparison(results["comparison"], out_dir)

if __name__ == "__main__":
    results = {
        "constituency-gcn-attn-cov-copy": {
            "train_loss": [210.1, 174.6, 162.7, 154.9, 150.0, 146.2, 143.8, 141.2, 139.4, 136.8, 134.4, 131.3, 130.5, 129.9, 128.3, 127.7, 127.1, 126.1],
            "val_rouge": [24.1, 29.8, 30.1, 29.8, 29.9, 30.5, 30.8, 30.9, 30.7, 30.4, 30.8, 30.5, 30.9, 30.8, 30.8, 30.8, 30.8, 30.8]
        },
        "comparison": {
            "color": ["salmon", "darkorange", "cornflowerblue", "mediumseagreen"],
            "graph_construction_cp": {
                "x": ["dependency-GCN-attn-cov", "constituency-GCN-attn-cov", "IE-GCN-attn-cov"],
                "y": [29.23, 29.98, 27.51]
            },
            "gnn_encoder_cp": {
                "x": ["IE-GCN-attn-cov", "IE-GAT-attn-cov", "IE-GraphSage-attn-cov", "IE-GGNN-attn-cov"],
                "y": [27.51, 27.10, 27.10, 27.40]
            },
            "graph_emb_cp": {
                "x": ["IE-GraphSage-attn-cov", "IE-GraphSage-mean-attn-cov", "IE-GraphSage-max-attn-cov"],
                "y": [27.10, 27.21, 27.54],
            },
            "decoder_cp": {
                "x": ["constituency-GCN-attn-cov-copy", "constituency-GCN-attn-cov", "constituency-GCN-attn", "constituency-GCN"],
                "y": [30.94, 29.98, 30.11, 29.92]
            }
        }
    }
    main(results)