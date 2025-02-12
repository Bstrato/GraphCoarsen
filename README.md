This project extends the GraphSAGE framework by incorporating a concept called graph coarsening.
Unlike GCN, GraphSAGE samples a subset of nodes for training, which improves efficiency.
However, the nodes that are not sampled may still contain valuable information that should not be ignored.
To address this, I propose coarsening the remaining unsampled nodes and integrating their coarsened embeddings into
the sampled nodes, enriching the overall representation.
