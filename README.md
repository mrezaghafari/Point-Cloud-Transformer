# Point-Cloud-Transformer

This repository contains a self-attention model that integrates sparse convolutions, specifically designed for point cloud coding tasks. The model is inspired by Point Transformer V2 @ https://arxiv.org/abs/2210.05666 and PCGFormer @ https://ieeexplore.ieee.org/document/10008892, but has been modified and adapted to better suit the requirements of point cloud coding.

Sparse convolutions obtained from '''MinkowskiEngine''' @https://github.com/NVIDIA/MinkowskiEngine and it allows for efficient handling of large-scale point clouds. The novel-developed self-attention captures the long-range dependencies between points using point features. This model is aimed at improving the performance of point cloud compression, segmentation, and other related tasks.

Key features:
* Differential Positional Embedding
* Relational Scoring
* Sparsemax probably

Input is a sparse tensor with points coordinates '''[n,3]''' and points features '''[n,d]''' and the attention is supposed to generate attended features.

Feel free to use the code and if you find code is useful please cite our work:
Paper: Point Cloud Geometry Coding with Relational Neighborhood Self-Attention
https://ieeexplore.ieee.org/abstract/document/10743834


