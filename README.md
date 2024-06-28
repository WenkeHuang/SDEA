# Self-Driven Entropy Aggregation for Byzantine-Robust Heterogeneous Federated Learning

> Self-Driven Entropy Aggregation for Byzantine-Robust Heterogeneous
Federated Learning,            
> Wenke Huang, Zekun Shi, Mang Ye, He Li, Bo Du
> *ICML, 2024*
> [Link]()

## News
* [2024-06-28] Repo created. Paper and code release.
* [2024-06-10] Repo created. Paper and code will come soon.

## Abstract
Federated learning presents massive potential for privacy-friendly collaboration. However, federated learning is deeply threatened by byzantine attacks, where malicious clients deliberately upload crafted vicious updates. While various robust aggregations have been proposed to defend against such attacks, they are subject to certain assumptions: homogeneous private data and related proxy datasets. To address these limitations, we propose Self-Driven Entropy Aggregation (SDEA), which leverages the random public dataset to conduct Byzantine-robust aggregation in heterogeneous federated learning. For Byzantine attackers, we observe that benign ones typically present more confident (sharper) predictions than evils on the public dataset. Thus, we highlight benign clients by introducing learnable aggregation weight to minimize the instance-prediction entropy of the global model on the random public dataset. Besides, with inherent data heterogeneity, we reveal that it brings heterogeneous sharpness. Specifically, clients are optimized under distinct distribution and thus present fruitful predictive preferences. The learnable aggregation weight blindly allocates high attention to limited ones for sharper predictions, resulting in a biased global model. To alleviate this problem, we encourage the global model to offer diverse predictions via batch-prediction entropy maximization and conduct clustering to equally divide honest weights to accommodate different tendencies. This endows SDEA to detect Byzantine attackers in heterogeneous federated learning. Empirical results demonstrate the effectiveness.
## Citation
```
@inproceedings{SDEA_ICML24,
    title    = {Self-Driven Entropy Aggregation for Byzantine-Robust Heterogeneous Federated Learning},
    author    = {Huang, Wenke and Shi, Zekun and Mang, Ye and Li, He and Bo, Du},
    booktitle = {ICML},
    year      = {2024}
}
```

## Relevant Projects
[3] Rethinking Federated Learning with Domain Shift: A Prototype View - CVPR 2023 [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)][[Code](https://github.com/WenkeHuang/RethinkFL)]

[2] Federated Graph Semantic and Structural Learning - IJCAI 2023 [[Link](https://marswhu.github.io/publications/files/FGSSL.pdf)][[Code](https://github.com/wgc-research/fgssl)]

[1] Learn from Others and Be Yourself in Heterogeneous Federated Learning - CVPR 2022 [[Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)][[Code](https://github.com/WenkeHuang/FCCL)]
