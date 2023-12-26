# Anomaly Detection Based on Sparse Reconstruction for Object Recognition

## ELEN E6876 Final Project

**Authors: Chengbo Zang (cz2678), Aishwarya Patange (aap2239)**

## Abstract

In this project, we explored anomaly detection based on sparse reconstruction in the context of object detection. We proposed to study the abnormality of the results from object detectors according to the sparsity of their LASSO and dictionary learning approximation given a set of annotated objects. Experiments were conducted on public PennFudan Dataset with manual annotations and COSMOS data processed by a custom object detector. We were able to obtain optimistic results on COSMOS data using a custom implementation of LASSO and dictionary learning using PyTorch. Effect of different optimization algorithms and choice of hyperparameters were also briefly discussed.

## Organization

```
.
├── annotators/                             # annotation tool
├── COSMOS_Notebook.ipynb                   # COSMOS experiments
├── data/                                   # datasets
├── dictlearn.py                            # dictionary learning algorithm
├── lasso.py                                # LASSO algorithm
├── Misc Notebooks/                         # sandboxes
├── optim.py                                # optimizers
├── PennFundan_Notebook.ipynb               # PennFudan experiments
├── README.md
└── utils.py                                # plottings & metrics

34 directories, 31 files
```

Steps to run the notebooks:
- Create a Virtual Environments
- Run ```pip install -r requirements.txt```
- Run the Notebooks

## Acknowledgement

The authors would like to thank Prof. John Wright for the inspiring lectures and insightful guidance.

## References

[1] Xuan Mo, V. Monga, R. Bala, and Zhigang Fan, “Adaptive Sparse Representations for Video Anomaly Detection,” IEEE Trans. Circuits Syst. Video Technol., vol. 24, no. 4, pp. 631–645, Apr. 2014, doi: 10.1109/TCSVT.2013.2280061.

[2] J. Yang and Y. Zhang, “Alternating Direction Algorithms for $\ell_1$-Problems in Compressive Sensing.” arXiv, Dec. 07, 2009. Accessed: Oct. 16, 2023. [Online]. Available: http://arxiv.org/abs/0912.1185

[3] H. Yan, K. Paynabar, and J. Shi, “Anomaly Detection in Images With Smooth Background via Smooth-Sparse Decomposition,” Technometrics, vol. 59, no. 1, pp. 102–114, Jan. 2017, doi: 10.1080/00401706.2015.1102764.

[4] L. Peng and R. Vidal, “Block Coordinate Descent on Smooth Manifolds: Convergence Theory and Twenty-One Examples.” arXiv, Oct. 12, 2023. Accessed: Oct. 16, 2023. [Online]. Available: http://arxiv.org/abs/2305.14744

[5] M. Dao, N. H. Nguyen, N. M. Nasrabadi, and T. D. Tran, “Collaborative Multi-sensor Classification via Sparsity-based Representation.” arXiv, Jun. 16, 2016. Accessed: Oct. 16, 2023. [Online]. Available: http://arxiv.org/abs/1410.7876

[6] Yu. Nesterov, “Gradient methods for minimizing composite functions,” Math. Program., vol. 140, no. 1, pp. 125–161, Aug. 2013, doi: 10.1007/s10107-012-0629-5.
[7] C. Xu, W. Mao, W. Zhang, and S. Chen, “Remember Intentions: Retrospective-Memory-based Trajectory Prediction.” arXiv, Mar. 22, 2022. Accessed: Oct. 16, 2023. [Online]. Available: http://arxiv.org/abs/2203.11474

[8] J. Wright, A. Y. Yang, A. Ganesh, S. S. Sastry, and Yi Ma, “Robust Face Recognition via Sparse Representation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 31, no. 2, pp. 210–227, Feb. 2009, doi: 10.1109/TPAMI.2008.79.

[9] J.-C. Wu, H.-Y. Hsieh, D.-J. Chen, C.-S. Fuh, and T.-L. Liu, “Self-supervised Sparse Representation for Video Anomaly Detection,” in Computer Vision – ECCV 2022, vol. 13673, S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, and T. Hassner, Eds., in Lecture Notes in Computer Science, vol. 13673. , Cham: Springer Nature Switzerland, 2022, pp. 729–745. doi: 10.1007/978-3-031-19778-9_42.

[10] A. Adler, M. Elad, Y. Hel-Or, and E. Rivlin, “Sparse Coding with Anomaly Detection,” J Sign Process Syst, vol. 79, no. 2, pp. 179–188, May 2015, doi: 10.1007/s11265-014-0913-0.

[11] D. Bertsimas, R. Cory-Wright, and N. A. G. Johnson, “Sparse Plus Low Rank Matrix Decomposition: A Discrete Optimization Approach.” arXiv, Oct. 01, 2023. Accessed: Oct. 16, 2023. [Online]. Available: http://arxiv.org/abs/2109.12701

[12] Y. Cong, J. Yuan, and J. Liu, “Sparse reconstruction cost for abnormal event detection,” in CVPR 2011, Colorado Springs, CO, USA: IEEE, Jun. 2011, pp. 3449–3456. doi: 10.1109/CVPR.2011.5995434.

