# [ICRA 2024] Benchmarking Classical and Learning-Based Multibeam Point Cloud Registration*
The data is now released and can be found [here](https://kth-my.sharepoint.com/:f:/g/personal/liling_ug_kth_se/EpXHLtknBFVIpvBIdMcNSOMBu8SPQIOt7tUNeUvQwB-O8g?e=ORQwEn).
This repository accompanies the ICRA 2024 paper:
### [Benchmarking Classical and Learning-Based Multibeam Point Cloud Registration](https://arxiv.org/pdf/2405.06279)
[Li Ling](https://www.kth.se/profile/liling)<sup>1</sup>, [Jun Zhang](https://www.tugraz.at/institute/icg/research/team-fraundorfer/people/jun-zhang)<sup>2</sup>, [Nils Bore](https://scholar.google.com/citations?user=wPea4DkAAAAJ&hl=en&oi=ao)<sup>3</sup>, [John Folkesson](https://www.kth.se/profile/johnf)<sup>1</sup>, [Anna Wåhlin](https://www.gu.se/en/about/find-staff/annawahlin)<sup>4</sup>

|<sup>1</sup>KTH Royal Institute of Technology|<sup>2</sup>TU Graz|<sup>3</sup>Ocean Infinity|<sup>4</sup>University of Gothenburg|

For more information, please check out the [project website](https://luxiya01.github.io/mbes-registration-project-page/)

### Contacts
If you have any questions, feel free to contact us at:
- Li Ling (liling@kth.se)

# Instructions
This repository contains the implementation for the MBES Dataset class and data loader, the classical methods GICP and FPFH, as well as the code for metrics computation and evaluations. 

The code use to run the learning-based models are found in the following repository forks:
- [FCGF](https://github.com/luxiya01/FCGF/tree/mbes_data)
- [DGR](https://github.com/luxiya01/DeepGlobalRegistration/tree/mbes_dataset)
- [Predator](https://github.com/luxiya01/OverlapPredator/tree/mbes_data)
- [BathyNN](https://github.com/luxiya01/bathy_nn_learning/tree/mbes-data)

The dataset, pretrained models and evaluation results can be found [here](https://kth-my.sharepoint.com/:f:/g/personal/liling_ug_kth_se/EpXHLtknBFVIpvBIdMcNSOMBu8SPQIOt7tUNeUvQwB-O8g?e=ORQwEn). Note that the dataset only contains the patches segmented according to the paper description. To construct a registration dataset, please consult [main.py](https://github.com/luxiya01/mbes-registration-data/blob/main/src/main.py). If you want the exact data pairs and transforms as used in the paper, you can also extract these from the _npz_ files containing in each method's evaluation results.

## Citation
If you find this code useful for your work, please consider citing:
```bibtex
@inproceedings{ling2024benchmarking,
            title={Benchmarking Classical and Learning-Based Multibeam Point Cloud Registration}, 
            author={Ling, Li and Zhang, Jun and Bore, Nils and Folkesson, John and Wåhlin, Anna},
            booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
            year={2024},
            organization={IEEE}
}
```

## Acknowledgements
In this project, we use part of the official implementations of the following work:
- [FCGF](https://github.com/chrischoy/FCGF)
- [DGR](https://github.com/chrischoy/DeepGlobalRegistration)
- [Predator](https://github.com/prs-eth/OverlapPredator)
- [BathyNN](https://github.com/tjr16/bathy_nn_learning)

We thank the respective authors for open sourcing their work.
