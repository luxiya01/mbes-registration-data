# [ICRA 2024] Benchmarking Classical and Learning-Based Multibeam Point Cloud Registration
This repository accompanies the ICRA 2024 paper:
### [Benchmarking Classical and Learning-Based Multibeam Point Cloud Registration](https://arxiv.org/pdf/2405.06279)
[Li Ling](https://www.kth.se/profile/liling)<sup>1</sup>, [Jun Zhang](https://www.tugraz.at/institute/icg/research/team-fraundorfer/people/jun-zhang)<sup>2</sup>, [Nils Bore](https://scholar.google.com/citations?user=wPea4DkAAAAJ&hl=en&oi=ao)<sup>3</sup>, [John Folkesson](https://www.kth.se/profile/johnf)<sup>1</sup>, [Anna Wåhlin](https://www.gu.se/en/about/find-staff/annawahlin)<sup>4</sup>

|<sup>1</sup>KTH Royal Institute of Technology|<sup>2</sup>TU Graz|<sup>3</sup>Ocean Infinity|<sup>4</sup>University of Gothenburg|

For more information, please check out the [project website](https://luxiya01.github.io/mbes-registration-project-page/)

### Contacts
If you have any questions, feel free to contact us at:
- Li Ling (liling@kth.se)

# Instructions
## Repository information
This repository contains the implementation for the MBES Dataset class and data loader, the classical methods GICP and FPFH, as well as the code for metrics computation and evaluations. The repositories for the learning-based methods, the pretrained models, as well as the instruction for installations, are to be updated.

## Dataset and pretrained models
To be updated.

## Citation
If you find this code useful for your work, please consider citing:
```bibtex
@misc{ling2024benchmarking,
      title={Benchmarking Classical and Learning-Based Multibeam Point Cloud Registration}, 
      author={Li Ling and Jun Zhang and Nils Bore and John Folkesson and Anna Wåhlin},
      year={2024},
      eprint={2405.06279},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
In this project, we use part of the official implementations of the following work:
- [FCGF](https://github.com/chrischoy/FCGF)
- [DGR](https://github.com/chrischoy/DeepGlobalRegistration)
- [Predator](https://github.com/prs-eth/OverlapPredator)
- [BathyNN](https://github.com/tjr16/bathy_nn_learning)
We thank the respective authors for open sourcing their work.
