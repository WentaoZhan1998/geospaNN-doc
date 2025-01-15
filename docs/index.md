# GeospaNN - Neural networks for geospatial data
**Authors**: Wentao Zhan (<wzhan3@jhu.edu>), Abhirup Datta (<abhidatta@jhu.edu>)

**A package based on the paper: [Neural networks for geospatial data](https://www.tandfonline.com/doi/abs/10.1080/01621459.2024.2356293?casa_token=UaGsBumw4JAAAAAA:RD4cFpZW7lk3pu8Q5uVdxm5o3_RXWKRLXgxByEgl68qENKJfiNsS_Ci5izQ9WMQkZUKgSXasagyLQw)**

[GeospaNN](https://github.com/WentaoZhan1998/geospaNN) is a formal implementation of NN-GLS, the Neural Networks for geospatial data proposed in Zhan et.al (2023), that explicitly accounts for spatial correlation in the data. The package is developed using PyTorch and under the framework of PyG library. NN-GLS is a geographically-informed Graph Neural Network (GNN) for analyzing large and irregular geospatial data, that combines multi-layer perceptrons, Gaussian processes, and generalized least squares (GLS) loss. NN-GLS offers both regression function estimation and spatial prediction, and can scale up to sample sizes of hundreds of thousands. A  vignette is available at [https://github.com/WentaoZhan1998/geospaNN/blob/main/vignette.pdf](https://github.com/WentaoZhan1998/geospaNN/blob/main/vignette.pdf). Users are welcome to provide any helpful suggestions and comments.

## Contents
This documentation provides a comprehensive guide to getting started with the project and understanding its features. **Overview** provides a detailed description to the package. **How to Start** contains instructions for installation, setup, and an easy running example. **Documentation** dives into the details of each module, explaining their functions and configurations. Finally, **Examples** gives practical applications and demonstrations of the package in action.


1. [Overview](Overview.md)
2. [How to start](start.md)
3. [Documentation](Modules.md)
4. [Examples](Examples.md)

## Acknowledgements

Acknowledgement: This work was partially supported by the National Institute of Environmental Health Sciences (NIEHS) under grant R01 ES033739.
