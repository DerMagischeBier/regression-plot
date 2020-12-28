# regression-plot

Fitting a basic neural network to samples of a simple function using the [tch](https://crates.io/crates/tch) crate. The resulting predictions for the training samples are shown in a created plot looking like the one below.

![Regression plot](regression_plot.png)

The underlying function the neural network is trained to approximate is given by `0.09 * x^3 + 2 * cos(x) - 1`. During the training data creation an artificial sampling error is applied given by normal distributed error fractions.

After checking out the repository and building and running by invoking `cargo run` a comparable image output should be created. Be aware that building can take a while as the `tch` crate will depending on the exact setup download the desired version of C++ PyTorch library (libtorch). 