use anyhow::Result;
use plotters::prelude::*;
use rand::{thread_rng, Rng};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Reduction, Tensor};

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(vs / "layer1", 1, 10, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 10, 1, Default::default()))
}

fn produce_data(n_data_points: usize) -> (Vec<f64>, Vec<f64>) {
    let data = (0..n_data_points)
        .map(|_| thread_rng().gen_range(-4., 4.))
        .collect::<Vec<_>>();

    let data_output = data.iter().map(|&x| x * x).collect::<Vec<_>>();

    (data, data_output)
}

fn main() -> Result<()> {
    let (input_data, output_data) = produce_data(200);

    // create tensors from regression data
    let input = Tensor::of_slice(&input_data)
        .view((input_data.len() as i64, 1))
        .to_kind(Kind::Float);
    let output = Tensor::of_slice(&output_data)
        .view(output_data.len() as i64)
        .to_kind(Kind::Float);

    let vs = nn::VarStore::new(Device::Cpu);
    // set up neural net structure
    let net = net(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 5e-2).unwrap();
    for epoch in 1..500 {
        let loss = net
            .forward(&input)
            .mse_loss(&output, Reduction::None)
            .diag(0)
            .sum(Kind::Float);
        opt.backward_step(&loss);

        println!("epoch: {:4} train loss: {:8.5}", epoch, f64::from(&loss),);
    }

    let root = BitMapBackend::new("regression_plot.png", (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut plot = ChartBuilder::on(&root).build_ranged(-4.5f64..4.5f64, -0.5f64..17f64)?;

    // plot training data
    plot.draw_series(
        input_data
            .iter()
            .zip(output_data.iter())
            .map(|(x, y)| Circle::new((*x, *y), 2, GREEN.filled())),
    )?;

    // create points to sample neural net output
    let sampling_points = (0..400)
        .map(|x| (x - 200) as f64 * 0.02)
        .collect::<Vec<_>>();

    plot.draw_series(sampling_points.iter().map(|x| {
        let sampling_point_input = Tensor::of_slice(&[*x]).view((1, 1)).to_kind(Kind::Float);
        let predicted_result = net.forward(&sampling_point_input);
        Circle::new((*x, predicted_result.double_value(&[0])), 1, RED.filled())
    }))?;

    Result::Ok(())
}
