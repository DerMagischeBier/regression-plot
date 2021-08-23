use anyhow::Result;
use plotters::prelude::*;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::path::Path;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Reduction, Tensor};

const SEED: u64 = 0;

fn main() -> Result<()> {
    create_plot(Path::new("regression_plot.png"))?;

    Result::Ok(())
}

fn create_plot(file_path: &Path) -> Result<()> {
    let (input_data, output_data) = generate_data(1000);

    // create tensors from regression data
    let input = Tensor::of_slice(&input_data)
        .view((input_data.len() as i64, 1))
        .to_kind(Kind::Float);
    let output = Tensor::of_slice(&output_data)
        .view((output_data.len() as i64, 1))
        .to_kind(Kind::Float);

    tch::manual_seed(SEED as i64);

    let vs = nn::VarStore::new(Device::Cpu);
    // set up neural network structure
    let net = net(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 5e-2).unwrap();

    // train neural network
    for epoch in 1..400 {
        let loss = net.forward(&input).mse_loss(&output, Reduction::Mean);
        opt.backward_step(&loss);

        println!("epoch: {:4} train loss: {:8.5}", epoch, f64::from(&loss),);
    }

    let root = BitMapBackend::new(file_path, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut plot = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-4.5f64..4.5f64, -10f64..6f64)?;

    plot.configure_mesh()
        .light_line_style(&WHITE)
        .draw()
        .unwrap();

    // plot training data points
    plot.draw_series(
        input_data
            .iter()
            .zip(output_data.iter())
            .map(|(x, y)| Circle::new((*x, *y), 2, GREEN.filled())),
    )?
    .label("Sampled Training Data")
    .legend(|(x, y)| Circle::new((x + 10, y), 2, GREEN.filled()));

    // create points to sample neural net output
    let sampling_points: Vec<f64> = (0..400).map(|x| (x - 200) as f64 * 0.02).collect();

    // plot neural network approximation of underlying function
    plot.draw_series(LineSeries::new(
        sampling_points.iter().map(|x| {
            let sampling_point_input = Tensor::of_slice(&[*x]).view((1, 1)).to_kind(Kind::Float);
            let predicted_result = net.forward(&sampling_point_input);

            (*x, predicted_result.double_value(&[0]))
        }),
        Into::<ShapeStyle>::into(&CYAN).stroke_width(2),
    ))?
    .label("Neural Network Approximation")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));

    plot.configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .border_style(&BLACK)
        .draw()?;

    Result::Ok(())
}

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(vs / "layer1", 1, 50, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 50, 1, Default::default()))
}

/// Samples points from function with added error following a normal distribution
fn generate_data(n_data_points: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    let data: Vec<f64> = (&mut rng)
        .sample_iter(Uniform::new(-4.0, 4.0))
        .take(n_data_points)
        .collect();

    // normal distribution to add sampling error
    let normal = Normal::new(0.0, 1.0).unwrap();
    let data_output = data
        .iter()
        .map(|&x| {
            let error = normal.sample(&mut rng);

            (0.09 * x.powi(3) + 2.0 * x.cos() - 1.0) + error
        })
        .collect::<Vec<_>>();

    (data, data_output)
}

#[cfg(test)]
mod test {
    use super::create_plot;
    use std::path::Path;

    #[test]
    fn test_creation_of_plot() {
        create_plot(Path::new("test_regression_plot.png")).unwrap();

        assert!(file_diff::diff(
            "regression_plot.png",
            "test_regression_plot.png"
        ));

        std::fs::remove_file("test_regression_plot.png").unwrap();
    }
}
