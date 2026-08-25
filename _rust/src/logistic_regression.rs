use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::mem::MaybeUninit;

use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;



struct BinaryLogisticProblem<'a> {
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, u32>,
    sample_weights: Option<ArrayView1<'a, f32>>,
    // NOTE: l1_reg and l2_reg are stored ALREADY DIVIDED by sample_weights_sum; see
    // fit_binary for why. The kernels use them as-is and must not rescale them.
    l1_reg: f32,
    l2_reg: f32,
    // Precomputed once in fit_binary. Recomputing `weights.sum()` inside cost() and
    // gradient() would repeat an O(n) reduction on every single function evaluation.
    sample_weights_sum: f32,
    cost_fn: fn(
        ArrayView2<f32>,
        ArrayView1<u32>,
        ArrayView1<f32>,
        f32,
        f32,
        f32,
        Option<ArrayView1<f32>>,
    ) -> f32,
    grad_fn: fn(
        ArrayView2<f32>,
        ArrayView1<u32>,
        ArrayView1<f32>,
        f32,
        f32,
        f32,
        Option<ArrayView1<f32>>,
    ) -> Array1<f32>,
}

impl<'a> CostFunction for BinaryLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Output = f32;

    fn cost(&self, w: &Array1<f32>) -> Result<f32, argmin::core::Error> {
        Ok((self.cost_fn)(
            self.x,
            self.y,
            w.view(),
            self.l1_reg,
            self.l2_reg,
            self.sample_weights_sum,
            self.sample_weights,
        ))
    }
}

impl<'a> Gradient for BinaryLogisticProblem<'a> {
    type Param = Array1<f32>;
    type Gradient = Array1<f32>;
    fn gradient(&self, w: &Array1<f32>) -> Result<Array1<f32>, argmin::core::Error> {
        Ok((self.grad_fn)(
            self.x,
            self.y,
            w.view(),
            self.l1_reg,
            self.l2_reg,
            self.sample_weights_sum,
            self.sample_weights,
        ))
    }
}

pub fn fit_binary<'a>(
    x: ArrayView2<'a, f32>,
    y: ArrayView1<'a, u32>,
    l1_reg: f32,
    l2_reg: f32,
    intercept: bool,
    max_iters: u64,
    m: usize,
    tolerance: f32,
    sample_weights: Option<ArrayView1<'a, f32>>,
) -> Result<Array1<f32>, Error> {

    // Sum of the sample weights (n when unweighted). Computed once here rather than on
    // every cost()/gradient() call, and used for two distinct purposes below: averaging
    // the data term, and normalising the penalty strengths.
    let sample_weights_sum = match sample_weights {
        Some(weights) => weights.sum(),
        None => x.nrows() as f32,
    };

    let l1_reg = l1_reg / sample_weights_sum;
    let l2_reg = l2_reg / sample_weights_sum;

    // the reason to define the problem in this way came
    // from https://docs.rs/argmin/latest/argmin/solver/quasinewton/struct.LBFGS.html#reference,
    // argmin requires "the problem" to implement the CostFunction and Gradient traits, so we define a struct that implements these traits and then pass it to the LBFGS solver.
    let problem = BinaryLogisticProblem {
        x,
        y,
        l1_reg,
        l2_reg,
        sample_weights,
        sample_weights_sum,
        cost_fn : match intercept {
            true => compute_loss_binary_intercept,
            false => compute_loss_binary_no_intercept,
        },
        grad_fn: match intercept {
            true => compute_new_gradient_binary_with_intercept,
            false => compute_new_gradient_binary_no_intercept,
        },
    };
    // The documentation of the lbfgs solver says to pass a line search method to the solver and I have chosen the MoreThuenteLineSearch due to its robustness as explained in:
    // https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/more-thuente-line-search-design.html
    let linesearch = MoreThuenteLineSearch::new();

    let w_shape: usize = match intercept {
        true => x.ncols() + 1,
        false => x.ncols(),
    };

    let w_init = Array1::<f32>::zeros(w_shape);

    // I instantiate the solver in the way the documnetation suggested
    let solver = LBFGS::new(linesearch, m).with_tolerance_grad(tolerance)?;

    // I pass the solver and the problem to the executor as the documentation suggested:
    // https://docs.rs/argmin/latest/argmin/core/struct.Executor.html
    let res = Executor::new(problem, solver)
        .configure(|state| state.param(w_init).max_iters(max_iters))
        .run()?;

    Ok(res
        .state
        .best_param
        .ok_or(Error::msg("No solution found"))?)
}




/// Kernels for the binary logistic_regression model.

pub fn dot_sigmoid_binary(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    bias: f32,
    intercept: bool,
) -> Array2<f32> {
    let z = x.dot(&w); // single gemv: score for every sample, shape (n_samples,)
    let b = if intercept { bias } else { 0.0 }; 

    // I fill the (n_samples, 2) output in one serial pass over z. The map is O(n) of
    // cheap scalar work and is dominated by the gemv above.
    let mut out = Array2::<f32>::uninit((z.len(), 2));
    out.axis_iter_mut(Axis(0))
        .zip(z.iter())
        .for_each(|(mut out_row, &zi)| {
            let prob = sigmoid(zi + b); // P(class 1)
            out_row[0] = MaybeUninit::new(1.0 - prob); // P(class 0)
            out_row[1] = MaybeUninit::new(prob);
        });

    unsafe { out.assume_init() } // every row was written above
}


pub fn dot_argmax_binary(
    x: ArrayView2<f32>,
    w: ArrayView1<f32>,
    bias: f32,
    intercept: bool,
) -> Array1<u32> {

    let z = x.dot(&w); // single gemv: w . x_row for every sample
    let b = if intercept { bias } else { 0.0 }; 

    z.mapv(|zi| (zi + b > 0.0) as u32) // class 1 iff z + b > 0, else class 0 (motivated by scikit-learn's implementation)
}

///

fn compute_loss_binary_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    
    let w_len = w.len();
    // last weight is the bias
    let (bias, base_w) = (w[w_len - 1], w.slice(s![..w_len - 1])); 

    // Reg over feature weights only (base_w), so the intercept stays unpenalized.
    // The two penalties are summed separately so that the L1 pass is skipped outright
    // when l1_reg is 0, which is every call the adapter makes today -- see l1_subgradient.
    let l2_term = base_w.iter().map(|wi| 0.5 * l2_reg * wi * wi).sum::<f32>();
    let l1_term = match l1_reg == 0.0 {
        true => 0.0,
        false => l1_reg * base_w.iter().map(|wi| wi.abs()).sum::<f32>(),
    };
    let reg = l1_term + l2_term;

    
    let log_loss = match sample_weights {
        Some(weights) => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap().into_par_iter())
            .zip(weights.as_slice().unwrap().into_par_iter())
            .map(|((row, label), weight)| {
                let z = row.dot(&base_w) + bias;
                (softplus(z) - (*label as f32) * z) * *weight
            })
            .sum::<f32>(),
        None => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap().into_par_iter())
            .map(|(row, label)| {
                let z = row.dot(&base_w) + bias;
                softplus(z) - (*label as f32) * z
            })
            .sum::<f32>(),
    };
    // We do the averaging here instead of inside the map due to numerical instability -> the accuracy after the fitting is better, when doing it here:
    // Possible reasons: using f32 and dividing small values by a much bigger value (sample_weights_sum) inside the map, can lead to  loss of precision. By summing first and then dividing, we reduce the risk of numerical instability and maintain better accuracy in the final loss calculation.
    (log_loss / sample_weights_sum) + reg
}

fn compute_loss_binary_no_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> f32 {
    // Same objective as the intercept variant, but here z = w . x with no bias term,
    // so every weight in w is a feature weight and the whole vector is regularized.
    // Split for the same reason as there: no L1 pass at all when l1_reg is 0.
    let l2_term = w.iter().map(|wi| 0.5 * l2_reg * wi * wi).sum::<f32>();
    let l1_term = match l1_reg == 0.0 {
        true => 0.0,
        false => l1_reg * w.iter().map(|wi| wi.abs()).sum::<f32>(),
    };
    let reg = l1_term + l2_term;

    let log_loss = match sample_weights {
        Some(weights) => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap().into_par_iter())
            .zip(weights.as_slice().unwrap().into_par_iter())
            .map(|((row, label), weight)| {
                let z = row.dot(&w);
                (softplus(z) - (*label as f32) * z) * *weight
            })
            .sum::<f32>(),
        None => x
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(y.as_slice().unwrap().into_par_iter())
            .map(|(row, label)| {
                let z = row.dot(&w);
                softplus(z) - (*label as f32) * z
            })
            .sum::<f32>(),
    };

    // We do the averaging here instead of inside the map due to numerical instability -> the accuracy after the fitting is better, when doing it here:
    // Possible reasons: using f32 and dividing small values by a much bigger value (sample_weights_sum) inside the map, can lead to  loss of precision. By summing first and then dividing, we reduce the risk of numerical instability and maintain better accuracy in the final loss calculation.
    (log_loss / sample_weights_sum) + reg
}

///

fn compute_new_gradient_binary_with_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {

    let w_len = w.len();
    let (bias, base_w) = (w[w_len - 1], w.slice(s![..w_len - 1]));


    let mut loss_grad = match sample_weights {
            Some(weights) => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .zip(weights.as_slice().unwrap().into_par_iter())
                .fold(
                    || Array1::<f32>::zeros(w_len),
                    |mut acc, ((row, label), weight)| {
                        let z = row.dot(&base_w) + bias;
                        // Hoist the per-row scalar out of the inner update: every
                        // feature is scaled by the same `weight * (sigmoid - y)`,
                        // so we compute it once instead of re-multiplying inside a loop.
                        let coeff = *weight * (sigmoid(z) - (*label as f32));

                        // `acc[..d] += coeff * row` as a single fused axpy: the scalar
                        // multiply (coeff * row) and the accumulate (acc += ...) are fused
                        // into one pass, so the intermediate coeff*row vector is never
                        // allocated. This replaces the hand-written `for j { acc[j] += ... }`
                        // loop, which paid a bounds check per feature and did not
                        // auto-vectorize well. scaled_add is ndarray's tuned kernel:
                        // contiguous, SIMD-friendly, one pass. The bias term sits at index
                        // w_len-1 and is updated separately since it has no matching x column.
                        acc.slice_mut(s![..w_len - 1]).scaled_add(coeff, &row);
                        acc[w_len - 1] += coeff;

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w_len), |a, b| a + b),
            None => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .fold(
                    || Array1::<f32>::zeros(w_len),
                    |mut acc, (row, label)| {
                        let z = row.dot(&base_w) + bias;
                        let coeff = sigmoid(z) - (*label as f32);

                        acc.slice_mut(s![..w_len - 1]).scaled_add(coeff, &row);
                        acc[w_len - 1] += coeff;

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w_len), |a, b| a + b)
    };

    for (g, &wi) in loss_grad.iter_mut().zip(base_w.iter()) {
        *g = *g / sample_weights_sum + l1_subgradient(wi, l1_reg) + l2_reg * wi;
    }

    loss_grad[w_len - 1] /= sample_weights_sum;

    loss_grad
}

fn compute_new_gradient_binary_no_intercept(
    x: ArrayView2<f32>,
    y: ArrayView1<u32>,
    w: ArrayView1<f32>,
    l1_reg: f32,
    l2_reg: f32,
    sample_weights_sum: f32,
    sample_weights: Option<ArrayView1<f32>>,
) -> Array1<f32> {
   
    let mut loss_grad = match sample_weights {
            Some(weights) => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .zip(weights.as_slice().unwrap().into_par_iter())
                .fold(
                    || Array1::<f32>::zeros(w.len()),
                    |mut acc, ((row, label), weight)| {
                        let z = row.dot(&w);
                        // One scalar per row; every feature shares it.
                        let coeff = *weight * (sigmoid(z) - (*label as f32));

                        // acc += coeff * row as a single fused axpy: the scalar multiply and
                        // the accumulate run in one pass, so no coeff*row temporary is
                        // allocated.
                        acc.scaled_add(coeff, &row);

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w.len()), |a, b| a + b),
            None => x
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(y.as_slice().unwrap().into_par_iter())
                .fold(
                    || Array1::<f32>::zeros(w.len()),
                    |mut acc, (row, label)| {
                        let z = row.dot(&w);
                        let coeff = sigmoid(z) - (*label as f32);

                        acc.scaled_add(coeff, &row);

                        acc
                    },
                )
                .reduce(|| Array1::<f32>::zeros(w.len()), |a, b| a + b)
    };

    for (g, &wi) in loss_grad.iter_mut().zip(w.iter()) {
        *g = *g / sample_weights_sum + l1_subgradient(wi, l1_reg) + l2_reg * wi;
    }

    loss_grad
}


// Helper Functions


#[inline(always)]
fn l1_subgradient(w: f32, l1_reg: f32) -> f32 {
    if l1_reg == 0.0 {
        0.0
    } else if w > 0.0 {
        l1_reg
    } else if w < 0.0 {
        -l1_reg
    } else {
        0.0
    }
}

#[inline(always)]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

// softplus, evaluated in a numerically stable way:
//   max(z, 0) + log(1 + exp(-|z|))
// The exponent is always <= 0, so exp() never overflows for finite z
// (the naive log(1 + exp(z)) hits inf in f32 around z = 88), and ln_1p
// keeps precision for small arguments. Used for the loss instead of
// routing through sigmoid -> ln, which returns -inf once sigmoid
// rounds to exactly 0 or 1.
#[inline(always)]
fn softplus(z: f32) -> f32 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}
