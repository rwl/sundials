use anyhow::{format_err, Result};
use sundials_sys::sunindextype;

use crate::context::Context;
use crate::kinsol::{Strategy, KIN};
use crate::nvector;
use crate::nvector::NVector;
use crate::sunlinsol::LinearSolver;
use crate::sunmatrix::{SparseMatrix, SparseType};

/// The following is a simple example problem, with the coding
/// needed for its solution by the accelerated fixed point solver in
/// KINSOL.
///
/// The problem is from chemical kinetics, and consists of solving
/// the first time step in a Backward Euler solution for the
/// following three rate equations:
/// ```txt
///    dy1/dt = -.04*y1 + 1.e4*y2*y3
///    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2
///    dy3/dt = 3.e2*(y2)^2
/// ```
/// on the interval from t = 0.0 to t = 0.1, with initial
/// conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
/// Run statistics (optional outputs) are printed at the end.
///
/// Programmers: Carol Woodward @ LLNL
#[test]
pub fn test_kinsol_roberts_fp() -> Result<()> {
    const NEQ: sunindextype = 3; // number of equations
    const Y10: f64 = 1.0; // initial y components
    const Y20: f64 = 0.0;
    const Y30: f64 = 0.0;
    const TOL: f64 = 1e-10; // function tolerance
    const DSTEP: f64 = 0.1; // Size of the single time step used

    const PRIORS: usize = 2;

    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;

    // This function is defined in order to write code which exactly matches
    // the mathematical problem description given.
    //
    // Ith(v,i) references the ith component of the vector v, where i is in
    // the range [1..NEQ] and NEQ is defined above. The Ith macro is defined
    // using the N_VIth macro in nvector.h. N_VIth numbers the components of
    // a vector starting from 0.
    fn ith(v: &NVector, i: usize) -> f64 {
        // Ith numbers components 1..NEQ
        v.as_slice()[i - 1]
    }
    fn set_ith(v: &mut NVector, i: usize, x: f64) {
        // Ith numbers components 1..NEQ
        v.as_slice_mut()[i - 1] = x;
    }

    // System function
    fn roberts(y: &NVector, g: &mut NVector, _: &Option<()>) -> i32 {
        let y1 = ith(y, 1);
        let y2 = ith(y, 2);
        let y3 = ith(y, 3);

        let yd1 = DSTEP * (-0.04 * y1 + 1.0e4 * y2 * y3);
        let yd3 = DSTEP * 3.0e2 * y2 * y2;

        set_ith(g, 1, yd1 + Y10);
        set_ith(g, 2, -yd1 - yd3 + Y20);
        set_ith(g, 3, yd3 + Y30);

        0
    }

    // Print solution at selected points
    fn print_output(y: &NVector) {
        let y1 = ith(y, 1);
        let y2 = ith(y, 2);
        let y3 = ith(y, 3);

        println!("y = {:e}  {:e}  {:e}", y1, y2, y3);
    }

    // Print final statistics
    fn print_final_stats(kmem: &KIN<()>) {
        let nni = kmem.num_nonlin_solv_iters().unwrap();
        let nfe = kmem.num_func_evals().unwrap();

        println!("\nFinal Statistics..\n");
        println!("nni      = {:6}    nfe     = {:6}", nni, nfe);
    }

    // compare the solution to a reference solution computed with a
    // tolerance of 1e-14
    fn check_ans(u: &NVector, rtol: f64, atol: f64) -> Result<()> {
        // create reference solution and error weight vectors
        let r#ref = u.clone();
        let mut ewt = u.clone();

        // set the reference solution data
        //sundials.IthS(ref, 0) = 9.9678538655358029e-01
        //sundials.IthS(ref, 1) = 2.9530060962800345e-03
        //sundials.IthS(ref, 2) = 2.6160735013975683e-04
        r#ref.as_slice_mut()[0] = 9.9678538655358029e-01;
        r#ref.as_slice_mut()[1] = 2.9530060962800345e-03;
        r#ref.as_slice_mut()[2] = 2.6160735013975683e-04;

        // compute the error weight vector
        r#ref.abs(&mut ewt);
        // ewt.scale(rtol);
        ewt *= rtol;
        // ewt.add_const(atol, &mut ewt);
        ewt += atol;
        if ewt.min() <= ZERO {
            return Err(format_err!("SUNDIALS_ERROR: check_ans failed - ewt <= 0"));
        }
        ewt.inv();

        // compute the solution error
        let err = NVector::linear_sum(ONE, u, -ONE, &r#ref).wrms_norm(&ewt);

        // is the solution within the tolerances?
        if err >= ONE {
            return Err(format_err!("SUNDIALS_WARNING: check_ans error={}", err));
        }

        Ok(())
    }

    // Print problem description
    println!("Example problem from chemical kinetics solving");
    println!("the first time step in a Backward Euler solution for the");
    println!("following three rate equations:");
    println!("    dy1/dt = -.04*y1 + 1.e4*y2*y3");
    println!("    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2");
    println!("    dy3/dt = 3.e2*(y2)^2");
    println!("on the interval from t = 0.0 to t = 0.1, with initial");
    println!("conditions: y1 = 1.0, y2 = y3 = 0.");
    println!("Solution method: Anderson accelerated fixed point iteration.");

    let sunctx = Context::new().unwrap();

    // Create vectors for solution and scales
    let mut y = NVector::new_serial(NEQ, &sunctx)?;

    let mut scale = NVector::new_serial(NEQ, &sunctx)?;

    // Initialize and allocate memory for KINSOL
    let mut kmem: KIN<()> = KIN::new(&sunctx)?;

    // y is used as a template

    // Set number of prior residuals used in Anderson acceleration.
    kmem.set_maa(PRIORS)?;

    kmem.init(Some(roberts), None, None, None, &y)?;

    /* Set optional inputs */

    // Specify stopping tolerance based on residual.
    let fnormtol = TOL;
    kmem.set_func_norm_tol(fnormtol)?;

    // Initial guess.
    y.fill_with(ZERO);
    set_ith(&mut y, 1, ONE);

    // Call KINSol to solve problem

    // No scaling used
    scale.fill_with(ONE);

    // Call main solver
    kmem.solve(
        &mut y,       // initial guess on input; solution vector
        Strategy::FP, // global strategy choice
        &scale,       // scaling vector, for the variable cc
        &scale,       // scaling vector for function values fval
    )?;

    /* Print solution and solver statistics */

    // Get scaled norm of the system function.
    let fnorm = kmem.func_norm()?;

    println!("\nComputed solution (||F|| = {:e}):\n", fnorm);
    print_output(&y);

    print_final_stats(&kmem);

    // check the solution error
    check_ans(&y, 1e-4, 1e-6)?;

    // Output:
    //
    // Example problem from chemical kinetics solving
    // the first time step in a Backward Euler solution for the
    // following three rate equations:
    //     dy1/dt = -.04*y1 + 1.e4*y2*y3
    //     dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2
    //     dy3/dt = 3.e2*(y2)^2
    // on the interval from t = 0.0 to t = 0.1, with initial
    // conditions: y1 = 1.0, y2 = y3 = 0.
    // Solution method: Anderson accelerated fixed point iteration.
    //
    // Computed solution (||F|| = 3.96494e-12):
    //
    // y =  9.967854e-01    2.953006e-03    2.616074e-04
    //
    // Final Statistics...
    //
    // nni      =      8    nfe     =      8

    Ok(())
}

/// This example solves a nonlinear system from.
///
/// Source: "Handbook of Test Problems in Local and Global Optimization",
///             C.A. Floudas, P.M. Pardalos et al.
///             Kluwer Academic Publishers, 1999.
/// Test problem 4 from Section 14.1, Chapter 14: Ferraris and Tronconi
///
/// This problem involves a blend of trigonometric and exponential terms.
/// ```txt
///    0.5 sin(x1 x2) - 0.25 x2/pi - 0.5 x1 = 0
///    (1-0.25/pi) ( exp(2 x1)-e ) + e x2 / pi - 2 e x1 = 0
/// ```
/// such that
/// ```txt
///    0.25 <= x1 <=1.0
///    1.5 <= x2 <= 2 pi
/// ```
/// The treatment of the bound constraints on x1 and x2 is done using
/// the additional variables
/// ```txt
///    l1 = x1 - x1_min >= 0
///    L1 = x1 - x1_max <= 0
///    l2 = x2 - x2_min >= 0
///    L2 = x2 - x2_max >= 0
/// ```
/// and using the constraint feature in KINSOL to impose
/// ```txt
///    l1 >= 0    l2 >= 0
///    L1 <= 0    L2 <= 0
/// ```
/// The Ferraris-Tronconi test problem has two known solutions.
/// The nonlinear system is solved by KINSOL using different
/// combinations of globalization and Jacobian update strategies
/// and with different initial guesses (leading to one or the other
/// of the known solutions).
///
/// Constraints are imposed to make all components of the solution
/// positive.
///
/// Programmers: Radu Serban @ LLNL
#[test]
fn test_fer_tron_kinsol() -> Result<()> {
    const NVAR: sunindextype = 2;
    const NEQ_FT: sunindextype = 3 * NVAR;
    const NNZ: sunindextype = 12;

    const FTOL: f64 = 1e-5; // function tolerance
    const STOL: f64 = 1e-5; // step tolerance

    const ZERO: f64 = 0.0;
    const PT25: f64 = 0.25;
    const PT5: f64 = 0.5;
    const ONE: f64 = 1.0;
    const ONEPT5: f64 = 1.5;
    const TWO: f64 = 2.0;

    const PI: f64 = 3.1415926;
    const E: f64 = 2.7182818;

    struct FerrarisTronconi {
        lb: Vec<f64>,
        ub: Vec<f64>,
        nnz: sunindextype,
    }

    // System function for predator-prey system
    #[allow(non_snake_case)]
    fn fer_tron_func(u: &NVector, f: &mut NVector, user_data: &Option<FerrarisTronconi>) -> i32 {
        let params = user_data.as_ref().unwrap();
        let lb = &params.lb;
        let ub = &params.ub;

        let udata = u.as_slice();
        let fdata = f.as_slice_mut();

        let x1 = udata[0];
        let x2 = udata[1];
        let l1 = udata[2];
        let L1 = udata[3];
        let l2 = udata[4];
        let L2 = udata[5];

        fdata[0] = PT5 * f64::sin(x1 * x2) - PT25 * x2 / PI - PT5 * x1;
        fdata[1] = (ONE - PT25 / PI) * (f64::exp(TWO * x1) - E) + E * x2 / PI - TWO * E * x1;
        fdata[2] = l1 - x1 + lb[0];
        fdata[3] = L1 - x1 + ub[0];
        fdata[4] = l2 - x2 + lb[1];
        fdata[5] = L2 - x2 + ub[1];

        0
    }

    // System Jacobian
    #[allow(non_snake_case)]
    fn fer_tron_jac(
        y: &NVector,
        _f: &mut NVector,
        J: &mut SparseMatrix,
        _user_data: &Option<FerrarisTronconi>,
        _tmp1: &NVector,
        _tmp2: &NVector,
    ) -> i32 {
        J.zero().unwrap();

        let (rowptrs, colvals, data) = J.index_pointers_values_data_mut();
        let yd = y.as_slice();

        rowptrs[0] = 0;
        rowptrs[1] = 2;
        rowptrs[2] = 4;
        rowptrs[3] = 6;
        rowptrs[4] = 8;
        rowptrs[5] = 10;
        rowptrs[6] = 12;

        // row 0: J(0,0) and J(0,1)
        data[0] = PT5 * f64::cos(yd[0] * yd[1]) * yd[1] - PT5;
        colvals[0] = 0;
        data[1] = PT5 * f64::cos(yd[0] * yd[1]) * yd[0] - PT25 / PI;
        colvals[1] = 1;

        // row 1: J(1,0) and J(1,1)
        data[2] = TWO * (ONE - PT25 / PI) * (f64::exp(TWO * yd[0]) - E);
        colvals[2] = 0;
        data[3] = E / PI;
        colvals[3] = 1;

        // row 2: J(2,0) and J(2,2)
        data[4] = -ONE;
        colvals[4] = 0;
        data[5] = ONE;
        colvals[5] = 2;

        // row 3: J(3,0) and J(3,3)
        data[6] = -ONE;
        colvals[6] = 0;
        data[7] = ONE;
        colvals[7] = 3;

        // row 4: J(4,1) and J(4,4)
        data[8] = -ONE;
        colvals[8] = 1;
        data[9] = ONE;
        colvals[9] = 4;

        // row 5: J(5,1) and J(5,5)
        data[10] = -ONE;
        colvals[10] = 1;
        data[11] = ONE;
        colvals[11] = 5;

        0
    }

    fn set_initial_guess1(u: &mut NVector, data: &FerrarisTronconi) {
        let udata = u.as_slice_mut();

        let lb = &data.lb;
        let ub = &data.ub;

        // There are two known solutions for this problem

        // this initial guess should take us to (0.29945; 2.83693)
        let x1 = lb[0];
        let x2 = lb[1];

        udata[0] = x1;
        udata[1] = x2;
        udata[2] = x1 - lb[0];
        udata[3] = x1 - ub[0];
        udata[4] = x2 - lb[1];
        udata[5] = x2 - ub[1];
    }

    fn set_initial_guess2(u: &NVector, data: &FerrarisTronconi) {
        let udata = u.as_slice_mut();

        let lb = &data.lb;
        let ub = &data.ub;

        // There are two known solutions for this problem

        // this initial guess should take us to (0.5; 3.1415926)
        let x1 = PT5 * (lb[0] + ub[0]);
        let x2 = PT5 * (lb[1] + ub[1]);

        udata[0] = x1;
        udata[1] = x2;
        udata[2] = x1 - lb[0];
        udata[3] = x1 - ub[0];
        udata[4] = x2 - lb[1];
        udata[5] = x2 - ub[1];
    }

    // Print first lines of output (problem description).
    fn print_header(fnormtol: f64, scsteptol: f64) {
        println!("\nFerraris and Tronconi test problem");
        println!("Tolerance parameters:");
        println!(
            "  fnormtol  = {:16.6}\n  scsteptol = {:16.6}",
            fnormtol, scsteptol
        );
    }

    // Print solution
    fn print_output(u: &NVector) {
        let ud = u.as_slice();
        println!(" {:8.6}  {:8.6}", ud[0], ud[1])
    }

    // Print final statistics contained in iopt
    fn print_final_stats(kmem: &KIN<FerrarisTronconi>) {
        //var nni, nfe, nje int64
        //var flag int

        let nni = kmem.num_nonlin_solv_iters().unwrap();
        // check(&flag, "KINGetNumNonlinSolvIters", 1)
        let nfe = kmem.num_func_evals().unwrap();
        // check(&flag, "KINGetNumFuncEvals", 1)
        let nje = kmem.num_jac_evals().unwrap();
        // check(&flag, "GetNumJacEvals", 1)

        println!("Final Statistics:");
        println!("  nni = {:5}    nfe  = {:5}", nni, nfe);
        println!("  nje = {:5}", nje)
    }

    fn solve_it(
        kmem: &mut KIN<FerrarisTronconi>,
        u: &mut NVector,
        s: &NVector,
        glstr: Strategy,
        mset: sunindextype,
    ) -> Result<()> {
        println!();

        if mset == 1 {
            print!("Exact Newton");
        } else {
            print!("Modified Newton");
        }

        if glstr == Strategy::None {
            println!();
        } else {
            println!(" with line search");
        }

        kmem.set_max_setup_calls(mset as i64)?;

        kmem.solve(u, glstr, s, s)?;

        print!("Solution:\n  [x1,x2] = ");
        print_output(u);

        print_final_stats(kmem);

        Ok(())
    }

    let mut data = FerrarisTronconi {
        lb: vec![0.0; NVAR as usize],
        ub: vec![0.0; NVAR as usize],
        nnz: 0,
    };
    data.lb[0] = PT25;
    data.ub[0] = ONE;
    data.lb[1] = ONEPT5;
    data.ub[1] = TWO * PI;
    data.nnz = NNZ;

    // Create serial vectors of length NEQ.
    let sunctx = Context::new()?;

    let mut u1 = NVector::new_serial(NEQ_FT, &sunctx)?;
    let mut u2 = NVector::new_serial(NEQ_FT, &sunctx)?;
    let mut u = NVector::new_serial(NEQ_FT, &sunctx)?;
    let mut s = NVector::new_serial(NEQ_FT, &sunctx)?;
    let c = NVector::new_serial(NEQ_FT, &sunctx)?;

    set_initial_guess1(&mut u1, &data);
    set_initial_guess2(&mut u2, &data);

    s.fill_with(ONE); // no scaling

    {
        let cd = c.as_slice_mut();
        cd[0] = ZERO; // no constraint on x1
        cd[1] = ZERO; // no constraint on x2
        cd[2] = ONE; // l1 = x1 - x1_min >= 0
        cd[3] = -ONE; // L1 = x1 - x1_max <= 0
        cd[4] = ONE; // l2 = x2 - x2_min >= 0
        cd[5] = -ONE; // L2 = x2 - x22_min <= 0
    }

    let fnormtol = FTOL;
    let scsteptol = STOL;

    let mut kmem: KIN<FerrarisTronconi> = KIN::new(&sunctx)?;

    kmem.set_constraints(&c)?;
    kmem.set_func_norm_tol(fnormtol)?;
    kmem.set_scaled_step_tol(scsteptol)?;

    // Create sparse SUNMatrix
    #[allow(non_snake_case)]
    let J = SparseMatrix::new(NEQ_FT, NEQ_FT, NNZ, SparseType::CSR, &sunctx);

    // Create KLU solver object
    #[allow(non_snake_case)]
    #[cfg(not(feature = "klu"))]
    let LS = LinearSolver::new_faer(&u, &J, &sunctx);
    #[cfg(feature = "klu")]
    let LS = LinearSolver::new_klu(&u, &J, &sunctx);

    kmem.init(
        Some(fer_tron_func),
        Some((&LS, &J)),
        Some(fer_tron_jac),
        Some(data),
        &u,
    )?;

    // Print out the problem size, solution parameters, initial guess.
    print_header(fnormtol, scsteptol);

    println!("\n------------------------------------------");
    println!("\nInitial guess on lower bounds");
    print!("  [x1,x2] = ");
    print_output(&u1);

    nvector::scale(ONE, &u1, &mut u);
    let glstr = Strategy::None;
    let mset = 1;
    solve_it(&mut kmem, &mut u, &s, glstr, mset)?;

    nvector::scale(ONE, &u1, &mut u);
    let glstr = Strategy::LineSearch;
    let mset = 1;
    solve_it(&mut kmem, &mut u, &s, glstr, mset)?;

    nvector::scale(ONE, &u1, &mut u);
    let glstr = Strategy::None;
    let mset = 0;
    solve_it(&mut kmem, &mut u, &s, glstr, mset)?;

    nvector::scale(ONE, &u1, &mut u);
    let glstr = Strategy::LineSearch;
    let mset = 0;
    solve_it(&mut kmem, &mut u, &s, glstr, mset)?;

    println!("\n------------------------------------------");
    println!("\nInitial guess in middle of feasible region");
    print!("  [x1,x2] = ");
    print_output(&u2);

    nvector::scale(ONE, &u2, &mut u);
    let glstr = Strategy::None;
    let mset = 1;
    solve_it(&mut kmem, &mut u, &s, glstr, mset)?;

    nvector::scale(ONE, &u2, &mut u);
    let glstr = Strategy::LineSearch;
    let mset = 1;
    solve_it(&mut kmem, &mut u, &s, glstr, mset)?;

    nvector::scale(ONE, &u2, &mut u);
    let glstr = Strategy::None;
    let mset = 0;
    solve_it(&mut kmem, &mut u, &s, glstr, mset)?;

    nvector::scale(ONE, &u2, &mut u);
    let glstr = Strategy::LineSearch;
    let mset = 0;
    solve_it(&mut kmem, &mut u, &s, glstr, mset)?;

    let recovered = kmem
        .user_data()
        .expect("expected FerrarisTronconi user data");
    assert_eq!(recovered.nnz, NNZ);

    // Output:
    //
    // Ferraris and Tronconi test problem
    // Tolerance parameters:
    //   fnormtol  =      1e-05
    //   scsteptol =      1e-05
    //
    // ------------------------------------------
    //
    // Initial guess on lower bounds
    //   [x1,x2] =      0.25       1.5
    //
    // Exact Newton
    // Solution:
    //   [x1,x2] =  0.299448   2.83693
    // Final Statistics:
    //   nni =     4    nfe  =     5
    //   nje =     4
    //
    // Exact Newton with line search
    // Solution:
    //   [x1,x2] =  0.299448   2.83693
    // Final Statistics:
    //   nni =     4    nfe  =     5
    //   nje =     4
    //
    // Modified Newton
    // Solution:
    //   [x1,x2] =  0.299448   2.83693
    // Final Statistics:
    //   nni =    12    nfe  =    13
    //   nje =     2
    //
    // Modified Newton with line search
    // Solution:
    //   [x1,x2] =  0.299448   2.83693
    // Final Statistics:
    //   nni =    12    nfe  =    13
    //   nje =     2
    //
    // ------------------------------------------
    //
    // Initial guess in middle of feasible region
    //   [x1,x2] =     0.625   3.89159
    //
    // Exact Newton
    // Solution:
    //   [x1,x2] =  0.499999    3.1416
    // Final Statistics:
    //   nni =     6    nfe  =     7
    //   nje =     6
    //
    // Exact Newton with line search
    // Solution:
    //   [x1,x2] =  0.500001   3.14159
    // Final Statistics:
    //   nni =     6    nfe  =     8
    //   nje =     6
    //
    // Modified Newton
    // Solution:
    //   [x1,x2] =       0.5   3.14159
    // Final Statistics:
    //   nni =    14    nfe  =    15
    //   nje =     2
    //
    // Modified Newton with line search
    // Solution:
    //   [x1,x2] =       0.5   3.14159
    // Final Statistics:
    //   nni =    14    nfe  =    15
    //   nje =     2

    Ok(())
}
