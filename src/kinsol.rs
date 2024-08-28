use crate::check::{check_is_success, check_non_null};
use crate::context::Context;
use crate::nvector::NVector;
use crate::sunlinsol::LinearSolver;
use crate::sunmatrix::SparseMatrix;

use anyhow::Result;
use std::os::raw::{c_int, c_long, c_void};
use std::pin::Pin;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use sundials_sys::{
    sunrealtype, KINCreate, KINFree, KINGetFuncNorm, KINGetNumFuncEvals, KINGetNumNonlinSolvIters,
    KINInit, KINSetFuncNormTol, KINSetJacFn, KINSetLinearSolver, KINSetMAA, KINSetUserData, KINSol,
    N_VGetArrayPointer_Serial, N_VGetLength, N_Vector, SUNMatrix,
};

pub enum Strategy {
    None = 0,
    LineSearch = 1,
    Picard = 2,
    FP = 3,
}

pub type SysFn<U> = fn(uu: &[sunrealtype], fval: &mut [sunrealtype], user_data: &Option<U>) -> i32;
pub type JacFn<U> = fn(
    u: &[sunrealtype],
    fu: &mut [sunrealtype],
    j_mat: &mut SparseMatrix,
    user_data: &Option<U>,
    tmp1: &[sunrealtype],
    tmp2: &[sunrealtype],
) -> i32;

struct UserDataWrapper<U> {
    actual_user_data: Option<U>,
    sys_fn: SysFn<U>,
    jac_fn: JacFn<U>,
}

fn empty_sys_fn<U>(_uu: &[sunrealtype], _fval: &mut [sunrealtype], _user_data: &Option<U>) -> i32 {
    sundials_sys::KIN_SUCCESS
}

fn empty_jac_fn<U>(
    _u: &[sunrealtype],
    _fu: &mut [sunrealtype],
    _j_mat: &mut SparseMatrix,
    _user_data: &Option<U>,
    _tmp1: &[sunrealtype],
    _tmp2: &[sunrealtype],
) -> i32 {
    sundials_sys::KIN_SUCCESS
}

extern "C" fn sys_fn_wrapper<U>(uu: N_Vector, fval: N_Vector, user_data: *mut c_void) -> c_int {
    let uu = unsafe {
        let pointer = N_VGetArrayPointer_Serial(uu);
        let length = N_VGetLength(uu);
        from_raw_parts(pointer, length as usize)
    };
    let fval = unsafe {
        let pointer = N_VGetArrayPointer_Serial(fval);
        let length = N_VGetLength(fval);
        from_raw_parts_mut(pointer, length as usize)
    };
    let wrapper = unsafe { &*(user_data as *const UserDataWrapper<U>) };
    (wrapper.sys_fn)(uu, fval, &wrapper.actual_user_data)
}

unsafe extern "C" fn jac_fn_wrapper<U>(
    u: N_Vector,
    fu: N_Vector,
    j_mat: SUNMatrix,
    user_data: *mut c_void,
    tmp1: N_Vector,
    tmp2: N_Vector,
) -> c_int {
    let u = unsafe {
        let pointer = N_VGetArrayPointer_Serial(u);
        let length = N_VGetLength(u);
        from_raw_parts(pointer, length as usize)
    };
    let fu = unsafe {
        let pointer = N_VGetArrayPointer_Serial(fu);
        let length = N_VGetLength(fu);
        from_raw_parts_mut(pointer, length as usize)
    };
    let mut j_mat = SparseMatrix::from_raw(j_mat);

    let wrapper = unsafe { &*(user_data as *const UserDataWrapper<U>) };

    let tmp1 = unsafe {
        let pointer = N_VGetArrayPointer_Serial(tmp1);
        let length = N_VGetLength(tmp1);
        from_raw_parts(pointer, length as usize)
    };
    let tmp2 = unsafe {
        let pointer = N_VGetArrayPointer_Serial(tmp2);
        let length = N_VGetLength(tmp2);
        from_raw_parts(pointer, length as usize)
    };

    (wrapper.jac_fn)(u, fu, &mut j_mat, &wrapper.actual_user_data, tmp1, tmp2)
}

/// Solves nonlinear algebraic systems.
pub struct KIN<U> {
    kinmem: *mut c_void,
    wrapped_user_data: Pin<Box<UserDataWrapper<U>>>,
}

impl<U> KIN<U> {
    pub fn new(context: &Context) -> Result<Self> {
        let kinmem = unsafe { KINCreate(context.sunctx) };
        check_non_null(kinmem, "KINCreate")?;
        Ok(KIN {
            kinmem,
            wrapped_user_data: Box::pin(UserDataWrapper {
                actual_user_data: None,
                sys_fn: empty_sys_fn,
                jac_fn: empty_jac_fn,
            }),
        })
    }

    pub fn init(
        &mut self,
        sys_fn: Option<SysFn<U>>,
        jac_fn: Option<JacFn<U>>,
        user_data: Option<U>,
        tmpl: &NVector,
    ) -> Result<()> {
        let retval = unsafe {
            KINInit(
                self.kinmem,
                match sys_fn {
                    Some(_) => Some(sys_fn_wrapper::<U>),
                    None => None,
                },
                tmpl.n_vector,
            )
        };
        check_is_success(retval, "KINInit")?;

        self.wrapped_user_data = Box::pin(UserDataWrapper {
            actual_user_data: user_data,
            sys_fn: match sys_fn {
                Some(sys_fn) => sys_fn,
                None => empty_sys_fn,
            },
            jac_fn: match jac_fn {
                Some(jac_fn) => jac_fn,
                None => empty_jac_fn,
            },
        });

        if let Some(_) = jac_fn {
            let retval = unsafe { KINSetJacFn(self.kinmem, Some(jac_fn_wrapper::<U>)) };
            check_is_success(retval, "KINSetJacFn")?;
        }

        let retval = unsafe {
            KINSetUserData(
                self.kinmem,
                self.wrapped_user_data.as_ref().get_ref() as *const _ as *mut c_void,
            )
        };
        check_is_success(retval, "KINSetUserData")?;

        Ok(())
    }

    // pub fn set_jac_func(&mut self) {
    //     unsafe { sundials_sys::KINSetJacFn(self.kinmem, jac) }
    // }

    // pub fn set_user_data(&mut self, user_data: U) -> Result<()> {
    //     self.wrapped_user_data.actual_user_data = Some(user_data);
    //
    //     let retval = unsafe {
    //         KINSetUserData(
    //             self.kinmem,
    //             &mut self.wrapped_user_data.as_ref().get_ref() as *mut _ as *mut c_void,
    //         )
    //     };
    //     check_is_success(retval, "KINSetUserData")?;
    //
    //     Ok(())
    // }

    pub fn set_func_norm_tol(&mut self, fnormtol: impl Into<sunrealtype>) -> Result<()> {
        let retval = unsafe { KINSetFuncNormTol(self.kinmem, fnormtol.into()) };
        check_is_success(retval, "KINSetFuncNormTol")
    }

    pub fn set_linear_solver(&mut self, ls: &LinearSolver, a: &SparseMatrix) -> Result<()> {
        let retval = unsafe { KINSetLinearSolver(self.kinmem, ls.sunlinsol, a.sunmatrix) };
        check_is_success(retval, "KINSetLinearSolver")
    }

    pub fn solve(
        &self,
        uu: &mut NVector,
        strategy: Strategy,
        u_scale: &NVector,
        f_scale: &NVector,
    ) -> Result<()> {
        let retval = unsafe {
            KINSol(
                self.kinmem,
                uu.n_vector,
                strategy as c_int,
                u_scale.n_vector,
                f_scale.n_vector,
            )
        };
        check_is_success(retval, "KINSol")
    }

    pub fn func_norm(&self) -> Result<sunrealtype> {
        let mut fnorm: sunrealtype = sunrealtype::default();
        let retval = unsafe { KINGetFuncNorm(self.kinmem, &mut fnorm) };
        check_is_success(retval, "KINGetFuncNorm")?;
        Ok(fnorm)
    }

    pub fn num_nonlin_solv_iters(&self) -> Result<usize> {
        let mut nniters: c_long = 0;
        let retval = unsafe { KINGetNumNonlinSolvIters(self.kinmem, &mut nniters) };
        check_is_success(retval, "KINGetNumNonlinSolvIters")?;
        Ok(nniters as usize)
    }

    pub fn num_func_evals(&self) -> Result<usize> {
        let mut nfevals: c_long = 0;
        let retval = unsafe { KINGetNumFuncEvals(self.kinmem, &mut nfevals) };
        check_is_success(retval, "KINGetNumFuncEvals")?;
        Ok(nfevals as usize)
    }

    pub fn set_maa(&mut self, maa: usize) -> Result<()> {
        let retval = unsafe { KINSetMAA(self.kinmem, maa as c_long) };
        check_is_success(retval, "KINSetMAA")?;
        Ok(())
    }
}

impl<U> Drop for KIN<U> {
    fn drop(&mut self) {
        unsafe {
            KINFree(&mut self.kinmem);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::context::Context;
    use crate::kinsol::{Strategy, KIN};
    use crate::nvector::NVector;
    use anyhow::{format_err, Result};
    use sundials_sys::{sunindextype, sunrealtype};

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
        fn ith(v: &[f64], i: usize) -> f64 {
            // Ith numbers components 1..NEQ
            // return nvector.IthS(v, i-1)
            v[i - 1]
        }
        fn set_ith(v: &mut [f64], i: usize, x: f64) {
            // Ith numbers components 1..NEQ
            // nvector.DataS(v)[i-1] = x
            v[i - 1] = x;
        }

        // System function
        fn roberts(y: &[sunrealtype], g: &mut [sunrealtype], _: &Option<()>) -> i32 {
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
        fn print_output(y: &[f64]) {
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
                // return false; //(-1)
            }
            ewt.inv();

            // compute the solution error
            // u.linear_sum(ONE, -ONE, &r#ref, &mut r#ref);
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

        kmem.init(Some(roberts), None, None, &y)?;

        /* Set optional inputs */

        // Specify stopping tolerance based on residual.
        let fnormtol = TOL;
        kmem.set_func_norm_tol(fnormtol)?;

        // Initial guess.
        y.fill_with(ZERO);
        set_ith(y.as_slice_mut(), 1, ONE);

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
        print_output(y.as_slice());

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
}
