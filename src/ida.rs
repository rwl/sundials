use crate::check::check_is_success;
use crate::context::Context;
use crate::nvector::NVector;
use crate::sunlinsol::LinearSolver;
use crate::sunmatrix::SparseMatrix;

use anyhow::Result;
use std::ffi::c_int;
use std::os::raw::{c_long, c_void};
use std::pin::Pin;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use sundials_sys::{
    realtype, IDACreate, IDAEwtFn, IDAFree, IDAGetNumSteps, IDAInit, IDASStolerances,
    IDASVtolerances, IDASetInitStep, IDASetLinearSolver, IDASetMaxConvFails, IDASetMaxNonlinIters,
    IDASetMaxNumSteps, IDASetMaxOrd, IDASetNonlinConvCoef, IDASetNonlinearSolver, IDASetStopTime,
    IDASetUserData, IDAWFtolerances, N_VGetArrayPointer_Serial, N_VGetLength, N_Vector,
    SUNNonlinearSolver,
};

pub type ResFn<U> =
    fn(tt: f64, yy: &[realtype], yp: &[realtype], rr: &[realtype], user_data: &Option<U>) -> i32;

struct UserDataWrapper<U> {
    actual_user_data: Option<U>,
    res_fn: ResFn<U>,
}

fn empty_res_fn<U>(
    _tt: f64,
    _yy: &[realtype],
    _yp: &[realtype],
    _rr: &[realtype],
    _user_data: &Option<U>,
) -> i32 {
    sundials_sys::KIN_SUCCESS
}

extern "C" fn res_fn_wrapper<U>(
    tt: realtype,
    yy: N_Vector,
    yp: N_Vector,
    rr: N_Vector,
    user_data: *mut c_void,
) -> c_int {
    let yy = unsafe {
        let pointer = N_VGetArrayPointer_Serial(yy);
        let length = N_VGetLength(yy);
        from_raw_parts(pointer, length as usize)
    };
    let yp = unsafe {
        let pointer = N_VGetArrayPointer_Serial(yp);
        let length = N_VGetLength(yp);
        from_raw_parts(pointer, length as usize)
    };
    let rr = unsafe {
        let pointer = N_VGetArrayPointer_Serial(rr);
        let length = N_VGetLength(rr);
        from_raw_parts_mut(pointer, length as usize)
    };
    let wrapper = unsafe { &*(user_data as *const UserDataWrapper<U>) };
    (wrapper.res_fn)(tt, yy, yp, rr, &wrapper.actual_user_data)
}

pub struct IDA<U> {
    ida_mem: *mut c_void,
    wrapped_user_data: Pin<Box<UserDataWrapper<U>>>,
}

impl<U> IDA<U> {
    /// Instantiates an IDA solver object.
    pub fn new(context: &Context) -> Self {
        Self {
            ida_mem: unsafe { IDACreate(context.sunctx) },
            wrapped_user_data: Box::pin(UserDataWrapper {
                actual_user_data: None,
                res_fn: empty_res_fn,
            }),
        }
    }

    /// Provides required problem and solution specifications, allocates internal memory,
    /// and initializes IDA.
    pub fn init(
        &mut self,
        res_fn: Option<ResFn<U>>,
        t0: f64,
        yy0: &NVector,
        yp0: &NVector,
        user_data: Option<U>,
    ) -> Result<()> {
        let retval = unsafe {
            IDAInit(
                self.ida_mem,
                Some(res_fn_wrapper::<U>),
                t0,
                yy0.n_vector,
                yp0.n_vector,
            )
        };
        check_is_success(retval, "IDAInit")?;

        self.wrapped_user_data = Box::pin(UserDataWrapper {
            actual_user_data: user_data,
            res_fn: match res_fn {
                Some(res_fn) => res_fn,
                None => empty_res_fn,
            },
        });
        let retval = unsafe {
            IDASetUserData(
                self.ida_mem,
                self.wrapped_user_data.as_ref().get_ref() as *const _ as *mut c_void,
            )
        };
        check_is_success(retval, "IDASetUserData")?;

        Ok(())
    }

    /// Specify the maximum order of the linear multistep method.
    pub fn set_max_ord(&mut self, maxord: usize) -> Result<()> {
        let retval = unsafe { IDASetMaxOrd(self.ida_mem, maxord as c_int) };
        check_is_success(retval, "IDASetMaxOrd")
    }

    /// Specify the maximum number of steps to be taken by the solver in its attempt
    /// to reach the next output time.
    pub fn set_max_num_steps(&mut self, mxsteps: usize) -> Result<()> {
        let retval = unsafe { IDASetMaxNumSteps(self.ida_mem, mxsteps as c_long) };
        check_is_success(retval, "IDASetMaxNumSteps")
    }

    /// Specify the initial step size.
    pub fn set_init_step(&mut self, hin: impl Into<realtype>) -> Result<()> {
        let retval = unsafe { IDASetInitStep(self.ida_mem, hin.into()) };
        check_is_success(retval, "IDASetInitStep")
    }

    /// Specify the value of the independent variable `t` past which the solution is not to proceed.
    pub fn set_stop_time(&mut self, tstop: impl Into<realtype>) -> Result<()> {
        let retval = unsafe { IDASetStopTime(self.ida_mem, tstop.into()) };
        check_is_success(retval, "IDASetStopTime")
    }

    /// Specify scalar relative and absolute tolerances.
    pub fn ss_tolerances(
        &mut self,
        reltol: impl Into<realtype>,
        abstol: impl Into<realtype>,
    ) -> Result<()> {
        let retval = unsafe { IDASStolerances(self.ida_mem, reltol.into(), abstol.into()) };
        check_is_success(retval, "IDASStolerances")
    }

    /// Specify scalar relative tolerance and vector absolute tolerances.
    pub fn sv_tolerances(&mut self, reltol: impl Into<realtype>, abstol: &NVector) -> Result<()> {
        let retval = unsafe { IDASVtolerances(self.ida_mem, reltol.into(), abstol.n_vector) };
        check_is_success(retval, "IDASVtolerances")
    }

    /// Specify a user-supplied function `efun` that sets the multiplicative error weights
    /// for use in the weighted RMS norm.
    pub fn wf_tolerances(&mut self, efun: IDAEwtFn) -> Result<()> {
        let retval = unsafe { IDAWFtolerances(self.ida_mem, efun) };
        check_is_success(retval, "IDAWFtolerances")
    }

    /// Attaches a [LinearSolver] object `ls` and corresponding template Jacobian
    /// [Matrix] object `J` (if applicable) to IDA, initializing the IDALS linear
    /// solver interface.
    pub fn set_linear_solver(
        &mut self,
        ls: LinearSolver,
        a_mat: &Option<SparseMatrix>,
    ) -> Result<()> {
        // let a: SUNMatrix = match a_mat {
        //     Some(a_mat) => a_mat.sunmatrix,
        //     None => null(),
        // };
        let a = a_mat.as_ref().unwrap().sunmatrix; // FIXME
        let retval = unsafe { IDASetLinearSolver(self.ida_mem, ls.sunlinsol, a) };
        check_is_success(retval, "IDASetLinearSolver")
    }

    /// Attaches a [SUNNonlinearSolver] object (`nls`) to IDA.
    pub fn set_non_linear_solver(&mut self, nls: SUNNonlinearSolver) -> Result<()> {
        let retval = unsafe { IDASetNonlinearSolver(self.ida_mem, nls) };
        check_is_success(retval, "IDASetNonlinearSolver")
    }

    /// Specify the maximum number of nonlinear solver iterations in one solve attempt.
    pub fn set_max_nonlin_iters(&mut self, maxcor: usize) -> Result<()> {
        let retval = unsafe { IDASetMaxNonlinIters(self.ida_mem, maxcor as c_int) };
        check_is_success(retval, "IDASetMaxNonlinIters")
    }

    /// Specify the maximum number of nonlinear solver convergence failures in one step.
    pub fn set_max_conv_fails(&mut self, maxncf: usize) -> Result<()> {
        let retval = unsafe { IDASetMaxConvFails(self.ida_mem, maxncf as c_int) };
        check_is_success(retval, "IDASetMaxConvFails")
    }

    /// Specifies the safety factor in the nonlinear convergence test.
    pub fn set_nonlin_conv_coef(&mut self, epcon: impl Into<realtype>) -> Result<()> {
        let retval = unsafe { IDASetNonlinConvCoef(self.ida_mem, epcon.into()) };
        check_is_success(retval, "IDASetNonlinConvCoef")
    }

    /// Returns the cumulative number of internal steps taken by the solver (total so far).
    pub fn num_steps(&self) -> Result<usize> {
        let mut nsteps: c_long = 0;
        let retval = unsafe { IDAGetNumSteps(self.ida_mem, &mut nsteps) };
        check_is_success(retval, "IDAGetNumSteps")?;
        Ok(nsteps as usize)
    }
}

impl<U> Drop for IDA<U> {
    fn drop(&mut self) {
        unsafe {
            IDAFree(&mut self.ida_mem);
        }
    }
}
