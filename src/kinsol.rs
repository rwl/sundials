use anyhow::Result;
use std::os::raw::{c_int, c_long, c_void};
use std::pin::Pin;

use sundials_sys::{
    sunindextype, sunrealtype, KINCreate, KINFree, KINGetFuncNorm, KINGetNumFuncEvals,
    KINGetNumJacEvals, KINGetNumNonlinSolvIters, KINInit, KINSetConstraints, KINSetFuncNormTol,
    KINSetJacFn, KINSetLinearSolver, KINSetMAA, KINSetMaxSetupCalls, KINSetNumMaxIters,
    KINSetScaledStepTol, KINSetUserData, KINSol, N_Vector, SUNMatrix,
};

use crate::check::{check_is_success, check_non_null};
use crate::context::Context;
use crate::nvector::NVector;
use crate::sunlinsol::LinearSolver;
use crate::sunmatrix::SparseMatrix;

#[derive(PartialEq)]
pub enum Strategy {
    None = 0,
    LineSearch = 1,
    Picard = 2,
    FP = 3,
}

pub type SysFn<U> = fn(uu: &NVector, fval: &mut NVector, user_data: &Option<U>) -> i32;
pub type JacFn<U> = fn(
    u: &NVector,
    fu: &mut NVector, // TODO: mut?
    j_mat: &mut SparseMatrix,
    user_data: &Option<U>,
    tmp1: &NVector,
    tmp2: &NVector,
) -> i32;

struct UserDataWrapper<U> {
    actual_user_data: Option<U>,
    sys_fn: SysFn<U>,
    jac_fn: JacFn<U>,
}

fn empty_sys_fn<U>(_uu: &NVector, _fval: &mut NVector, _user_data: &Option<U>) -> i32 {
    sundials_sys::KIN_SUCCESS
}

fn empty_jac_fn<U>(
    _u: &NVector,
    _fu: &mut NVector,
    _j_mat: &mut SparseMatrix,
    _user_data: &Option<U>,
    _tmp1: &NVector,
    _tmp2: &NVector,
) -> i32 {
    sundials_sys::KIN_SUCCESS
}

extern "C" fn sys_fn_wrapper<U>(uu: N_Vector, fval: N_Vector, user_data: *mut c_void) -> c_int {
    let uu = NVector::from_raw(uu);
    let mut fval = NVector::from_raw(fval);
    let wrapper = unsafe { &*(user_data as *const UserDataWrapper<U>) };
    (wrapper.sys_fn)(&uu, &mut fval, &wrapper.actual_user_data)
}

unsafe extern "C" fn jac_fn_wrapper<U>(
    u: N_Vector,
    fu: N_Vector,
    j_mat: SUNMatrix,
    user_data: *mut c_void,
    tmp1: N_Vector,
    tmp2: N_Vector,
) -> c_int {
    let u = NVector::from_raw(u);
    let mut fu = NVector::from_raw(fu);
    let mut j_mat = SparseMatrix::from_raw(j_mat);

    let wrapper = unsafe { &*(user_data as *const UserDataWrapper<U>) };

    let tmp1 = NVector::from_raw(tmp1);
    let tmp2 = NVector::from_raw(tmp2);

    (wrapper.jac_fn)(
        &u,
        &mut fu,
        &mut j_mat,
        &wrapper.actual_user_data,
        &tmp1,
        &tmp2,
    )
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
        ls_a: Option<(&LinearSolver, &SparseMatrix)>,
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

        if let Some((ls, a)) = ls_a {
            // Linear solver must be set after KINInit, but before KINSetJacFn.
            let retval = unsafe { KINSetLinearSolver(self.kinmem, ls.sunlinsol, a.sunmatrix) };
            check_is_success(retval, "KINSetLinearSolver")?;
        }

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

    pub fn set_func_norm_tol(&mut self, fnormtol: impl Into<sunrealtype>) -> Result<()> {
        let retval = unsafe { KINSetFuncNormTol(self.kinmem, fnormtol.into()) };
        check_is_success(retval, "KINSetFuncNormTol")
    }

    pub fn set_num_max_iters(&mut self, mxiter: impl Into<sunindextype>) -> Result<()> {
        let retval = unsafe { KINSetNumMaxIters(self.kinmem, mxiter.into().try_into()?) };
        check_is_success(retval, "KINSetNumMaxIters")
    }

    pub fn set_max_setup_calls(&mut self, msbset: impl Into<sunindextype>) -> Result<()> {
        let retval = unsafe { KINSetMaxSetupCalls(self.kinmem, msbset.into().try_into()?) };
        check_is_success(retval, "KINSetMaxSetupCalls")
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

    pub fn num_jac_evals(&self) -> Result<usize> {
        let mut njevals: c_long = 0;
        let retval = unsafe { KINGetNumJacEvals(self.kinmem, &mut njevals) };
        check_is_success(retval, "KINGetNumJacEvals")?;
        Ok(njevals as usize)
    }

    pub fn set_maa(&mut self, maa: usize) -> Result<()> {
        let retval = unsafe { KINSetMAA(self.kinmem, maa as c_long) };
        check_is_success(retval, "KINSetMAA")
    }

    pub fn set_constraints(&mut self, constraints: &NVector) -> Result<()> {
        let retval = unsafe { KINSetConstraints(self.kinmem, constraints.n_vector) };
        check_is_success(retval, "KINSetConstraints")
    }

    pub fn set_scaled_step_tol(&mut self, scsteptol: sunrealtype) -> Result<()> {
        let retval = unsafe { KINSetScaledStepTol(self.kinmem, scsteptol) };
        check_is_success(retval, "KINSetScaledStepTol")
    }
}

impl<U> Drop for KIN<U> {
    fn drop(&mut self) {
        unsafe {
            KINFree(&mut self.kinmem);
        }
    }
}
