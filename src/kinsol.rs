use crate::check::{check_is_success, check_non_null};
use crate::context::Context;
use crate::nvector::NVector;
use crate::sunlinsol::LinearSolver;
use crate::sunmatrix::SparseMatrix;

use anyhow::Result;
use std::os::raw::{c_int, c_void};
use std::pin::Pin;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use sundials_sys::{
    sunrealtype, KINCreate, KINFree, KINGetFuncNorm, KINInit, KINSetFuncNormTol, KINSetJacFn,
    KINSetLinearSolver, KINSetUserData, KINSol, N_VGetArrayPointer_Serial,
    N_VGetLength, N_Vector, SUNMatrix,
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
    j_mat: &SparseMatrix,
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
    _j_mat: &SparseMatrix,
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
    let j_mat = SparseMatrix::from_raw(j_mat);

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

    (wrapper.jac_fn)(u, fu, &j_mat, &wrapper.actual_user_data, tmp1, tmp2)
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

    pub fn func_form(&self) -> Result<sunrealtype> {
        let mut fnorm: sunrealtype = sunrealtype::default();
        let retval = unsafe { KINGetFuncNorm(self.kinmem, &mut fnorm) };
        check_is_success(retval, "KINGetFuncNorm")?;
        Ok(fnorm)
    }
}

impl<U> Drop for KIN<U> {
    fn drop(&mut self) {
        unsafe {
            KINFree(&mut self.kinmem);
        }
    }
}
