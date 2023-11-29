use crate::check::{check_is_success, check_non_null};
use crate::context::Context;
use crate::vector::NVector;

use anyhow::Result;
use std::os::raw::{c_int, c_void};
use sundials_sys::{
    realtype, KINCreate, KINFree, KINGetFuncNorm, KINInit, KINSetFuncNormTol, KINSetUserData,
    KINSol, N_VGetArrayPointer_Serial, N_VGetLength, N_Vector,
};

pub enum Strategy {
    None = 0,
    LineSearch = 1,
    Picard = 2,
    FP = 3,
}

pub type SysFn<U> =
    fn(uu: &[realtype], fval: &mut [realtype], user_data: &mut Option<U>) -> Result<i32>;

struct UserDataWrapper<U> {
    actual_user_data: Option<U>,
    sys_fn: SysFn<U>,
}

fn empty_sys_fn<UU>(
    _uu: &[realtype],
    _fval: &mut [realtype],
    _user_data: &mut Option<UU>,
) -> Result<i32> {
    Ok(0)
}

extern "C" fn sys_fn_wrapper<U>(uu: N_Vector, fval: N_Vector, user_data: *mut c_void) -> c_int {
    let uu = unsafe {
        let pointer = N_VGetArrayPointer_Serial(uu);
        let length = N_VGetLength(uu);
        std::slice::from_raw_parts(pointer, length as usize)
    };
    let fval = unsafe {
        let pointer = N_VGetArrayPointer_Serial(fval);
        let length = N_VGetLength(fval);
        std::slice::from_raw_parts_mut(pointer, length as usize)
    };
    let wrapper = unsafe { &mut *(user_data as *mut UserDataWrapper<U>) };
    let res: i32 = (wrapper.sys_fn)(uu, fval, &mut wrapper.actual_user_data).unwrap();
    res
}

pub struct KIN<U> {
    kinmem: *mut c_void,
    wrapped_user_data: UserDataWrapper<U>,
}

impl<U> KIN<U> {
    pub fn new(context: Context) -> Result<Self> {
        let kinmem = unsafe { KINCreate(*context.sunctx) };
        check_non_null(kinmem, "KINCreate")?;
        Ok(KIN {
            kinmem,
            wrapped_user_data: UserDataWrapper {
                actual_user_data: None,
                sys_fn: empty_sys_fn,
            },
        })
    }

    pub fn init(&mut self, sys_fn: Option<SysFn<U>>, tmpl: &NVector) -> Result<()> {
        let retval = unsafe { KINInit(self.kinmem, Some(sys_fn_wrapper::<U>), tmpl.n_vector) };
        check_is_success(retval, "KINInit")?;

        self.wrapped_user_data.sys_fn = match sys_fn {
            Some(sys_fn) => sys_fn,
            None => empty_sys_fn,
        };
        let retval = unsafe {
            KINSetUserData(
                self.kinmem,
                &mut self.wrapped_user_data as *mut _ as *mut c_void,
            )
        };
        check_is_success(retval, "KINSetUserData")?;

        Ok(())
    }

    pub fn set_user_data(&mut self, user_data: U) -> Result<()> {
        self.wrapped_user_data.actual_user_data = Some(user_data);

        let retval = unsafe {
            KINSetUserData(
                self.kinmem,
                &mut self.wrapped_user_data as *mut _ as *mut c_void,
            )
        };
        check_is_success(retval, "KINSetUserData")?;

        Ok(())
    }

    pub fn set_func_norm_tol(&mut self, fnormtol: impl Into<realtype>) -> Result<()> {
        let retval = unsafe { KINSetFuncNormTol(self.kinmem, fnormtol.into()) };
        check_is_success(retval, "KINSetFuncNormTol")
    }

    pub fn solve(
        &self,
        uu: &NVector,
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

    pub fn func_form(&self) -> Result<realtype> {
        let mut fnorm: realtype = realtype::default();
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
