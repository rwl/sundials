use crate::check::check_non_null;
use crate::context::Context;
use anyhow::Result;
use sundials_sys::{
    realtype, N_VClone, N_VConst, N_VDestroy, N_VGetArrayPointer_Serial, N_VNew_Serial, N_Vector,
};

pub struct NVector {
    vec_length: usize,
    pub(crate) n_vector: N_Vector,
}

impl NVector {
    pub fn new_serial(vec_length: impl Into<i64>, sunctx: &Context) -> Result<Self> {
        let l: i64 = vec_length.into();
        let n_vector = unsafe { N_VNew_Serial(l, sunctx.sunctx) };
        check_non_null(n_vector, "N_VNew_Serial")?;
        Ok(Self {
            vec_length: l as usize,
            n_vector,
        })
    }

    pub fn as_slice(&self) -> &[sundials_sys::realtype] {
        unsafe {
            let pointer = N_VGetArrayPointer_Serial(self.n_vector);
            std::slice::from_raw_parts(pointer, self.vec_length)
        }
    }

    pub fn as_slice_mut(&self) -> &[realtype] {
        unsafe {
            let pointer = N_VGetArrayPointer_Serial(self.n_vector);
            std::slice::from_raw_parts_mut(pointer, self.vec_length)
        }
    }

    pub fn fill_with(&mut self, c: impl Into<realtype>) {
        unsafe { N_VConst(c.into(), self.n_vector) }
    }
}

impl Drop for NVector {
    fn drop(&mut self) {
        unsafe {
            N_VDestroy(self.n_vector);
        }
    }
}

impl Clone for NVector {
    fn clone(&self) -> Self {
        let n_vector = unsafe { N_VClone(self.n_vector) };
        Self {
            vec_length: self.vec_length,
            n_vector,
        }
    }
}
