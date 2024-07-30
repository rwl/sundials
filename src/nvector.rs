use crate::check::check_non_null;
use crate::context::Context;
use anyhow::Result;
use sundials_sys::{
    sunrealtype, N_VClone, N_VConst, N_VDestroy, N_VGetArrayPointer_Serial, N_VGetLength,
    N_VNew_Serial, N_VPrint, N_Vector,
};

pub struct NVector {
    vec_length: usize,
    pub(crate) n_vector: N_Vector,
    raw: bool,
}

impl NVector {
    pub fn new_serial(vec_length: impl Into<i64>, sunctx: &Context) -> Result<Self> {
        let l: i64 = vec_length.into();
        let n_vector = unsafe { N_VNew_Serial(l, sunctx.sunctx) };
        check_non_null(n_vector, "N_VNew_Serial")?;
        Ok(Self {
            vec_length: l as usize,
            n_vector,
            raw: false,
        })
    }

    pub fn from_raw(n_vector: N_Vector) -> Self {
        Self {
            n_vector,
            vec_length: unsafe { N_VGetLength(n_vector) } as usize,
            raw: true,
        }
    }

    pub fn as_slice(&self) -> &[sunrealtype] {
        unsafe {
            let pointer = N_VGetArrayPointer_Serial(self.n_vector);
            std::slice::from_raw_parts(pointer, self.vec_length)
        }
    }

    pub fn as_slice_mut(&self) -> &mut [sunrealtype] {
        unsafe {
            let pointer = N_VGetArrayPointer_Serial(self.n_vector);
            std::slice::from_raw_parts_mut(pointer, self.vec_length)
        }
    }

    pub fn fill_with(&mut self, c: impl Into<sunrealtype>) {
        unsafe { N_VConst(c.into(), self.n_vector) }
    }

    pub fn print(&mut self) {
        unsafe { N_VPrint(self.n_vector) }
    }

    pub fn len(&self) -> usize {
        self.vec_length
    }
}

impl Drop for NVector {
    fn drop(&mut self) {
        if !self.raw {
            unsafe {
                N_VDestroy(self.n_vector);
            }
        }
    }
}

impl Clone for NVector {
    fn clone(&self) -> Self {
        let n_vector = unsafe { N_VClone(self.n_vector) };
        Self {
            vec_length: self.vec_length,
            n_vector,
            raw: self.raw,
        }
    }
}
