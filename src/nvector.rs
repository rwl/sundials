use anyhow::Result;
use std::ops::{AddAssign, MulAssign};
use sundials_sys::{
    sunrealtype, N_VAbs, N_VAddConst, N_VClone, N_VConst, N_VDestroy, N_VGetArrayPointer_Serial,
    N_VGetLength, N_VInv, N_VLinearSum, N_VMin, N_VNew_Serial, N_VPrint, N_VScale, N_VWrmsNorm,
    N_Vector,
};

use crate::check::check_non_null;
use crate::context::Context;

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

    pub fn abs(&self, z: &mut NVector) {
        unsafe { N_VAbs(self.n_vector, z.n_vector) }
    }

    pub fn scale(&self, c: sunrealtype, z: &mut NVector) {
        // pub fn scale(&mut self, c: sunrealtype) {
        unsafe { N_VScale(c, self.n_vector, z.n_vector) }
        // unsafe { N_VScale(c, self.n_vector, self.n_vector) }
    }

    pub fn add_const(&self, b: sunrealtype, z: &mut NVector) {
        // pub fn add_const(&mut self, b: sunrealtype) {
        unsafe { N_VAddConst(self.n_vector, b, z.n_vector) }
        // unsafe { N_VAddConst(self.n_vector, b, self.n_vector) }
    }

    pub fn min(&self) -> sunrealtype {
        unsafe { N_VMin(self.n_vector) }
    }

    // pub fn inv(&self, z: &mut NVector) {
    pub fn inv(&mut self) {
        // unsafe { N_VInv(self.n_vector, z.n_vector) }
        unsafe { N_VInv(self.n_vector, self.n_vector) }
    }

    /// Linear combination of two vectors: `z = a*x + b*y`.
    pub fn linear_sum(a: sunrealtype, x: &NVector, b: sunrealtype, y: &NVector) -> NVector {
        let z = x.clone();
        unsafe { N_VLinearSum(a, x.n_vector, b, y.n_vector, z.n_vector) }
        z
    }

    pub fn wrms_norm(&self, w: &NVector) -> sunrealtype {
        unsafe { N_VWrmsNorm(self.n_vector, w.n_vector) }
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

impl AddAssign<sunrealtype> for NVector {
    fn add_assign(&mut self, rhs: sunrealtype) {
        unsafe { N_VAddConst(self.n_vector, rhs, self.n_vector) }
    }
}

impl MulAssign<sunrealtype> for NVector {
    fn mul_assign(&mut self, rhs: sunrealtype) {
        unsafe { N_VScale(rhs, self.n_vector, self.n_vector) }
    }
}
