use crate::context::Context;
use crate::nvector::NVector;
use crate::sunmatrix::SparseMatrix;
use sundials_sys::{SUNLinSolFree, SUNLinearSolver};

pub struct LinearSolver {
    pub(crate) sunlinsol: SUNLinearSolver,
}

impl LinearSolver {
    pub fn new_klu(y: &NVector, a_mat: &SparseMatrix, sunctx: &Context) -> Self {
        Self {
            sunlinsol: unsafe {
                sundials_sys::SUNLinSol_KLU(y.n_vector, a_mat.sunmatrix, sunctx.sunctx)
            },
        }
    }
}

impl Drop for LinearSolver {
    fn drop(&mut self) {
        unsafe { SUNLinSolFree(self.sunlinsol) };
    }
}
