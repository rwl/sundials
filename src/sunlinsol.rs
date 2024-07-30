use crate::check::check_is_success;
use crate::nvector::NVector;
use crate::sunmatrix::SparseMatrix;

use anyhow::Result;
use sundials_sys::{
    sunrealtype, SUNLinSolFree, SUNLinSolInitialize, SUNLinSolSetup, SUNLinSolSolve,
    SUNLinearSolver,
};

pub struct LinearSolver {
    pub(crate) sunlinsol: SUNLinearSolver,
}

impl LinearSolver {
    #[cfg(feature = "klu")]
    pub fn new_klu(y: &NVector, a_mat: &SparseMatrix, sunctx: &crate::context::Context) -> Self {
        Self {
            sunlinsol: unsafe {
                sundials_sys::SUNLinSol_KLU(y.n_vector, a_mat.sunmatrix, sunctx.sunctx)
            },
        }
    }

    pub fn initialize(&self) -> Result<()> {
        let rv = unsafe { SUNLinSolInitialize(self.sunlinsol) };
        check_is_success(rv, "SUNLinSolInitialize")
    }

    pub fn setup(&self, a_mat: &SparseMatrix) -> Result<()> {
        let rv = unsafe { SUNLinSolSetup(self.sunlinsol, a_mat.sunmatrix) };
        check_is_success(rv, "SUNLinSolSetup")
    }

    pub fn solve(
        &self,
        a_mat: &SparseMatrix,
        x: &mut NVector,
        b: &NVector,
        tol: sunrealtype,
    ) -> Result<()> {
        let rv =
            unsafe { SUNLinSolSolve(self.sunlinsol, a_mat.sunmatrix, x.n_vector, b.n_vector, tol) };
        check_is_success(rv, "SUNLinSolSolve")
    }
}

impl Drop for LinearSolver {
    fn drop(&mut self) {
        unsafe { SUNLinSolFree(self.sunlinsol) };
    }
}
