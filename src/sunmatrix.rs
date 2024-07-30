use crate::check::check_is_success;
use crate::context::Context;
use crate::nvector::NVector;

use anyhow::Result;
use std::os::raw::c_int;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use sundials_sys::{
    stdout, sunindextype, sunrealtype, SUNMatDestroy_Sparse, SUNMatMatvec_Sparse, SUNMatrix,
    SUNSparseMatrix, SUNSparseMatrix_Columns, SUNSparseMatrix_Data, SUNSparseMatrix_IndexPointers,
    SUNSparseMatrix_IndexValues, SUNSparseMatrix_NNZ, SUNSparseMatrix_Print, SUNSparseMatrix_Rows,
    SUNSparseMatrix_SparseType,
};

pub enum SparseType {
    CSC = 0,
    CSR = 1,
}

pub struct SparseMatrix {
    pub(crate) sunmatrix: SUNMatrix,
    raw: bool,
}

impl SparseMatrix {
    pub fn new(
        m: impl Into<sunindextype>,
        n: impl Into<sunindextype>,
        nnz: impl Into<sunindextype>,
        sparse_type: SparseType,
        context: &Context,
    ) -> Self {
        Self {
            sunmatrix: unsafe {
                SUNSparseMatrix(
                    m.into(),
                    n.into(),
                    nnz.into(),
                    sparse_type as c_int,
                    context.sunctx,
                )
            },
            raw: false,
        }
    }

    pub fn from_raw(sunmatrix: SUNMatrix) -> Self {
        Self {
            sunmatrix,
            raw: true,
        }
    }

    pub fn rows(&self) -> usize {
        unsafe { SUNSparseMatrix_Rows(self.sunmatrix) as usize }
    }

    pub fn columns(&self) -> usize {
        unsafe { SUNSparseMatrix_Columns(self.sunmatrix) as usize }
    }

    pub fn nnz(&self) -> usize {
        unsafe { SUNSparseMatrix_NNZ(self.sunmatrix) as usize }
    }

    pub fn sparse_type(&self) -> SparseType {
        match unsafe { SUNSparseMatrix_SparseType(self.sunmatrix) } {
            0 => SparseType::CSC,
            1 => SparseType::CSR,
            t => {
                panic!("sparse matrix type ({}) must be CSC or CSR", t);
            }
        }
    }

    pub fn index_pointers(&self) -> &[sunindextype] {
        let indptr = unsafe { SUNSparseMatrix_IndexPointers(self.sunmatrix) };
        match self.sparse_type() {
            SparseType::CSC => {
                let columns = self.columns();
                unsafe { from_raw_parts(indptr, columns + 1) }
            }
            SparseType::CSR => {
                let rows = self.rows();
                unsafe { from_raw_parts(indptr, rows + 1) }
            }
        }
    }

    pub fn index_values(&self) -> &[sunindextype] {
        let indval = unsafe { SUNSparseMatrix_IndexValues(self.sunmatrix) };
        let nnz = self.nnz();
        unsafe { from_raw_parts(indval, nnz) }
    }

    pub fn data(&self) -> &[sunrealtype] {
        let indval = unsafe { SUNSparseMatrix_Data(self.sunmatrix) };
        let nnz = self.nnz();
        unsafe { from_raw_parts(indval, nnz) }
    }

    pub fn index_pointers_mut(&mut self) -> &mut [sunindextype] {
        let indptr = unsafe { SUNSparseMatrix_IndexPointers(self.sunmatrix) };
        match self.sparse_type() {
            SparseType::CSC => {
                let columns = self.columns();
                unsafe { from_raw_parts_mut(indptr, columns + 1) }
            }
            SparseType::CSR => {
                let rows = self.rows();
                unsafe { from_raw_parts_mut(indptr, rows + 1) }
            }
        }
    }

    pub fn index_values_mut(&mut self) -> &mut [sunindextype] {
        let indval = unsafe { SUNSparseMatrix_IndexValues(self.sunmatrix) };
        let nnz = self.nnz();
        unsafe { from_raw_parts_mut(indval, nnz) }
    }

    pub fn data_mut(&mut self) -> &mut [sunrealtype] {
        let indval = unsafe { SUNSparseMatrix_Data(self.sunmatrix) };
        let nnz = self.nnz();
        unsafe { from_raw_parts_mut(indval, nnz) }
    }

    pub fn mat_vec(&self, x: &NVector, y: &mut NVector) -> Result<()> {
        let retval = unsafe { SUNMatMatvec_Sparse(self.sunmatrix, x.n_vector, y.n_vector) };
        check_is_success(retval, "SUNMatMatvec_Sparse")
    }

    pub fn print(&self) {
        unsafe { SUNSparseMatrix_Print(self.sunmatrix, stdout) }
    }
}

impl Drop for SparseMatrix {
    fn drop(&mut self) {
        if !self.raw {
            unsafe { SUNMatDestroy_Sparse(self.sunmatrix) }
        }
    }
}
