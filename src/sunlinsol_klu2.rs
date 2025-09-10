use crate::context::Context;
use crate::nvector::NVector;
use crate::sunlinsol::LinearSolver;
use crate::sunmatrix::SparseMatrix;

use std::os::raw::{c_int, c_long, c_void};
use std::ptr::null_mut;

use suitesparse_sys::{
    klu_analyze, klu_common, klu_condest, klu_defaults, klu_factor, klu_free_numeric,
    klu_free_symbolic, klu_numeric, klu_rcond, klu_refactor, klu_solve, klu_symbolic, klu_tsolve,
};

use sundials_sys::{
    sunindextype, sunrealtype, N_VGetArrayPointer, N_VGetLength, N_VGetVectorID, N_VScale,
    N_Vector, N_Vector_ID_SUNDIALS_NVEC_OPENMP, N_Vector_ID_SUNDIALS_NVEC_PTHREADS,
    N_Vector_ID_SUNDIALS_NVEC_SERIAL, SUNContext, SUNErrCode, SUNLinSolFree, SUNLinSolNewEmpty,
    SUNLinearSolver, SUNLinearSolver_ID, SUNLinearSolver_ID_SUNLINEARSOLVER_CUSTOM,
    SUNLinearSolver_Type, SUNLinearSolver_Type_SUNLINEARSOLVER_DIRECT, SUNMatGetID, SUNMatrix,
    SUNMatrix_ID_SUNMATRIX_SPARSE, SUNSparseMatrix_Columns, SUNSparseMatrix_Rows,
    SUNSparseMatrix_SparseType, CSC_MAT, SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE,
    SUN_ERR_MEM_FAIL, SUN_SUCCESS,
};

impl LinearSolver {
    pub fn new_klu2(y: &NVector, a_mat: &SparseMatrix, sunctx: &Context) -> Self {
        Self {
            sunlinsol: unsafe { sunlinsol_klu2(y.n_vector, a_mat.sunmatrix, sunctx.sunctx) },
        }
    }
}

macro_rules! content {
    ($s:expr) => {{
        unsafe { &mut *((*$s).content as *mut SUNLinearSolverContentKlu2) }
    }};
}

#[repr(C)]
struct SUNLinearSolverContentKlu2 {
    last_flag: c_int,
    first_factorize: bool,
    symbolic: *mut klu_symbolic,
    numeric: *mut klu_numeric,
    common: klu_common,
    transpose: bool,
}

pub unsafe extern "C" fn sunlinsol_klu2(
    y: N_Vector,
    a_mat: SUNMatrix,
    sunctx: SUNContext,
) -> SUNLinearSolver {
    // Check compatibility with supplied SUNMatrix and N_Vector.
    if SUNMatGetID(a_mat) != SUNMatrix_ID_SUNMATRIX_SPARSE {
        return null_mut();
    }

    if SUNSparseMatrix_Rows(a_mat) != SUNSparseMatrix_Columns(a_mat) {
        return null_mut();
    }

    if (N_VGetVectorID(y) != N_Vector_ID_SUNDIALS_NVEC_SERIAL)
        && (N_VGetVectorID(y) != N_Vector_ID_SUNDIALS_NVEC_OPENMP)
        && (N_VGetVectorID(y) != N_Vector_ID_SUNDIALS_NVEC_PTHREADS)
    {
        return null_mut();
    }

    if SUNSparseMatrix_Rows(a_mat) != N_VGetLength(y) {
        return null_mut();
    }

    // Create an empty linear solver.
    let s: SUNLinearSolver = SUNLinSolNewEmpty(sunctx);
    if s.is_null() {
        return null_mut();
    }

    // Attach operations //
    (*(*s).ops).gettype = Some(sunlinsol_get_type_klu2);
    (*(*s).ops).getid = Some(sunlinsol_get_id_klu2);
    (*(*s).ops).initialize = Some(sunlinsol_initialize_klu2);
    (*(*s).ops).setup = Some(sunlinsol_setup_klu2);
    (*(*s).ops).solve = Some(sunlinsol_solve_klu2);
    (*(*s).ops).lastflag = Some(sunlinsol_last_flag_klu2);
    (*(*s).ops).space = Some(sunlinsol_space_klu2);
    (*(*s).ops).free = Some(sunlinsol_free_klu2);

    // Create content //
    let content_box = Box::new(SUNLinearSolverContentKlu2 {
        last_flag: 0,
        first_factorize: true,
        symbolic: null_mut(),
        numeric: null_mut(),
        common: klu_common {
            tol: 0.001,
            memgrow: 1.2,
            initmem: 1.0,
            initmem_amd: 1.0,
            maxwork: 0.0,
            btf: 1,
            ordering: 0,
            scale: 0,
            user_order: None,
            user_data: null_mut(),
            halt_if_singular: 1,
            status: 0,
            nrealloc: 0,
            structural_rank: 0,
            numerical_rank: 0,
            singular_col: 0,
            noffdiag: 0,
            flops: 0.0,
            rcond: 0.0,
            condest: 0.0,
            rgrowth: 0.0,
            work: 0.0,
            mempeak: 0,
            memusage: 0,
        },
        transpose: false,
    });
    let content = Box::into_raw(content_box);
    if content.is_null() {
        SUNLinSolFree(s);
        return null_mut();
    }

    // Attach content //
    (*s).content = content as *mut c_void;

    // Fill content //
    (*content).last_flag = 0;
    (*content).first_factorize = true;
    (*content).symbolic = null_mut();
    (*content).numeric = null_mut();

    // Initialize KLU defaults
    if klu_defaults(&mut (*content).common) != 1 {
        SUNLinSolFree(s);
        return null_mut();
    }

    // Set transpose based on matrix type
    if SUNSparseMatrix_SparseType(a_mat) == CSC_MAT as i32 {
        (*content).transpose = false;
    } else {
        (*content).transpose = true;
    }

    s
}

// Implementation of linear solver operations //

extern "C" fn sunlinsol_get_type_klu2(_: SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLinearSolver_Type_SUNLINEARSOLVER_DIRECT
}

extern "C" fn sunlinsol_get_id_klu2(_: SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLinearSolver_ID_SUNLINEARSOLVER_CUSTOM
}

extern "C" fn sunlinsol_initialize_klu2(s: SUNLinearSolver) -> SUNErrCode {
    // Force factorization //
    content!(s).first_factorize = true;

    content!(s).last_flag = SUN_SUCCESS as c_int;
    content!(s).last_flag
}

unsafe extern "C" fn sunlinsol_setup_klu2(s: SUNLinearSolver, a_mat: SUNMatrix) -> c_int {
    // Ensure that A is a sparse matrix.
    if SUNMatGetID(a_mat) != SUNMatrix_ID_SUNMATRIX_SPARSE {
        content!(s).last_flag = SUN_ERR_ARG_INCOMPATIBLE;
        return content!(s).last_flag;
    }

    let a_mat = SparseMatrix::from_raw(a_mat);
    let n = a_mat.rows() as c_int;
    let _nnz = a_mat.nnz() as c_int;

    // Get matrix data and convert to i32 for KLU
    let col_ptrs = a_mat.index_pointers();
    let row_indices = a_mat.index_values();
    let data = a_mat.data();

    // Convert sunindextype (i64) to i32 for KLU
    let col_ptrs_i32: Vec<i32> = col_ptrs.iter().map(|&x| x as i32).collect();
    let row_indices_i32: Vec<i32> = row_indices.iter().map(|&x| x as i32).collect();

    // On first decomposition, get the symbolic factorization.
    if content!(s).first_factorize {
        // Free any existing symbolic factorization
        if !content!(s).symbolic.is_null() {
            klu_free_symbolic(&mut content!(s).symbolic, &mut content!(s).common);
        }

        // Perform symbolic analysis
        content!(s).symbolic = klu_analyze(
            n,
            col_ptrs_i32.as_ptr(),
            row_indices_i32.as_ptr(),
            &mut content!(s).common,
        );
        if content!(s).symbolic.is_null() {
            content!(s).last_flag = SUN_ERR_MEM_FAIL;
            return content!(s).last_flag;
        }

        // Free any existing numeric factorization
        if !content!(s).numeric.is_null() {
            klu_free_numeric(&mut content!(s).numeric, &mut content!(s).common);
        }

        // Compute the LU factorization
        content!(s).numeric = klu_factor(
            col_ptrs_i32.as_ptr(),
            row_indices_i32.as_ptr(),
            data.as_ptr(),
            content!(s).symbolic,
            &mut content!(s).common,
        );
        if content!(s).numeric.is_null() {
            content!(s).last_flag = SUN_ERR_MEM_FAIL;
            return content!(s).last_flag;
        }

        content!(s).first_factorize = false;
    } else {
        // Not the first decomposition, so just refactor
        // Create mutable copies for refactor
        let mut col_ptrs_mut = col_ptrs_i32.clone();
        let mut row_indices_mut = row_indices_i32.clone();
        let mut data_mut = data.to_vec();

        let retval = klu_refactor(
            col_ptrs_mut.as_mut_ptr(),
            row_indices_mut.as_mut_ptr(),
            data_mut.as_mut_ptr(),
            content!(s).symbolic,
            content!(s).numeric,
            &mut content!(s).common,
        );
        if retval == 0 {
            content!(s).last_flag = SUN_ERR_MEM_FAIL;
            return content!(s).last_flag;
        }

        // Check condition number
        let retval = klu_rcond(
            content!(s).symbolic,
            content!(s).numeric,
            &mut content!(s).common,
        );
        if retval == 0 {
            content!(s).last_flag = SUN_ERR_MEM_FAIL;
            return content!(s).last_flag;
        }

        // If condition number is too large, recompute factorization
        let uround_twothirds = 0.666666666666666666666666666666667;
        if content!(s).common.rcond < uround_twothirds {
            // Compute more accurate condition estimate
            let mut col_ptrs_mut = col_ptrs_i32.clone();
            let mut data_mut = data.to_vec();
            let retval = klu_condest(
                col_ptrs_mut.as_mut_ptr(),
                data_mut.as_mut_ptr(),
                content!(s).symbolic,
                content!(s).numeric,
                &mut content!(s).common,
            );
            if retval == 0 {
                content!(s).last_flag = SUN_ERR_MEM_FAIL;
                return content!(s).last_flag;
            }

            if content!(s).common.condest > (1.0 / uround_twothirds) {
                // Recompute numeric factorization
                klu_free_numeric(&mut content!(s).numeric, &mut content!(s).common);
                content!(s).numeric = klu_factor(
                    col_ptrs_i32.as_ptr(),
                    row_indices_i32.as_ptr(),
                    data.as_ptr(),
                    content!(s).symbolic,
                    &mut content!(s).common,
                );
                if content!(s).numeric.is_null() {
                    content!(s).last_flag = SUN_ERR_MEM_FAIL;
                    return content!(s).last_flag;
                }
            }
        }
    }

    content!(s).last_flag = SUN_SUCCESS as c_int;
    content!(s).last_flag
}

unsafe extern "C" fn sunlinsol_solve_klu2(
    s: SUNLinearSolver,
    a_mat: SUNMatrix,
    x: N_Vector,
    b: N_Vector,
    _tol: sunrealtype,
) -> c_int {
    // Check for valid inputs.
    if a_mat.is_null() || s.is_null() || x.is_null() || b.is_null() {
        return SUN_ERR_ARG_CORRUPT;
    }

    // Copy b into x.
    N_VScale(1.0, b, x);

    // Access x data array.
    let xdata = N_VGetArrayPointer(x);
    if xdata.is_null() {
        content!(s).last_flag = SUN_ERR_MEM_FAIL;
        return content!(s).last_flag;
    }

    let a_mat = SparseMatrix::from_raw(a_mat);
    let n = a_mat.rows() as c_int;

    // Call KLU to solve the linear system
    let flag = if content!(s).transpose {
        klu_tsolve(
            content!(s).symbolic,
            content!(s).numeric,
            n,
            1,
            xdata,
            &mut content!(s).common,
        )
    } else {
        klu_solve(
            content!(s).symbolic,
            content!(s).numeric,
            n,
            1,
            xdata,
            &mut content!(s).common,
        )
    };

    if flag == 0 {
        content!(s).last_flag = SUN_ERR_MEM_FAIL;
        return content!(s).last_flag;
    }

    content!(s).last_flag = SUN_SUCCESS as c_int;
    content!(s).last_flag
}

unsafe extern "C" fn sunlinsol_last_flag_klu2(s: SUNLinearSolver) -> sunindextype {
    // Return the stored 'last_flag' value.
    if s.is_null() {
        return -1;
    }
    content!(s).last_flag as sunindextype
}

unsafe extern "C" fn sunlinsol_space_klu2(
    _: SUNLinearSolver,
    lenrw_ls: *mut c_long,
    leniw_ls: *mut c_long,
) -> SUNErrCode {
    // Since the structures are opaque objects, we
    // omit those from these results.
    *leniw_ls = 2;
    *lenrw_ls = 0;
    SUN_SUCCESS
}

unsafe extern "C" fn sunlinsol_free_klu2(s: SUNLinearSolver) -> SUNErrCode {
    // Return with success if already freed.
    if s.is_null() {
        return SUN_SUCCESS;
    }

    // Delete items from the contents structure (if it exists).
    if !(*s).content.is_null() {
        // Free KLU structures
        if !content!(s).numeric.is_null() {
            klu_free_numeric(&mut content!(s).numeric, &mut content!(s).common);
        }
        if !content!(s).symbolic.is_null() {
            klu_free_symbolic(&mut content!(s).symbolic, &mut content!(s).common);
        }

        // Convert back to Box, which will drop when it goes out of scope
        let _ = Box::from_raw((*s).content as *mut SUNLinearSolverContentKlu2);
        (*s).content = null_mut();
    }

    // Delete generic structures.
    if !(*s).ops.is_null() {
        sundials_sys::free((*s).ops as *mut c_void);
        (*s).ops = null_mut();
    }
    sundials_sys::free(s as *mut c_void);
    SUN_SUCCESS
}

#[cfg(test)]
mod tests {
    use crate::context::Context;
    use crate::nvector::NVector;
    use crate::sunlinsol::LinearSolver;
    use crate::sunmatrix::{SparseMatrix, SparseType};
    use std::f64;

    #[test]
    fn test_sunlinsol_klu2() {
        let n = 3;
        //     | 4 1 0 |
        // A = | 1 3 0 |
        //     | 0 1 2 |
        let rowptr = vec![0, 2, 4, 6];
        let colind = vec![0, 1, 0, 1, 1, 2];
        let data = vec![4.0, 1.0, 1.0, 3.0, 1.0, 2.0];

        let sunctx = Context::new().unwrap();

        let mut a_mat = SparseMatrix::new(n, n, *rowptr.last().unwrap(), SparseType::CSR, &sunctx);
        a_mat.index_pointers_mut().clone_from_slice(&rowptr);
        a_mat.index_values_mut().clone_from_slice(&colind);
        a_mat.data_mut().clone_from_slice(&data);
        // a_mat.print();

        let mut x = NVector::new_serial(n, &sunctx).unwrap();
        x.as_slice_mut().clone_from_slice(&[1.0, 2.0, 3.0]);
        let mut b = NVector::new_serial(n, &sunctx).unwrap();

        a_mat.mat_vec(&x, &mut b).unwrap();
        // b.print();

        x.fill_with(0.0);
        // x.print();

        let ls = LinearSolver::new_klu2(&x, &a_mat, &sunctx);
        // let ls = LinearSolver::new_klu(&x, &a_mat, &sunctx);

        ls.initialize().unwrap();
        ls.setup(&a_mat).unwrap();
        ls.solve(&a_mat, &mut x, &b, 1000.0 * f64::EPSILON).unwrap();
        // x.print();

        assert_eq!(x.as_slice(), &[1.0, 2.0, 3.0])
    }
}
