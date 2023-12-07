use crate::nvector::NVector;
use crate::sunmatrix::SparseMatrix;

use spsolve::FactorSolver;
use std::alloc::{alloc, Layout};
use std::os::raw::{c_int, c_long, c_void};
use std::ptr::null_mut;
use sundials_sys::{
    realtype, sunindextype, N_VGetArrayPointer, N_VGetLength, N_VGetVectorID, N_VScale, N_Vector,
    N_Vector_ID_SUNDIALS_NVEC_OPENMP, N_Vector_ID_SUNDIALS_NVEC_PTHREADS,
    N_Vector_ID_SUNDIALS_NVEC_SERIAL, SUNContext, SUNLinSolFree, SUNLinSolNewEmpty,
    SUNLinearSolver, SUNLinearSolver_ID, SUNLinearSolver_ID_SUNLINEARSOLVER_CUSTOM,
    SUNLinearSolver_Type, SUNLinearSolver_Type_SUNLINEARSOLVER_DIRECT, SUNMatGetID, SUNMatrix,
    SUNMatrix_ID_SUNMATRIX_SPARSE, SUNSparseMatrix_Columns, SUNSparseMatrix_Rows,
    SUNSparseMatrix_SparseType, CSC_MAT, SUNLS_ILL_INPUT, SUNLS_MEM_FAIL, SUNLS_MEM_NULL,
    SUNLS_SUCCESS,
};

macro_rules! content {
    ($s:expr, $F:ident, $S:ident) => {{
        unsafe { &mut *((*$s).content as *mut SUNLinearSolverContentSpSolve<$F, $S>) }
    }};
}

struct SUNLinearSolverContentSpSolve<F, S: FactorSolver<sunindextype, f64, F>> {
    last_flag: c_int,
    first_factorize: bool,
    factors: F,
    solver: S,
    transpose: bool,
}

pub unsafe extern "C" fn sunlinsol_spsolve<F, S: FactorSolver<sunindextype, f64, F>>(
    y: N_Vector,
    a_mat: SUNMatrix,
    solver: S,
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
    (*(*s).ops).gettype = Some(sunlinsol_get_type_spsolve);
    (*(*s).ops).getid = Some(sunlinsol_get_id_spsolve);
    (*(*s).ops).initialize = Some(sunlinsol_initialize_spsolve::<F, S>);
    (*(*s).ops).setup = Some(sunlinsol_setup_spsolve::<F, S>);
    (*(*s).ops).solve = Some(sunlinsol_solve_spsolve::<F, S>);
    (*(*s).ops).lastflag = Some(sunlinsol_last_flag_spsolve::<F, S>);
    (*(*s).ops).space = Some(sunlinsol_space_spsolve);
    (*(*s).ops).free = Some(sunlinsol_free_spsolve);

    // Create content //
    let content: *mut SUNLinearSolverContentSpSolve<F, S> =
        alloc(Layout::new::<SUNLinearSolverContentSpSolve<F, S>>())
            as *mut SUNLinearSolverContentSpSolve<F, S>;
    if content.is_null() {
        SUNLinSolFree(s);
        return null_mut();
    }

    // Attach content //
    (*s).content = content as *mut c_void;

    // Fill content //
    (*content).last_flag = 0;
    (*content).first_factorize = true;
    (*content).factors = None;

    (*content).solver = solver;
    if SUNSparseMatrix_SparseType(a_mat) == CSC_MAT as i32 {
        (*content).transpose = false;
    } else {
        (*content).transpose = true;
    }

    s
}

// Implementation of linear solver operations //

extern "C" fn sunlinsol_get_type_spsolve(_: SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLinearSolver_Type_SUNLINEARSOLVER_DIRECT
}

extern "C" fn sunlinsol_get_id_spsolve(_: SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLinearSolver_ID_SUNLINEARSOLVER_CUSTOM
}

extern "C" fn sunlinsol_initialize_spsolve<F, S: FactorSolver<sunindextype, f64, F>>(
    s: SUNLinearSolver,
) -> c_int {
    // Force factorization //
    content!(s, F, S).first_factorize = true;

    content!(s, F, S).last_flag = SUNLS_SUCCESS as c_int;
    content!(s, F, S).last_flag
}

unsafe extern "C" fn sunlinsol_setup_spsolve<F, S: FactorSolver<sunindextype, f64, F>>(
    s: SUNLinearSolver,
    a_mat: SUNMatrix,
) -> c_int {
    // let retval: c_int;
    // let uround_twothirds: realtype = f64::pow(UNIT_ROUNDOFF, TWOTHIRDS);

    // Ensure that A is a sparse matrix.
    if SUNMatGetID(a_mat) != SUNMatrix_ID_SUNMATRIX_SPARSE {
        content!(s, F, S).last_flag = SUNLS_ILL_INPUT;
        return content!(s, F, S).last_flag;
    }

    // On first decomposition, get the symbolic factorization.

    // if content!(s).first_factorize {
    /* Perform symbolic analysis of sparsity structure */
    // if (SYMBOLIC(S))
    //   sun_klu_free_symbolic(&SYMBOLIC(S), &COMMON(S));
    // content!(S).symbolic = sun_klu_analyze(
    //     SUNSparseMatrix_NP(A),
    // (KLU_INDEXTYPE*) SUNSparseMatrix_IndexPointers(A),
    // (KLU_INDEXTYPE*) SUNSparseMatrix_IndexValues(A),
    // &COMMON(S),
    // );
    // if (SYMBOLIC(S) == NULL) {
    //   LASTFLAG(S) = SUNLS_PACKAGE_FAIL_UNREC;
    //   return(LASTFLAG(S));
    // }

    // Compute the LU factorization of the matrix.

    // if(NUMERIC(S))
    //   sun_klu_free_numeric(&NUMERIC(S), &COMMON(S));
    // content!(S).numeric = sun_klu_factor(
    // (KLU_INDEXTYPE*) SUNSparseMatrix_IndexPointers(A),
    //                         (KLU_INDEXTYPE*) SUNSparseMatrix_IndexValues(A),
    // SUNSparseMatrix_Data(A),
    // SYMBOLIC(S),
    // &COMMON(S),
    // );
    // if NUMERIC(S) == NULL {
    //   LASTFLAG(S) = SUNLS_PACKAGE_FAIL_UNREC;
    //   return(LASTFLAG(S));
    // }

    let a = SparseMatrix::from_raw(a_mat);

    let factors: F = content!(s, F, S)
        .solver
        .factor(
            a.columns(),
            &a.index_values(),
            &a.index_pointers(),
            a.data(),
        )
        .unwrap();
    content!(s, F, S).factors = factors;

    //     content!(s).first_factorize = false;
    // } else {

    // ...not the first decomposition, so just refactor.

    // retval = sun_klu_refactor(
    // (KLU_INDEXTYPE*) SUNSparseMatrix_IndexPointers(A),
    //                       (KLU_INDEXTYPE*) SUNSparseMatrix_IndexValues(A),
    // SUNSparseMatrix_Data(A),
    // SYMBOLIC(S),
    // NUMERIC(S),
    // &COMMON(S),
    // );
    // if (retval == 0) {
    //   LASTFLAG(S) = SUNLS_PACKAGE_FAIL_REC;
    //   return(LASTFLAG(S));
    // }

    // Check if a cheap estimate of the reciprocal of the condition
    // number is getting too small.  If so, delete
    // the prior numeric factorization and recompute it.

    // retval = sun_klu_rcond(SYMBOLIC(S), NUMERIC(S), &COMMON(S));
    // if (retval == 0) {
    //     content!(S).last_flag = SUNLS_PACKAGE_FAIL_REC as c_int;
    //     return content!(S).last_flag;
    // }

    // if content!(S).common.rcond < uround_twothirds {

    // Condition number may be getting large.
    // Compute more accurate estimate.

    // retval = sun_klu_condest(
    //     // (KLU_INDEXTYPE*) SUNSparseMatrix_IndexPointers(A),
    //     SUNSparseMatrix_Data(A),
    //     SYMBOLIC(S),
    //     NUMERIC(S),
    //     &COMMON(S),
    // );
    // if retval == 0 {
    //     content!(S).last_flag = SUNLS_PACKAGE_FAIL_REC as c_int;
    //     return content!(S).last_flag;
    // }
    //
    // if content!(S).common.condest > (ONE / uround_twothirds) {

    // More accurate estimate also says condition number is
    // large, so recompute the numeric factorization.

    //     sun_klu_free_numeric(&NUMERIC(S), &COMMON(S));
    //     content!(S).numeric = sun_klu_factor(
    //         // (KLU_INDEXTYPE*) SUNSparseMatrix_IndexPointers(A),
    //         //                             (KLU_INDEXTYPE*) SUNSparseMatrix_IndexValues(A),
    //         SUNSparseMatrix_Data(A),
    //         SYMBOLIC(S),
    //         &COMMON(S),
    //     );
    //     // if NUMERIC(S) == NULL {
    //     //   LASTFLAG(S) = SUNLS_PACKAGE_FAIL_UNREC;
    //     //       return(LASTFLAG(S));
    //     // }
    // }
    // }
    // }

    content!(s, F, S).last_flag = SUNLS_SUCCESS as c_int;
    content!(s, F, S).last_flag
}

unsafe extern "C" fn sunlinsol_solve_spsolve<F, S: FactorSolver<sunindextype, f64, F>>(
    s: SUNLinearSolver,
    a_mat: SUNMatrix,
    x: N_Vector,
    b: N_Vector,
    _tol: realtype,
) -> c_int {
    // Check for valid inputs.
    if a_mat.is_null() || s.is_null() || x.is_null() || b.is_null() {
        return SUNLS_MEM_NULL;
    }

    // Copy b into x.
    N_VScale(1.0, b, x);

    // Access x data array.
    let xdata = N_VGetArrayPointer(x);
    if xdata.is_null() {
        content!(s, F, S).last_flag = SUNLS_MEM_FAIL;
        return content!(s, F, S).last_flag;
    }

    // Call to solve the linear system.

    // flag = content!(S).solve(
    //     SYMBOLIC(S),
    //     NUMERIC(S),
    //     SUNSparseMatrix_NP(A),
    //     1,
    //     xdata,
    //     &COMMON(S),
    // );
    // if flag == 0 {
    //     content!(S).last_flag = SUNLS_PACKAGE_FAIL_REC as c_int;
    //     return content!(S).last_flag;
    // }
    let x = NVector::from_raw(x);

    content!(s, F, S)
        .solver
        .solve(
            &content!(s, F, S).factors,
            x.as_slice_mut(),
            content!(s, F, S).transpose,
        )
        .unwrap();

    content!(s, F, S).last_flag = SUNLS_SUCCESS as c_int;
    content!(s, F, S).last_flag
}

unsafe extern "C" fn sunlinsol_last_flag_spsolve<F, S: FactorSolver<sunindextype, f64, F>>(
    s: SUNLinearSolver,
) -> sunindextype {
    // Return the stored 'last_flag' value.
    if s.is_null() {
        return -1;
    }
    content!(s, F, S).last_flag as sunindextype
}

unsafe extern "C" fn sunlinsol_space_spsolve(
    _: SUNLinearSolver,
    lenrw_ls: *mut c_long,
    leniw_ls: *mut c_long,
) -> c_int {
    // Since the structures are opaque objects, we
    // omit those from these results.
    *leniw_ls = 2;
    *lenrw_ls = 0;
    SUNLS_SUCCESS as c_int
}

unsafe extern "C" fn sunlinsol_free_spsolve(s: SUNLinearSolver) -> c_int {
    // Return with success if already freed.
    if s.is_null() {
        return SUNLS_SUCCESS as c_int;
    }

    // Delete items from the contents structure (if it exists).
    if !(*s).content.is_null() {
        //   if (NUMERIC(s))
        //     sun_klu_free_numeric(&NUMERIC(s), &COMMON(s));
        //   if (SYMBOLIC(s))
        //     sun_klu_free_symbolic(&SYMBOLIC(s), &COMMON(s));
        sundials_sys::free((*s).content);
        (*s).content = null_mut();
    }

    // Delete generic structures.
    if !(*s).ops.is_null() {
        sundials_sys::free((*s).ops as *mut c_void);
        (*s).ops = null_mut();
    }
    sundials_sys::free(s as *mut c_void);
    // s = null_mut();
    SUNLS_SUCCESS as c_int
}
