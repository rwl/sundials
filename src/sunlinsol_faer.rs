use crate::context::Context;
use crate::nvector::NVector;
use crate::sunlinsol::LinearSolver;
use crate::sunmatrix::{SparseMatrix, SparseType};

use std::alloc::{alloc, Layout};
use std::os::raw::{c_int, c_long, c_void};
use std::ptr::null_mut;

use faer::dyn_stack::{GlobalPodBuffer, PodStack};
use faer::perm::Perm;
use faer::sparse::linalg::amd;
use faer::sparse::linalg::lu::simplicial::{
    factorize_simplicial_numeric_lu, factorize_simplicial_numeric_lu_req, SimplicialLu,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, Mat, Parallelism};

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
    pub fn new_faer(y: &NVector, a_mat: &SparseMatrix, sunctx: &Context) -> Self {
        Self {
            sunlinsol: unsafe { sunlinsol_faer(y.n_vector, a_mat.sunmatrix, sunctx.sunctx) },
        }
    }
}

macro_rules! content {
    ($s:expr) => {{
        unsafe { &mut *((*$s).content as *mut SUNLinearSolverContentFaer) }
    }};
}

struct SUNLinearSolverContentFaer {
    last_flag: c_int,
    first_factorize: bool,
    factors: Option<SimplicialLu<usize, sunrealtype>>,
    row_perm: Option<Perm<usize>>,
    col_perm: Option<Perm<usize>>,
    transpose: bool,
}

pub unsafe extern "C" fn sunlinsol_faer(
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
    (*(*s).ops).gettype = Some(sunlinsol_get_type_faer);
    (*(*s).ops).getid = Some(sunlinsol_get_id_faer);
    (*(*s).ops).initialize = Some(sunlinsol_initialize_faer);
    (*(*s).ops).setup = Some(sunlinsol_setup_faer);
    (*(*s).ops).solve = Some(sunlinsol_solve_faer);
    (*(*s).ops).lastflag = Some(sunlinsol_last_flag_faer);
    (*(*s).ops).space = Some(sunlinsol_space_faer);
    (*(*s).ops).free = Some(sunlinsol_free_faer);

    // Create content //
    let content: *mut SUNLinearSolverContentFaer =
        alloc(Layout::new::<SUNLinearSolverContentFaer>()) as *mut SUNLinearSolverContentFaer;
    if content.is_null() {
        SUNLinSolFree(s);
        return null_mut();
    }

    // Attach content //
    (*s).content = content as *mut c_void;

    // Fill content //
    (*content).last_flag = 0;
    (*content).first_factorize = true;
    // (*content).factors = null();
    (*content).factors = None;

    // (*content).solver = solver.as_ref();
    // (*content).solver = solver;
    if SUNSparseMatrix_SparseType(a_mat) == CSC_MAT as i32 {
        (*content).transpose = false;
    } else {
        (*content).transpose = true;
    }

    s
}

// Implementation of linear solver operations //

extern "C" fn sunlinsol_get_type_faer(_: SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLinearSolver_Type_SUNLINEARSOLVER_DIRECT
}

extern "C" fn sunlinsol_get_id_faer(_: SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLinearSolver_ID_SUNLINEARSOLVER_CUSTOM
}

extern "C" fn sunlinsol_initialize_faer(s: SUNLinearSolver) -> SUNErrCode {
    // Force factorization //
    content!(s).first_factorize = true;

    content!(s).last_flag = SUN_SUCCESS as c_int;
    content!(s).last_flag
}

unsafe extern "C" fn sunlinsol_setup_faer(s: SUNLinearSolver, a_mat: SUNMatrix) -> c_int {
    // type E = faer::complex_native::c64;

    // let retval: c_int;
    // let uround_twothirds: sunrealtype = f64::pow(UNIT_ROUNDOFF, TWOTHIRDS);

    // Ensure that A is a sparse matrix.
    if SUNMatGetID(a_mat) != SUNMatrix_ID_SUNMATRIX_SPARSE {
        content!(s).last_flag = SUN_ERR_ARG_INCOMPATIBLE;
        return content!(s).last_flag;
    }

    // On first decomposition, get the symbolic factorization.
    // if content!(s).first_factorize {

    let a_mat = SparseMatrix::from_raw(a_mat);
    let (m, n) = (a_mat.rows(), a_mat.columns());

    let col_ptrs = a_mat
        .index_pointers()
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<usize>>();
    let row_indices = a_mat
        .index_values()
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<usize>>();
    let a_sym = SymbolicSparseColMatRef::new_checked(m, n, &col_ptrs, None, &row_indices);
    let mut row_perm = vec![0usize; n];
    let mut row_perm_inv = vec![0usize; n];
    let mut col_perm = vec![0usize; n];
    let mut col_perm_inv = vec![0usize; n];
    let control = amd::Control::default();
    amd::order(
        &mut col_perm,
        &mut col_perm_inv,
        a_sym.clone(),
        control,
        PodStack::new(&mut GlobalPodBuffer::new(
            amd::order_req::<usize>(m, n).unwrap(),
        )),
    )
    .unwrap();

    // Compute the LU factorization of the matrix.

    let a_ref = SparseColMatRef::<'_, usize, sunrealtype>::new(a_sym, &a_mat.data());
    // let col_perm = PermRef::<'_, usize>::new_checked(&col_perm, &col_perm_inv);
    let col_perm = Perm::<usize>::new_checked(Box::from(col_perm), Box::from(col_perm_inv));
    let mut lu: SimplicialLu<usize, sunrealtype> = SimplicialLu::new();
    factorize_simplicial_numeric_lu(
        &mut row_perm,
        &mut row_perm_inv,
        &mut lu,
        a_ref,
        col_perm.as_ref(),
        PodStack::new(&mut GlobalPodBuffer::new(
            factorize_simplicial_numeric_lu_req::<usize, sunrealtype>(m, n).unwrap(),
        )),
    )
    .unwrap();

    content!(s).factors = Some(lu);
    content!(s).row_perm = Some(Perm::<usize>::new_checked(
        Box::from(row_perm),
        Box::from(row_perm_inv),
    ));
    content!(s).col_perm = Some(col_perm);

    //     content!(s).first_factorize = false;
    // } else {

    // ...not the first decomposition, so just refactor.

    // Check if a cheap estimate of the reciprocal of the condition
    // number is getting too small. If so, delete
    // the prior numeric factorization and recompute it.

    // if content!(S).rcond < uround_twothirds {

    // Condition number may be getting large.
    // Compute more accurate estimate.

    // if content!(S).common.condest > (ONE / uround_twothirds) {

    // More accurate estimate also says condition number is
    // large, so recompute the numeric factorization.

    // }
    // }
    // }

    content!(s).last_flag = SUN_SUCCESS as c_int;
    content!(s).last_flag
}

unsafe extern "C" fn sunlinsol_solve_faer(
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

    // Call to solve the linear system.

    let a_mat = SparseMatrix::from_raw(a_mat);
    let x_vec = NVector::from_raw(x);
    let x = x_vec.as_slice();
    let (nrows, ncols) = (x.len(), 1);
    let mut rhs = Mat::<f64>::from_fn(nrows, ncols, |r, _| x[r]);

    let f = content!(s).factors.as_ref().unwrap();

    let row_perm = content!(s).row_perm.as_ref().unwrap().as_ref();
    let col_perm = content!(s).col_perm.as_ref().unwrap().as_ref();

    let mut work = rhs.clone();

    match a_mat.sparse_type() {
        SparseType::CSC => f.solve_in_place_with_conj(
            row_perm,
            col_perm,
            Conj::No,
            rhs.as_mut(),
            Parallelism::None,
            work.as_mut(),
        ),
        SparseType::CSR => f.solve_transpose_in_place_with_conj(
            row_perm,
            col_perm,
            Conj::No,
            rhs.as_mut(),
            Parallelism::None,
            work.as_mut(),
        ),
    }

    x_vec.as_slice_mut().clone_from_slice(rhs.col_as_slice(0));

    content!(s).last_flag = SUN_SUCCESS as c_int;
    content!(s).last_flag
}

unsafe extern "C" fn sunlinsol_last_flag_faer(s: SUNLinearSolver) -> sunindextype {
    // Return the stored 'last_flag' value.
    if s.is_null() {
        return -1;
    }
    content!(s).last_flag as sunindextype
}

unsafe extern "C" fn sunlinsol_space_faer(
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

unsafe extern "C" fn sunlinsol_free_faer(s: SUNLinearSolver) -> SUNErrCode {
    // Return with success if already freed.
    if s.is_null() {
        return SUN_SUCCESS;
    }

    // Delete items from the contents structure (if it exists).
    if !(*s).content.is_null() {
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
    fn test_sunlinsol_faer() {
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
        b.print();

        x.fill_with(0.0);
        // x.print();

        let ls = LinearSolver::new_faer(&x, &a_mat, &sunctx);
        // let ls = LinearSolver::new_klu(&x, &a_mat, &sunctx);

        ls.initialize().unwrap();
        ls.setup(&a_mat).unwrap();
        ls.solve(&a_mat, &mut x, &b, 1000.0 * f64::EPSILON).unwrap();
        // x.print();

        assert_eq!(x.as_slice(), &[1.0, 2.0, 3.0])
    }
}
