pub(crate) mod check;

pub mod context;
pub mod ida;
pub mod kinsol;
pub mod nvector;
pub mod sunlinsol;
#[cfg(feature = "faer")]
pub mod sunlinsol_faer;
#[cfg(feature = "klu2")]
pub mod sunlinsol_klu2;
pub mod sunmatrix;

#[cfg(test)]
mod kinsol_tests;

pub use sundials_sys::sunindextype;
pub use sundials_sys::sunrealtype;
