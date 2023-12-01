use anyhow::{format_err, Result};
use std::os::raw::c_int;

pub(crate) fn check_non_null<T>(ptr: *mut T, func_id: &'static str) -> Result<()> {
    if ptr.is_null() {
        return Err(format_err!(
            "SUNDIALS_ERROR: {}() failed - returned NULL pointer",
            func_id
        ));
    }
    Ok(())
}

pub(crate) fn check_is_success(retval: c_int, func_name: &'static str) -> Result<()> {
    if retval != 0 {
        Err(format_err!(
            "SUNDIALS_ERROR: {}() failed with retval = {}",
            func_name,
            retval
        ))
    } else {
        Ok(())
    }
}
