use crate::check::check_is_success;

use anyhow::Result;
use std::alloc::{alloc, Layout};
use std::ptr::null_mut;
use sundials_sys::{SUNContext, SUNContext_Create, SUNContext_Free};

pub struct Context {
    pub(crate) sunctx: *mut SUNContext,
}

impl Context {
    pub fn new() -> Result<Self> {
        let context = unsafe { alloc(Layout::new::<SUNContext>()) as *mut SUNContext };
        let retval = unsafe { SUNContext_Create(null_mut(), context) };
        check_is_success(retval, "SUNContext_Create")?;

        Ok(Self { sunctx: context })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            SUNContext_Free(self.sunctx);
        }
    }
}
