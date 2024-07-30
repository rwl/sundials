use crate::check::check_is_success;

use anyhow::Result;
use std::alloc::{alloc, Layout};
use sundials_sys::{SUNContext, SUNContext_Create, SUNContext_Free, SUN_COMM_NULL};

pub struct Context {
    pub(crate) sunctx: SUNContext,
}

impl Context {
    pub fn new() -> Result<Self> {
        let context = unsafe { alloc(Layout::new::<SUNContext>()) as *mut SUNContext };
        let retval = unsafe { SUNContext_Create(SUN_COMM_NULL, context) };
        check_is_success(retval, "SUNContext_Create")?;

        unsafe { Ok(Self { sunctx: *context }) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            SUNContext_Free(&mut self.sunctx);
        }
    }
}
