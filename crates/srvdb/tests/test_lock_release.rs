//! Dropping a database releases its lock even while a child process that
//! was forked in the meantime still holds inherited descriptors (issue #82,
//! ADR 0005)
#![cfg(unix)]

use srvdb::{CreateOptions, SrvDB};
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Duration;
use tempfile::TempDir;

/// Spawn `true` from another thread, with the child paused between fork
/// and exec. During the pause it holds a copy of every descriptor of this
/// process, including the database's lock file.
fn spawn_paused_child() -> std::thread::JoinHandle<()> {
    let handle = std::thread::spawn(|| {
        let mut cmd = Command::new("true");
        // SAFETY: the hook runs in the forked child before exec and only
        // calls nanosleep (through thread::sleep), which is
        // async-signal-safe; it touches no locks or allocations.
        unsafe {
            cmd.pre_exec(|| {
                std::thread::sleep(Duration::from_millis(1500));
                Ok(())
            });
        }
        let status = cmd.status().unwrap();
        assert!(status.success());
    });
    // Give the thread time to fork
    std::thread::sleep(Duration::from_millis(300));
    handle
}

#[test]
fn drop_releases_the_lock_while_a_forked_child_holds_it() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let db = SrvDB::create(path, 8, &CreateOptions::default()).unwrap();
    let child = spawn_paused_child();
    drop(db);
    let reopened = SrvDB::open_existing(path);
    child.join().unwrap();
    assert!(reopened.is_ok(), "{:?}", reopened.err());
}
