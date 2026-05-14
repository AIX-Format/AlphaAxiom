use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, TrayIconBuilder, TrayIconEvent},
    Manager,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // System Tray Setup
            let quit_i = MenuItem::with_id(app, "quit", "Quit Money Machine", true, None::<&str>)?;
            let show_i = MenuItem::with_id(app, "show", "Show/Hide Dashboard", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_i, &quit_i])?;

            let _tray = TrayIconBuilder::with_id("tray")
                .menu(&menu)
                .icon(app.default_window_icon().unwrap().clone())
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => {
                        app.exit(0);
                    }
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.toggle_visibility();
                        }
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| match event {
                    TrayIconEvent::Click {
                        button: MouseButton::Left,
                        ..
                    } => {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.toggle_visibility();
                        }
                    }
                    _ => {}
                })
                .build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_ignore_mouse_events,
            set_always_on_top,
            enable_keep_alive,
            disable_keep_alive,
            store_api_key,
            get_api_key,
            delete_api_key,
            get_ipc_auth_token
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

trait WindowExt {
    fn toggle_visibility(&self) -> tauri::Result<()>;
}

impl WindowExt for tauri::WebviewWindow {
    fn toggle_visibility(&self) -> tauri::Result<()> {
        if self.is_visible()? {
            self.hide()?;
        } else {
            self.show()?;
            self.set_focus()?;
        }
        Ok(())
    }
}

#[tauri::command]
fn set_ignore_mouse_events(window: tauri::WebviewWindow, ignore: bool) {
    let _ = window.set_ignore_cursor_events(ignore);
}

#[tauri::command]
fn set_always_on_top(window: tauri::WebviewWindow, state: bool) {
    let _ = window.set_always_on_top(state);
}

// ============================================================
// OS KEEP-ALIVE (Prevent system sleep during trading)
// ============================================================

use keepawake::{Builder as KeepAwakeBuilder, KeepAwake};
use once_cell::sync::Lazy;
use std::sync::Mutex;

static KEEP_AWAKE_HANDLE: Lazy<Mutex<Option<KeepAwake>>> = Lazy::new(|| Mutex::new(None));

/// Enables the OS keep-alive handle to prevent system idle and sleep for the application.
///
/// If a keep-alive handle is already active this returns `Ok("Keep-Alive already active")`.
/// On success it returns `Ok("Keep-Alive enabled")`. On failure it returns `Err` with a string describing the error (mutex lock failure or keep-awake creation failure).
///
/// # Examples
///
/// ```
/// // Call and check that a keep-alive response was returned.
/// let res = enable_keep_alive();
/// assert!(res.is_ok());
/// assert!(res.unwrap().contains("Keep-Alive"));
/// ```
#[tauri::command]
fn enable_keep_alive() -> Result<String, String> {
    let mut handle = KEEP_AWAKE_HANDLE.lock().map_err(|e| e.to_string())?;

    if handle.is_some() {
        return Ok("Keep-Alive already active".to_string());
    }

    let awake = KeepAwakeBuilder::default()
        .display(false) // Keep display on (optional)
        .idle(true) // Prevent idle sleep
        .sleep(true) // Prevent sleep
        .reason("Money Machine trading session")
        .app_name("Money Machine")
        .app_reverse_domain("com.antigravity.moneymachine")
        .create()
        .map_err(|e| format!("Failed to enable Keep-Alive: {}", e))?;

    *handle = Some(awake);
    log::info!("✅ OS Keep-Alive enabled");
    Ok("Keep-Alive enabled".to_string())
}

#[tauri::command]
fn disable_keep_alive() -> Result<String, String> {
    let mut handle = KEEP_AWAKE_HANDLE.lock().map_err(|e| e.to_string())?;

    if handle.is_none() {
        return Ok("Keep-Alive not active".to_string());
    }

    *handle = None; // Dropping the KeepAwake handle re-enables sleep
    log::info!("💤 OS Keep-Alive disabled");
    Ok("Keep-Alive disabled".to_string())
}

// ============================================================
// SECURE API KEY STORAGE (OS Keychain)
// ============================================================

use keyring::Entry;

const SERVICE_NAME: &str = "money-machine";

#[tauri::command]
fn store_api_key(key_name: String, key_value: String) -> Result<String, String> {
    let entry = Entry::new(SERVICE_NAME, &key_name).map_err(|e| format!("Keyring error: {}", e))?;

    entry
        .set_password(&key_value)
        .map_err(|e| format!("Failed to store key: {}", e))?;

    log::info!("🔐 Stored API key: {}", key_name);
    Ok(format!("Key '{}' stored securely", key_name))
}

#[tauri::command]
fn get_api_key(key_name: String) -> Result<String, String> {
    let entry = Entry::new(SERVICE_NAME, &key_name).map_err(|e| format!("Keyring error: {}", e))?;

    entry
        .get_password()
        .map_err(|e| format!("Failed to retrieve key: {}", e))
}

#[tauri::command]
fn delete_api_key(key_name: String) -> Result<String, String> {
    let entry = Entry::new(SERVICE_NAME, &key_name).map_err(|e| format!("Keyring error: {}", e))?;

    entry
        .delete_credential()
        .map_err(|e| format!("Failed to delete key: {}", e))?;

    log::info!("🗑️ Deleted API key: {}", key_name);
    Ok(format!("Key '{}' deleted", key_name))
}

// ============================================================
// IPC AUTH TOKEN (Rust <-> Python sidecar shared secret)
// ============================================================
//
// Returns the secret used to authenticate every TCP IPC request to
// the Python trading engine. The token lives in the OS keychain under
// service "money-machine" / account "ipc-auth-token". If it does not
// exist yet (first launch on a clean machine), one is generated with
// 32 bytes of OS-grade randomness and persisted. Subsequent calls are
// constant-time lookups, no rotation.
//
// The Python sidecar reads the same keychain entry (via the `keyring`
// Python package) when the IPC_AUTH_TOKEN env var is unset, so both
// sides converge on the same secret without copying it through dev
// dotfiles. Callers should treat the return value as a credential.

const IPC_AUTH_ACCOUNT: &str = "ipc-auth-token";

fn ensure_ipc_token() -> Result<String, String> {
    let entry =
        Entry::new(SERVICE_NAME, IPC_AUTH_ACCOUNT).map_err(|e| format!("Keyring error: {}", e))?;

    match entry.get_password() {
        Ok(token) if !token.is_empty() => Ok(token),
        _ => {
            use rand::RngCore;
            let mut bytes = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut bytes);
            let token = hex::encode(bytes);
            entry
                .set_password(&token)
                .map_err(|e| format!("Failed to persist IPC token: {}", e))?;
            log::info!("Provisioned a fresh IPC auth token in the OS keychain");
            Ok(token)
        }
    }
}

#[tauri::command]
fn get_ipc_auth_token() -> Result<String, String> {
    ensure_ipc_token()
}
