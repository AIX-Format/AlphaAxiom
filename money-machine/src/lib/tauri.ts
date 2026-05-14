/**
 * Tauri API wrapper for Money Machine
 * Handles communication with Rust backend and Python sidecar
 */

import { invoke } from '@tauri-apps/api/core';

// Types
export interface PortfolioData {
    balance: number;
    positions: Record<string, any>;
    pnl: number;
    timestamp: number;
}

export interface EngineStatus {
    trading_active: boolean;
    connected: boolean;
    skills_loaded: number;
    uptime_seconds: number;
    ai_enabled?: boolean;
}

export interface ShadowReport {
    window_minutes: number;
    compared: number;
    drift_rate: number;
    error_rate: number;
    match_rate?: number;
    accepted: boolean;
}

// Check if running in Tauri
export const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

/**
 * Send IPC command to Python engine via TCP
 */
async function sendIPCCommand<T>(command: string, payload: Record<string, any> = {}): Promise<T> {
    if (isTauri) {
        return await invoke(command.toLowerCase(), payload);
    }

    // Development fallback: direct API route
    const response = await fetch('/api/ipc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, payload }),
    });

    const data = await response.json();
    if (data.error) {
        throw new Error(data.error);
    }
    return data.result;
}

// ============================================================
// Trading Commands
// ============================================================

export async function startTrading(): Promise<boolean> {
    const result = await sendIPCCommand<{ status: string }>('START_TRADING');
    return result.status === 'trading_active';
}

export async function stopTrading(): Promise<boolean> {
    const result = await sendIPCCommand<{ status: string }>('STOP_TRADING');
    return result.status === 'trading_stopped';
}

export async function getPortfolio(): Promise<PortfolioData> {
    return await sendIPCCommand<PortfolioData>('GET_PORTFOLIO');
}

export async function getStatus(): Promise<EngineStatus> {
    return await sendIPCCommand<EngineStatus>('GET_STATUS');
}

export async function ping(): Promise<boolean> {
    try {
        const result = await sendIPCCommand<{ status: string }>('PING');
        return result.status === 'pong';
    } catch {
        return false;
    }
}

export async function executeSkill(skillName: string, params: Record<string, any> = {}): Promise<any> {
    return await sendIPCCommand('EXECUTE_SKILL', { skill: skillName, params });
}

export async function getShadowReport(windowMinutes: number = 60): Promise<ShadowReport> {
    return await sendIPCCommand<ShadowReport>('GET_SHADOW_REPORT', { window_minutes: windowMinutes });
}

// ============================================================
// Window Commands (Wispr Flow UX)
// ============================================================

export async function setIgnoreMouseEvents(ignore: boolean): Promise<void> {
    if (isTauri) {
        await invoke('set_ignore_mouse_events', { ignore });
    }
}

export async function setAlwaysOnTop(state: boolean): Promise<void> {
    if (isTauri) {
        await invoke('set_always_on_top', { state });
    }
}

// ============================================================
// OS Keep-Alive (Prevent sleep during trading)
// ============================================================

export async function enableKeepAlive(): Promise<string> {
    if (!isTauri) return 'Not in Tauri environment';
    return await invoke('enable_keep_alive');
}

export async function disableKeepAlive(): Promise<string> {
    if (!isTauri) return 'Not in Tauri environment';
    return await invoke('disable_keep_alive');
}

// ============================================================
// Secure API Key Storage (OS Keychain)
// ============================================================

export async function storeApiKey(keyName: string, keyValue: string): Promise<string> {
    if (!isTauri) return 'Not in Tauri environment';
    return await invoke('store_api_key', { keyName, keyValue });
}

export async function getApiKey(keyName: string): Promise<string> {
    if (!isTauri) throw new Error('Not in Tauri environment');
    return await invoke('get_api_key', { keyName });
}

export async function deleteApiKey(keyName: string): Promise<string> {
    if (!isTauri) return 'Not in Tauri environment';
    return await invoke('delete_api_key', { keyName });
}
