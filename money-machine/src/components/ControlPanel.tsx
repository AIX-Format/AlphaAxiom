'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';
import { startTrading, stopTrading, setAlwaysOnTop, setIgnoreMouseEvents } from '@/lib/tauri';

export function ControlPanel() {
    const { tradingActive, setTradingActive, connected } = useAppStore();
    const [loading, setLoading] = useState(false);
    const [isAlwaysOnTop, setIsAlwaysOnTop] = useState(true);
    const [isGhostMode, setIsGhostMode] = useState(false);
    const tradingLabel = tradingActive ? 'Pause Shadow Trading' : 'Start Shadow Trading';

    const handleToggleTrading = async () => {
        if (!connected) return;

        setLoading(true);
        try {
            if (tradingActive) {
                const success = await stopTrading();
                if (success) setTradingActive(false);
            } else {
                const success = await startTrading();
                if (success) setTradingActive(true);
            }
        } catch (error) {
            console.error('Failed to toggle trading:', error);
        } finally {
            setLoading(false);
        }
    };

    const toggleAlwaysOnTop = async () => {
        const newState = !isAlwaysOnTop;
        await setAlwaysOnTop(newState);
        setIsAlwaysOnTop(newState);
    };

    const toggleGhostMode = async () => {
        const newState = !isGhostMode;
        await setIgnoreMouseEvents(newState);
        setIsGhostMode(newState);
    };

    return (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.2 }} className="glass-card p-6">
            <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4 flex justify-between items-center">
                <span>Mission Control</span>
                <span className={`status-pill ${connected ? 'status-pill-online' : 'status-pill-offline'}`}>
                    {connected ? 'ENGINE LIVE' : 'ENGINE OFFLINE'}
                </span>
            </h3>

            {!connected && (
                <div className="mb-4 rounded-xl border border-amber-400/20 bg-amber-400/10 p-3 text-xs text-amber-100">
                    Shadow dashboard is ready. Start the Python engine to unlock live controls.
                </div>
            )}

            <div className="space-y-4">
                {/* Main Trading Button */}
                <button
                    onClick={handleToggleTrading}
                    disabled={!connected || loading}
                    className={`w-full py-4 px-6 rounded-xl font-semibold text-lg transition-all duration-300 ${tradingActive
                        ? 'btn-danger'
                        : 'btn-primary'
                        } ${(!connected || loading) && 'opacity-50 cursor-not-allowed'}`}
                >
                    {loading ? (
                        <span className="flex items-center justify-center gap-2">
                            Processing...
                        </span>
                    ) : tradingActive ? (
                        'Pause Shadow Trading'
                    ) : (
                        'Start Shadow Trading'
                    )}
                </button>
                <div className="rounded-xl border border-[var(--accent-blue)]/20 bg-[var(--accent-blue)]/10 px-3 py-2 text-xs text-[var(--text-secondary)]">
                    {tradingLabel} only toggles supervised shadow mode until a reviewed live adapter is connected.
                </div>

                {/* Window Controls (Wispr Flow) */}
                <div className="grid grid-cols-2 gap-3">
                    <button
                        onClick={toggleAlwaysOnTop}
                        className={`glass-card-subtle py-2 px-3 text-xs font-medium transition-colors flex items-center justify-center gap-2 ${isAlwaysOnTop ? 'text-[var(--profit-green)] bg-[var(--profit-green)]/10' : 'text-[var(--text-secondary)]'
                            }`}
                    >
                        {isAlwaysOnTop ? 'Pinned' : 'Floating'}
                    </button>
                    <button
                        onClick={toggleGhostMode}
                        title="Click-Through Mode (Exit via Tray)"
                        className={`glass-card-subtle py-2 px-3 text-xs font-medium transition-colors flex items-center justify-center gap-2 ${isGhostMode ? 'text-[var(--accent-blue)] bg-[var(--accent-blue)]/10' : 'text-[var(--text-secondary)]'
                            }`}
                    >
                        {isGhostMode ? 'Ghost' : 'Interactive'}
                    </button>
                </div>

                {/* Quick Actions */}
                <div className="grid grid-cols-2 gap-3">
                    <button disabled={!connected} className="glass-card-subtle py-3 px-4 text-sm hover:bg-white/5 disabled:opacity-50 text-[var(--text-secondary)]">
                        History
                    </button>
                    <button disabled={!connected} className="glass-card-subtle py-3 px-4 text-sm hover:bg-white/5 disabled:opacity-50 text-[var(--text-secondary)]">
                        Config
                    </button>
                </div>
            </div>
        </motion.div>
    );
}
