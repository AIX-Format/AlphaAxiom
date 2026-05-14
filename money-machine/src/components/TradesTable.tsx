'use client';

import { motion } from 'framer-motion';
import { useAppStore, Trade } from '@/store/useAppStore';

/**
 * Render a list of recent trades with animated rows and an empty-state for shadow mode.
 *
 * This component reads `trades` from the application store and displays each trade with side badge,
 * symbol, amount/price, PnL (colored by sign), and a formatted time. If there are no trades, it
 * shows a dashed empty-state card indicating that shadow mode is waiting for verified signals.
 *
 * @returns A React element displaying recent trades or an empty-state card when no trades are available.
 */
export function TradesTable() {
    const { trades } = useAppStore();

    const recentTrades: Trade[] = trades;

    const formatTime = (timestamp: number) => {
        return new Date(timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="glass-card p-6"
        >
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-[var(--text-secondary)]">
                    Recent Trades
                </h3>
                <span className="text-xs text-[var(--text-muted)]">
                    Last 24 hours
                </span>
            </div>

            {recentTrades.length === 0 ? (
                <div className="rounded-xl border border-dashed border-white/10 bg-white/[0.02] px-6 py-12 text-center">
                    <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-[var(--profit-green)]/10 text-[var(--profit-green)]">
                        S
                    </div>
                    <div className="font-medium text-[var(--text-secondary)]">Shadow mode is waiting for verified signals</div>
                    <div className="mt-1 text-xs text-[var(--text-muted)]">
                        No live or simulated trades are shown until the engine reports real data.
                    </div>
                </div>
            ) : (
                <div className="space-y-3">
                    {recentTrades.map((trade, index) => (
                        <motion.div
                            key={trade.id}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.1 }}
                            className="glass-card-subtle p-4 flex items-center justify-between"
                        >
                            <div className="flex items-center gap-4">
                                <div className={`px-2 py-1 rounded text-xs font-semibold ${trade.side === 'buy'
                                        ? 'bg-[var(--profit-green)]/20 text-[var(--profit-green)]'
                                        : 'bg-[var(--loss-red)]/20 text-[var(--loss-red)]'
                                    }`}>
                                    {trade.side.toUpperCase()}
                                </div>
                                <div>
                                    <div className="font-medium">{trade.symbol}</div>
                                    <div className="text-xs text-[var(--text-muted)]">
                                        {trade.amount} @ ${trade.price.toLocaleString()}
                                    </div>
                                </div>
                            </div>

                            <div className="text-right">
                                <div className={`font-semibold ${trade.pnl >= 0 ? 'text-[var(--profit-green)]' : 'text-[var(--loss-red)]'
                                    }`}>
                                    {trade.pnl >= 0 ? '+' : ''}{trade.pnl.toFixed(2)}
                                </div>
                                <div className="text-xs text-[var(--text-muted)]">
                                    {formatTime(trade.timestamp)}
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}
        </motion.div>
    );
}
