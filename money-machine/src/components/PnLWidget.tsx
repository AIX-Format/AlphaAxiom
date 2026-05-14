'use client';

import { motion } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';

/**
 * Render a styled P&L widget showing today's profit or loss, the portfolio balance, and the current mode.
 *
 * The widget conditionally styles its appearance based on whether the P&L is >= 0, formats numeric values to two decimal places, and includes entry/scale animations for the container and amount.
 *
 * @returns A JSX element displaying today's P&L (prefixing positive values with `+`), the portfolio balance, and the mode label.
 */
export function PnLWidget() {
    const { portfolio } = useAppStore();
    const pnl = portfolio.pnl;
    const isProfit = pnl >= 0;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={`glass-card p-6 ${isProfit ? 'profit-glow' : 'loss-glow'}`}
        >
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-[var(--text-secondary)]">
                    Today&apos;s P&amp;L
                </h3>
                <div className={`status-dot ${isProfit ? 'active' : 'error'}`} />
            </div>

            <motion.div
                key={pnl}
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                className="flex items-baseline gap-2"
            >
                <span className={`text-4xl font-bold ${isProfit ? 'text-[var(--profit-green)]' : 'text-[var(--loss-red)]'
                    }`}>
                    {isProfit ? '+' : ''}{pnl.toFixed(2)}
                </span>
                <span className="text-lg text-[var(--text-secondary)]">USD</span>
            </motion.div>

            <div className="mt-4 flex items-center gap-4 text-sm">
                <div>
                    <span className="text-[var(--text-muted)]">Balance: </span>
                    <span className="font-medium">${portfolio.balance.toFixed(2)}</span>
                </div>
                <div>
                    <span className="text-[var(--text-muted)]">Mode: </span>
                    <span className="font-medium text-[var(--accent-blue)]">Shadow</span>
                </div>
            </div>
        </motion.div>
    );
}
