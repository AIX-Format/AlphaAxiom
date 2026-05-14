'use client';

import { motion } from 'framer-motion';
import { PnLWidget } from '@/components/PnLWidget';
import { StatusWidget } from '@/components/StatusWidget';
import { ControlPanel } from '@/components/ControlPanel';
import { TradesTable } from '@/components/TradesTable';
import { useEnginePolling } from '@/hooks/useEnginePolling';

export default function Dashboard() {
  useEnginePolling(5000);
  return (
    <div className="min-h-screen bg-[var(--bg-primary)] p-6">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 cursor-move"
        data-tauri-drag-region
      >
        <div className="flex items-center gap-3 mb-2 pointer-events-none">
          <span className="text-3xl">💰</span>
          <h1 className="text-3xl font-bold tracking-tight">Money Machine</h1>
        </div>
        <p className="text-[var(--text-secondary)]">
          Autonomous AI Trading Agent
        </p>
      </motion.header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
        {/* Left Column - Main Stats */}
        <div className="lg:col-span-2 space-y-6">
          <PnLWidget />
          <TradesTable />
        </div>

        {/* Right Column - Control & Status */}
        <div className="space-y-6">
          <ControlPanel />
          <StatusWidget />

          {/* Skills Summary */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="glass-card p-6"
          >
            <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-4">
              Active Skills
            </h3>
            <div className="space-y-3">
              <div className="glass-card-subtle p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">🎯</span>
                  <div>
                    <div className="font-medium text-sm">Aggressive Scalper</div>
                    <div className="text-xs text-[var(--text-muted)]">v1.0.0</div>
                  </div>
                </div>
                <div className="status-dot active" />
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Footer */}
      <motion.footer
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-12 text-center text-sm text-[var(--text-muted)]"
      >
        Money Machine v0.1.0 • Part of the AlphaAxiom Ecosystem
      </motion.footer>
    </div>
  );
}
