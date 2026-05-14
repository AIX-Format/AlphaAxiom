'use client';

import { motion } from 'framer-motion';
import Image from 'next/image';
import { PnLWidget } from '@/components/PnLWidget';
import { StatusWidget } from '@/components/StatusWidget';
import { ControlPanel } from '@/components/ControlPanel';
import { TradesTable } from '@/components/TradesTable';
import { useEnginePolling } from '@/hooks/useEnginePolling';

export default function Dashboard() {
  useEnginePolling(5000);
  return (
    <div className="min-h-screen overflow-hidden bg-[var(--bg-primary)] p-6">
      <div className="fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,rgba(0,255,136,0.14),transparent_36%),radial-gradient(circle_at_80%_20%,rgba(59,130,246,0.16),transparent_30%),linear-gradient(135deg,#05070b,#101320_52%,#05070b)]" />
      <motion.header initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8 cursor-move" data-tauri-drag-region>
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-4 pointer-events-none">
            <div className="relative h-16 w-16 overflow-hidden rounded-2xl border border-[var(--profit-green)]/30 bg-black/30 shadow-[0_0_30px_rgba(0,255,136,0.25)]">
              <Image src="/images/logo.png" alt="Money Machine" fill className="object-cover" priority />
            </div>
            <div>
              <div className="mb-2 flex items-center gap-2">
                <span className="status-pill status-pill-online">Shadow Safe</span>
                <span className="status-pill">AQT Desktop</span>
              </div>
              <h1 className="text-3xl font-bold tracking-tight">Money Machine</h1>
              <p className="text-[var(--text-secondary)]">Autonomous AI trading overlay for AlphaAxiom</p>
            </div>
          </div>
          <div className="hidden rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-right lg:block">
            <div className="text-xs uppercase tracking-[0.25em] text-[var(--text-muted)]">Latency</div>
            <div className="text-xl font-semibold text-[var(--profit-green)]">12ms</div>
          </div>
        </div>
      </motion.header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
        <div className="lg:col-span-2 space-y-6">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="hero-panel p-6">
            <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="mb-2 text-xs uppercase tracking-[0.35em] text-[var(--accent-blue)]">AlphaQuanTopology</p>
                <h2 className="max-w-xl text-2xl font-semibold">Trade signals with a calm cockpit, safer defaults, and clearer engine state.</h2>
              </div>
              <div className="grid grid-cols-3 gap-3 text-center">
                <div className="metric-tile"><span>Gen</span><strong>3</strong></div>
                <div className="metric-tile"><span>Risk</span><strong>2%</strong></div>
                <div className="metric-tile"><span>Mode</span><strong>Dry</strong></div>
              </div>
            </div>
          </motion.div>
          <PnLWidget />
          <TradesTable />
        </div>

        <div className="space-y-6">
          <ControlPanel />
          <StatusWidget />

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
                  <span className="skill-orb">A</span>
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
