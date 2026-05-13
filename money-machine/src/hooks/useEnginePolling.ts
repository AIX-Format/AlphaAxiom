'use client';

import { useEffect, useCallback } from 'react';
import { useAppStore } from '@/store/useAppStore';
import { getPortfolio, getShadowReport, getStatus, ping } from '@/lib/tauri';

/**
 * Hook to poll engine data at regular intervals
 */
export function useEnginePolling(intervalMs: number = 5000) {
    const { setPortfolio, setEngineStatus, setError, setShadowReport } = useAppStore((state) => ({
        setPortfolio: state.setPortfolio,
        setEngineStatus: state.setEngineStatus,
        setError: state.setError,
        setShadowReport: state.setShadowReport,
    }));

    const fetchData = useCallback(async () => {
        try {
            // Health check
            const isAlive = await ping();
            if (!isAlive) {
                setError('Engine not responding');
                return;
            }

            // Fetch status
            const status = await getStatus();
            setEngineStatus(status);

            // Fetch portfolio
            const portfolio = await getPortfolio();
            setPortfolio(portfolio);
            const shadow = await getShadowReport(60);
            setShadowReport({
                drift_rate: shadow.drift_rate,
                error_rate: shadow.error_rate,
                compared: shadow.compared,
                accepted: shadow.accepted,
            });

            setError(null);
        } catch (err) {
            console.error('Engine polling error:', err);
            setError(err instanceof Error ? err.message : 'Unknown error');
        }
    }, [setPortfolio, setEngineStatus, setError, setShadowReport]);

    useEffect(() => {
        // Initial fetch
        fetchData();

        // Set up polling interval
        const interval = setInterval(fetchData, intervalMs);

        return () => clearInterval(interval);
    }, [fetchData, intervalMs]);
}
