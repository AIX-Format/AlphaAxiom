import { create } from 'zustand';

export interface Trade {
    id: string;
    symbol: string;
    side: 'buy' | 'sell';
    amount: number;
    price: number;
    pnl: number;
    timestamp: number;
}

export interface Portfolio {
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
}

interface AppState {
    // Engine State
    tradingActive: boolean;
    connected: boolean;
    skillsLoaded: number;
    uptime: number;

    // Portfolio State
    portfolio: Portfolio;
    trades: Trade[];

    // UI State
    loading: boolean;
    error: string | null;
    shadowReport: {
        drift_rate: number;
        error_rate: number;
        compared: number;
        accepted: boolean;
    };

    // Actions
    setTradingActive: (active: boolean) => void;
    setPortfolio: (portfolio: Portfolio) => void;
    setEngineStatus: (status: EngineStatus) => void;
    addTrade: (trade: Trade) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    setShadowReport: (report: AppState['shadowReport']) => void;
}

export const useAppStore = create<AppState>((set) => ({
    // Initial State
    tradingActive: false,
    connected: false,
    skillsLoaded: 0,
    uptime: 0,

    portfolio: {
        balance: 10000,
        positions: {},
        pnl: 0,
        timestamp: Date.now() / 1000,
    },
    trades: [],

    loading: false,
    error: null,
    shadowReport: {
        drift_rate: 0,
        error_rate: 0,
        compared: 0,
        accepted: false,
    },

    // Actions
    setTradingActive: (active) => set({ tradingActive: active }),

    setPortfolio: (portfolio) => set({ portfolio }),

    setEngineStatus: (status) => set({
        tradingActive: status.trading_active,
        connected: status.connected,
        skillsLoaded: status.skills_loaded,
        uptime: status.uptime_seconds,
    }),

    addTrade: (trade) => set((state) => ({
        trades: [trade, ...state.trades].slice(0, 50), // Keep last 50 trades
    })),

    setLoading: (loading) => set({ loading }),

    setError: (error) => set({ error }),
    setShadowReport: (report) => set({ shadowReport: report }),
}));
