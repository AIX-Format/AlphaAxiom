"""
Signal Generator - AI-powered trading signal generation using Google Gemini API
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os

from engine.strategies.base import TradingSignal

logger = logging.getLogger(__name__)


class MarketContext:
    """Manages market data context window for Gemini"""
    
    def __init__(self, max_candles: int = 100):
        self.max_candles = max_candles
        self.data_cache: Dict[str, List] = {}
    
    def add_market_data(self, symbol: str, ohlcv: List[List]) -> None:
        """Add OHLCV data to context cache"""
        self.data_cache[symbol] = ohlcv[-self.max_candles:]
    
    def get_context_string(self, symbol: str) -> str:
        """Format market data for Gemini context window"""
        if symbol not in self.data_cache:
            return "No market data available"
        
        data = self.data_cache[symbol]
        if not data:
            return "No market data available"
        
        # Get last N candles
        recent_data = data[-20:]
        
        # Format as readable table
        lines = [f"## {symbol} Market Data (Last {len(recent_data)} candles)"]
        lines.append("| Time | Open | High | Low | Close | Volume |")
        lines.append("|------|------|------|-----|-------|--------|")
        
        for candle in recent_data:
            timestamp, open_p, high, low, close, volume = candle[:6]
            time_str = datetime.fromtimestamp(timestamp / 1000).strftime('%H:%M')
            lines.append(f"| {time_str} | {open_p:.2f} | {high:.2f} | {low:.2f} | {close:.2f} | {volume:.0f} |")
        
        # Add summary statistics
        closes = [c[4] for c in recent_data]
        current_price = closes[-1]
        price_change = ((current_price - closes[0]) / closes[0]) * 100
        avg_volume = sum(c[5] for c in recent_data) / len(recent_data)
        
        lines.append(f"\n**Current Price:** ${current_price:.2f}")
        lines.append(f"**Period Change:** {price_change:+.2f}%")
        lines.append(f"**Avg Volume:** {avg_volume:.0f}")
        
        return "\n".join(lines)


class SignalGenerator:
    """AI-powered signal generator using Google Gemini API"""
    
    SYSTEM_PROMPT = """You are an expert quantitative trader for the Money Machine trading system.
Your role is to analyze market data and generate precise trading signals.

## Your Trading Philosophy
- Risk management is paramount - never risk more than 2% per trade
- Look for high-probability setups with clear risk/reward
- Be patient - HOLD is a valid decision when conditions are unclear
- Consider market structure, momentum, and volume

## Response Format
You MUST respond with a valid JSON object containing:
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "entry_price": number or null,
    "stop_loss": number or null,
    "take_profit": number or null,
    "amount_pct": 0.0-0.02 (max 2% of portfolio),
    "reasoning": "brief explanation"
}

## Rules
- Only set confidence > 0.7 for very clear setups
- For HOLD, set entry_price, stop_loss, take_profit to null
- Reasoning should be 1-2 sentences maximum
"""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.model = None
        self.context = MarketContext()
        self.last_signals: Dict[str, TradingSignal] = {}
        
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                # Use Gemini 2.0 Flash or 1.5 Flash (Generic fallback)
                # Note: 'gemini-2.0-flash-exp' is the latest if available, else 'gemini-1.5-flash'
                model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
                
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=self.SYSTEM_PROMPT,
                    generation_config={"response_mime_type": "application/json"}
                )
                logger.info(f"✅ Gemini API initialized for SignalGenerator using {model_name}")
            except ImportError:
                logger.warning("Google Generative AI SDK not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
    
    async def generate_signal(
        self,
        symbol: str,
        market_data: List[List],
        portfolio_balance: float = 10000.0,
        additional_context: str = ""
    ) -> TradingSignal:
        """Generate a trading signal for the given symbol"""
        
        # Update context with market data
        self.context.add_market_data(symbol, market_data)
        
        if not self.model:
            # Fallback to rule-based signal
            return self._generate_rule_based_signal(symbol, market_data)
        
        try:
            # Build user message
            market_context = self.context.get_context_string(symbol)
            
            user_message = f"""Analyze the following market data and generate a trading signal.

{market_context}

**Portfolio Balance:** ${portfolio_balance:.2f}
**Max Risk Per Trade:** 2%

{additional_context}

Generate a trading signal based on this data."""

            # Call Gemini API
            # Note: run_in_executor needed because Gemini SDK is synchronous
            start_time = datetime.now()
            response = await asyncio.to_thread(
                self.model.generate_content, 
                user_message
            )
            
            # Parse response
            response_text = response.text
            signal = self._parse_json_response(symbol, response_text, market_data)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug(f"Gemini generation took {latency:.0f}ms")
            
            # Cache the signal
            self.last_signals[symbol] = signal
            
            return signal
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_rule_based_signal(symbol, market_data)
    
    def _parse_json_response(
        self,
        symbol: str,
        response_text: str,
        market_data: List[List]
    ) -> TradingSignal:
        """Parse JSON response into a TradingSignal"""
        try:
            data = json.loads(response_text)
            
            current_price = market_data[-1][4] if market_data else 0
            
            return TradingSignal(
                symbol=symbol,
                action=data.get('action', 'HOLD').upper(),
                confidence=float(data.get('confidence', 0.5)),
                entry_price=data.get('entry_price') or current_price,
                stop_loss=data.get('stop_loss'),
                take_profit=data.get('take_profit'),
                amount=data.get('amount_pct'),
                reasoning=data.get('reasoning', '')
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse Gemini response: {e}. Raw: {response_text[:100]}...")
        
        # Fallback
        return TradingSignal(
            symbol=symbol,
            action='HOLD',
            confidence=0.3,
            reasoning=f"Could not parse AI response"
        )
    
    def _generate_rule_based_signal(
        self,
        symbol: str,
        market_data: List[List]
    ) -> TradingSignal:
        """Generate a simple rule-based signal (fallback when AI unavailable)"""
        
        if not market_data or len(market_data) < 20:
            return TradingSignal(
                symbol=symbol,
                action='HOLD',
                confidence=0.0,
                reasoning='Insufficient market data'
            )
        
        # Simple momentum-based signal
        closes = [c[4] for c in market_data[-20:]]
        current = closes[-1]
        sma_20 = sum(closes) / len(closes)
        
        # Calculate momentum
        momentum = ((current - closes[0]) / closes[0]) * 100
        
        # Generate signal based on simple rules
        if current > sma_20 * 1.02 and momentum > 2:
            action = 'SELL'  # Overbought
            confidence = min(0.6, abs(momentum) / 10)
        elif current < sma_20 * 0.98 and momentum < -2:
            action = 'BUY'  # Oversold
            confidence = min(0.6, abs(momentum) / 10)
        else:
            action = 'HOLD'
            confidence = 0.5
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=current,
            stop_loss=current * 0.98 if action == 'BUY' else current * 1.02 if action == 'SELL' else None,
            take_profit=current * 1.04 if action == 'BUY' else current * 0.96 if action == 'SELL' else None,
            reasoning=f"Rule-based: Price {'above' if current > sma_20 else 'below'} SMA20, Momentum: {momentum:.1f}%"
        )
    
    def get_last_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Get the last generated signal for a symbol"""
        return self.last_signals.get(symbol)
