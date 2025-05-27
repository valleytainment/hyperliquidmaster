"""
LLM-based Sentiment Analyzer for Cryptocurrency Trading

This module provides sentiment analysis capabilities using Large Language Models (LLMs)
to analyze news, social media, and market narratives for trading signals.

Features:
- News sentiment analysis
- Social media sentiment analysis
- Market narrative detection
- Sentiment-adjusted trading signals
- Confidence scoring for potential trades
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
import os
import requests
from datetime import datetime, timedelta

class LLMSentimentAnalyzer:
    """
    Sentiment analyzer using LLMs to process market-related text data
    and extract trading-relevant insights.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the LLM Sentiment Analyzer.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # API configuration
        self.api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
        self.model = config.get("sentiment_model", "gpt-3.5-turbo")
        self.use_local_llm = config.get("use_local_llm", False)
        self.local_llm_url = config.get("local_llm_url", "http://localhost:8000/v1")
        
        # Cache configuration
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = config.get("sentiment_cache_ttl", 3600)  # 1 hour default
        
        # Sentiment thresholds
        self.bullish_threshold = config.get("bullish_threshold", 0.6)
        self.bearish_threshold = config.get("bearish_threshold", 0.4)
        
        # Sentiment impact configuration
        self.sentiment_impact_weight = config.get("sentiment_impact_weight", 0.2)
        self.min_confidence_threshold = config.get("min_sentiment_confidence", 0.3)
        
        self.logger.info(f"LLM Sentiment Analyzer initialized with model: {self.model}")
        
    async def analyze_news(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a list of news items for market sentiment.
        
        Args:
            news_items: List of news items with 'title', 'content', and 'published_at' keys
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not news_items:
            return self._get_neutral_sentiment()
            
        # Filter recent news (last 24 hours)
        current_time = datetime.now()
        recent_news = [
            item for item in news_items 
            if 'published_at' not in item or 
            (datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')) 
             > current_time - timedelta(hours=24))
        ]
        
        if not recent_news:
            self.logger.info("No recent news found for sentiment analysis")
            return self._get_neutral_sentiment()
            
        # Create a concise summary of news for analysis
        news_summary = "\n".join([
            f"- {item.get('title', 'Untitled')}: {item.get('content', '')[:100]}..." 
            for item in recent_news[:10]
        ])
        
        # Check cache
        cache_key = hash(news_summary)
        current_time = time.time()
        
        if cache_key in self.cache and current_time < self.cache_expiry.get(cache_key, 0):
            self.logger.debug("Using cached news sentiment analysis")
            return self.cache[cache_key]
            
        # Prepare prompt for LLM
        prompt = f"""
        Analyze the following cryptocurrency news for market sentiment:
        
        {news_summary}
        
        Provide a JSON response with the following fields:
        - sentiment: "bullish", "bearish", or "neutral"
        - score: a number from 0 (extremely bearish) to 1 (extremely bullish)
        - confidence: a number from 0 to 1 indicating confidence in the assessment
        - key_factors: list of key factors influencing the sentiment
        - affected_coins: list of specific cryptocurrencies mentioned and how they might be affected
        """
        
        try:
            # Make API call to LLM
            result = await self._call_llm(prompt)
            
            # Cache result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = current_time + self.cache_ttl
            
            self.logger.info(f"News sentiment analysis: {result.get('sentiment')} (score: {result.get('score')}, confidence: {result.get('confidence')})")
            return result
        except Exception as e:
            self.logger.error(f"Error in news sentiment analysis: {str(e)}")
            return self._get_neutral_sentiment()
            
    async def analyze_social_media(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze social media posts for market sentiment.
        
        Args:
            posts: List of social media posts with 'text', 'platform', and 'timestamp' keys
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not posts:
            return self._get_neutral_sentiment()
            
        # Filter recent posts (last 12 hours)
        current_time = datetime.now()
        recent_posts = [
            post for post in posts 
            if 'timestamp' not in post or 
            (datetime.fromisoformat(post['timestamp'].replace('Z', '+00:00')) 
             > current_time - timedelta(hours=12))
        ]
        
        if not recent_posts:
            self.logger.info("No recent social media posts found for sentiment analysis")
            return self._get_neutral_sentiment()
            
        # Create a concise summary of posts for analysis
        posts_summary = "\n".join([
            f"- [{post.get('platform', 'social')}] {post.get('text', '')[:100]}..." 
            for post in recent_posts[:15]
        ])
        
        # Check cache
        cache_key = hash(posts_summary)
        current_time = time.time()
        
        if cache_key in self.cache and current_time < self.cache_expiry.get(cache_key, 0):
            self.logger.debug("Using cached social media sentiment analysis")
            return self.cache[cache_key]
            
        # Prepare prompt for LLM
        prompt = f"""
        Analyze the following cryptocurrency social media posts for market sentiment:
        
        {posts_summary}
        
        Provide a JSON response with the following fields:
        - sentiment: "bullish", "bearish", or "neutral"
        - score: a number from 0 (extremely bearish) to 1 (extremely bullish)
        - confidence: a number from 0 to 1 indicating confidence in the assessment
        - key_topics: list of key topics or coins mentioned
        - trending_coins: list of cryptocurrencies that appear to be trending
        - market_concerns: list of any concerns or risks mentioned
        """
        
        try:
            # Make API call to LLM
            result = await self._call_llm(prompt)
            
            # Cache result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = current_time + self.cache_ttl
            
            self.logger.info(f"Social media sentiment analysis: {result.get('sentiment')} (score: {result.get('score')}, confidence: {result.get('confidence')})")
            return result
        except Exception as e:
            self.logger.error(f"Error in social media sentiment analysis: {str(e)}")
            return self._get_neutral_sentiment()
            
    async def detect_market_narratives(self, news_items: List[Dict[str, Any]], posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect emerging market narratives by analyzing both news and social media.
        
        Args:
            news_items: List of news items
            posts: List of social media posts
            
        Returns:
            Dictionary with detected narratives and their potential impact
        """
        if not news_items and not posts:
            return {"narratives": [], "confidence": 0.0}
            
        # Combine recent news and posts
        news_summary = "\n".join([
            f"[NEWS] {item.get('title', 'Untitled')}" 
            for item in news_items[:5]
        ])
        
        posts_summary = "\n".join([
            f"[SOCIAL] {post.get('text', '')[:100]}..." 
            for post in posts[:10]
        ])
        
        combined_text = f"{news_summary}\n\n{posts_summary}"
        
        # Check cache
        cache_key = hash(combined_text)
        current_time = time.time()
        
        if cache_key in self.cache and current_time < self.cache_expiry.get(cache_key, 0):
            self.logger.debug("Using cached market narrative analysis")
            return self.cache[cache_key]
            
        # Prepare prompt for LLM
        prompt = f"""
        Analyze the following cryptocurrency news and social media posts to identify emerging market narratives:
        
        {combined_text}
        
        Provide a JSON response with the following fields:
        - narratives: list of detected narratives, each with:
          * theme: short description of the narrative
          * sentiment: "bullish", "bearish", or "neutral"
          * impact: potential market impact from 0 (minimal) to 1 (significant)
          * affected_assets: list of cryptocurrencies likely affected
        - confidence: overall confidence in the narrative detection from 0 to 1
        - market_regime: current market regime ("risk-on", "risk-off", "neutral", "uncertain")
        """
        
        try:
            # Make API call to LLM
            result = await self._call_llm(prompt)
            
            # Cache result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = current_time + self.cache_ttl
            
            self.logger.info(f"Market narrative detection: found {len(result.get('narratives', []))} narratives with {result.get('confidence')} confidence")
            return result
        except Exception as e:
            self.logger.error(f"Error in market narrative detection: {str(e)}")
            return {"narratives": [], "confidence": 0.0, "market_regime": "uncertain"}
            
    async def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze market sentiment for a specific trading symbol.
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        self.logger.info(f"Analyzing market sentiment for {symbol}")
        
        try:
            # In a real implementation, this would fetch recent news and social media posts
            # related to the symbol and analyze them. For now, we'll return a neutral sentiment.
            
            # Check cache
            cache_key = f"market_sentiment_{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
            current_time = time.time()
            
            if cache_key in self.cache and current_time < self.cache_expiry.get(cache_key, 0):
                self.logger.debug(f"Using cached market sentiment for {symbol}")
                return self.cache[cache_key]
            
            # For demonstration, return a neutral sentiment
            result = {
                "symbol": symbol,
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.3,
                "sources_analyzed": 0,
                "key_factors": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = current_time + self.cache_ttl
            
            self.logger.info(f"Market sentiment for {symbol}: {result['sentiment']} (score: {result['score']}, confidence: {result['confidence']})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.0,
                "sources_analyzed": 0,
                "key_factors": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        # Prepare technical data summary
        technical_summary = json.dumps(technical_data, indent=2)
        sentiment_summary = json.dumps(sentiment_data, indent=2)
        
        # Prepare prompt for LLM
        prompt = f"""
        Evaluate this potential {signal_type} trade for {symbol} based on technical and sentiment data.
        
        Technical Analysis:
        {technical_summary}
        
        Sentiment Analysis:
        {sentiment_summary}
        
        Provide a JSON response with the following fields:
        - recommendation: "strong_buy", "buy", "hold", "sell", "strong_sell"
        - confidence: a number from 0 to 1 indicating confidence in the recommendation
        - reasoning: detailed explanation for the recommendation
        - risk_assessment: evaluation of potential risks
        - suggested_entry: suggested entry price or range
        - suggested_stop_loss: suggested stop loss level
        - suggested_take_profit: suggested take profit level
        """
        
        try:
            # Make API call to LLM
            result = await self._call_llm(prompt)
            
            self.logger.info(f"Trade evaluation for {symbol} {signal_type}: {result.get('recommendation')} (confidence: {result.get('confidence')})")
            return result
        except Exception as e:
            self.logger.error(f"Error in trade opportunity evaluation: {str(e)}")
            return {
                "recommendation": "hold",
                "confidence": 0.0,
                "reasoning": "Error in evaluation process",
                "risk_assessment": "Unable to assess risk due to evaluation error"
            }
            
    def adjust_trading_signals(self, base_signal: Dict[str, Any], sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust trading signals based on sentiment analysis.
        
        Args:
            base_signal: Original trading signal
            sentiment_data: Sentiment analysis data
            
        Returns:
            Adjusted trading signal
        """
        if not sentiment_data or not base_signal:
            return base_signal
            
        sentiment_score = sentiment_data.get("score", 0.5)
        confidence = sentiment_data.get("confidence", 0.0)
        
        # Only adjust if confidence is reasonable
        if confidence < self.min_confidence_threshold:
            return base_signal
            
        # Calculate adjustment factor
        sentiment_factor = (sentiment_score - 0.5) * 2  # -1 to 1 range
        
        # Apply sentiment adjustment to signal confidence
        if base_signal["signal"] == "LONG" and sentiment_factor > 0:
            # Bullish sentiment reinforces long signal
            base_signal["confidence"] = min(1.0, base_signal["confidence"] + (sentiment_factor * confidence * self.sentiment_impact_weight))
        elif base_signal["signal"] == "LONG" and sentiment_factor < 0:
            # Bearish sentiment weakens long signal
            base_signal["confidence"] = max(0.0, base_signal["confidence"] + (sentiment_factor * confidence * self.sentiment_impact_weight))
        elif base_signal["signal"] == "SHORT" and sentiment_factor < 0:
            # Bearish sentiment reinforces short signal
            base_signal["confidence"] = min(1.0, base_signal["confidence"] - (sentiment_factor * confidence * self.sentiment_impact_weight))
        elif base_signal["signal"] == "SHORT" and sentiment_factor > 0:
            # Bullish sentiment weakens short signal
            base_signal["confidence"] = max(0.0, base_signal["confidence"] - (sentiment_factor * confidence * self.sentiment_impact_weight))
            
        # Add sentiment data to signal details
        if "details" not in base_signal:
            base_signal["details"] = {}
        base_signal["details"]["sentiment"] = {
            "score": sentiment_score,
            "confidence": confidence,
            "adjustment_applied": sentiment_factor * confidence * self.sentiment_impact_weight
        }
        
        return base_signal
        
    async def generate_market_summary(self, symbol: str, market_data: Dict[str, Any], 
                                     sentiment_data: Dict[str, Any]) -> str:
        """
        Generate a natural language summary of market conditions.
        
        Args:
            symbol: Trading symbol
            market_data: Market data
            sentiment_data: Sentiment analysis data
            
        Returns:
            Natural language summary
        """
        # Prepare market data summary
        market_summary = json.dumps(market_data, indent=2)
        sentiment_summary = json.dumps(sentiment_data, indent=2)
        
        # Prepare prompt for LLM
        prompt = f"""
        Generate a concise natural language summary of current market conditions for {symbol}.
        
        Market Data:
        {market_summary}
        
        Sentiment Analysis:
        {sentiment_summary}
        
        Provide a clear, concise summary that would be helpful for a trader, including:
        1. Current price action and trend
        2. Key technical levels
        3. Market sentiment
        4. Potential catalysts or risks
        5. Overall trading outlook
        
        Keep the summary under 200 words and focus on actionable insights.
        """
        
        try:
            # Make API call to LLM
            response = await self._call_llm_raw(prompt)
            
            # Extract the text response
            summary = response.strip()
            
            self.logger.info(f"Generated market summary for {symbol} ({len(summary)} chars)")
            return summary
        except Exception as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            return f"Unable to generate market summary for {symbol} due to an error."
            
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM API and parse JSON response.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Parsed JSON response
        """
        response_text = await self._call_llm_raw(prompt)
        
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                self.logger.warning(f"No valid JSON found in LLM response: {response_text[:100]}...")
                return self._get_neutral_sentiment()
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON from LLM response: {str(e)}")
            return self._get_neutral_sentiment()
            
    async def _call_llm_raw(self, prompt: str) -> str:
        """
        Call LLM API and get raw text response.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Raw text response
        """
        if self.use_local_llm:
            return await self._call_local_llm(prompt)
        else:
            return await self._call_openai(prompt)
            
    async def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Response text
        """
        if not self.api_key:
            self.logger.error("OpenAI API key not provided")
            return ""
            
        try:
            import openai
            openai.api_key = self.api_key
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency market analyst specializing in sentiment analysis and trading insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            return ""
            
    async def _call_local_llm(self, prompt: str) -> str:
        """
        Call local LLM API.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Response text
        """
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a cryptocurrency market analyst specializing in sentiment analysis and trading insights."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.local_llm_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                self.logger.error(f"Error from local LLM API: {response.status_code} {response.text}")
                return ""
        except Exception as e:
            self.logger.error(f"Error calling local LLM API: {str(e)}")
            return ""
            
    def _get_neutral_sentiment(self) -> Dict[str, Any]:
        """
        Get neutral sentiment result.
        
        Returns:
            Neutral sentiment dictionary
        """
        return {
            "sentiment": "neutral",
            "score": 0.5,
            "confidence": 0.0,
            "key_factors": [],
            "affected_coins": []
        }
