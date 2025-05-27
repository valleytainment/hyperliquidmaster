"""
Sentiment Analysis Integration for HyperLiquid Trading Bot

This module provides advanced sentiment analysis capabilities by integrating
with news sources, social media, and LLM-based analysis to enhance trading decisions.
"""

import os
import time
import logging
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import aiohttp
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Try to import OpenAI, but don't fail if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import local LLM support, but don't fail if not available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

class SentimentAnalyzer:
    """
    Advanced sentiment analyzer with multiple backends and data sources.
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Initialize sentiment analyzer.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or logging.getLogger("SentimentAnalyzer")
        
        # Store configuration
        self.config = config
        
        # Initialize NLTK
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            self.logger.info("NLTK sentiment analyzer initialized")
        except Exception as e:
            self.logger.error(f"Error initializing NLTK: {e}")
            self.sia = None
        
        # Initialize OpenAI if available
        self.openai_client = None
        if OPENAI_AVAILABLE and config.get("openai_api_key"):
            try:
                openai.api_key = config["openai_api_key"]
                self.openai_client = openai.Client(api_key=config["openai_api_key"])
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}")
        
        # Initialize local LLM if available
        self.local_llm = None
        if LLAMA_CPP_AVAILABLE and config.get("local_llm_path"):
            try:
                model_path = config["local_llm_path"]
                if os.path.exists(model_path):
                    self.local_llm = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_threads=config.get("local_llm_threads", 4)
                    )
                    self.logger.info(f"Local LLM initialized from {model_path}")
                else:
                    self.logger.warning(f"Local LLM model not found: {model_path}")
            except Exception as e:
                self.logger.error(f"Error initializing local LLM: {e}")
        
        # Initialize HTTP session
        self.session = None
        
        # Cache for sentiment results
        self.sentiment_cache = {}
        self.cache_ttl = config.get("sentiment_cache_ttl", 3600)  # 1 hour default
        
        self.logger.info("Sentiment analyzer initialized")
    
    async def initialize(self):
        """Initialize async resources."""
        self.session = aiohttp.ClientSession()
        self.logger.info("Async HTTP session initialized")
    
    async def close(self):
        """Close async resources."""
        if self.session:
            await self.session.close()
            self.logger.info("Async HTTP session closed")
    
    async def analyze_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze sentiment for a symbol.
        
        Args:
            symbol: Symbol to analyze
            lookback_hours: Hours to look back for news and social media
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Check cache first
        cache_key = f"{symbol}_{lookback_hours}"
        if cache_key in self.sentiment_cache:
            cache_entry = self.sentiment_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                self.logger.info(f"Using cached sentiment for {symbol}")
                return cache_entry["data"]
        
        self.logger.info(f"Analyzing sentiment for {symbol} (lookback: {lookback_hours}h)")
        
        # Initialize result
        result = {
            "symbol": symbol,
            "timestamp": time.time(),
            "lookback_hours": lookback_hours,
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "confidence": 0.0,
            "sources": {},
            "keywords": {},
            "summary": ""
        }
        
        try:
            # Gather data from multiple sources in parallel
            news_task = asyncio.create_task(self._fetch_news(symbol, lookback_hours))
            social_task = asyncio.create_task(self._fetch_social_media(symbol, lookback_hours))
            
            # Wait for all tasks to complete
            news_data = await news_task
            social_data = await social_task
            
            # Combine all text for analysis
            all_texts = []
            all_texts.extend(news_data.get("articles", []))
            all_texts.extend(social_data.get("posts", []))
            
            # If no data found, return neutral sentiment
            if not all_texts:
                self.logger.warning(f"No sentiment data found for {symbol}")
                result["summary"] = f"No sentiment data found for {symbol} in the last {lookback_hours} hours."
                return result
            
            # Analyze sentiment using multiple methods
            nltk_sentiment = await self._analyze_with_nltk(all_texts)
            textblob_sentiment = await self._analyze_with_textblob(all_texts)
            
            # Try to use LLM for advanced analysis if available
            llm_sentiment = None
            if self.openai_client or self.local_llm:
                llm_sentiment = await self._analyze_with_llm(symbol, all_texts)
            
            # Combine sentiment scores with weights
            weights = {
                "nltk": 0.3,
                "textblob": 0.3,
                "llm": 0.4
            }
            
            sentiment_scores = []
            confidence_scores = []
            
            if nltk_sentiment:
                sentiment_scores.append(nltk_sentiment["score"] * weights["nltk"])
                confidence_scores.append(nltk_sentiment["confidence"] * weights["nltk"])
                result["sources"]["nltk"] = nltk_sentiment
            
            if textblob_sentiment:
                sentiment_scores.append(textblob_sentiment["score"] * weights["textblob"])
                confidence_scores.append(textblob_sentiment["confidence"] * weights["textblob"])
                result["sources"]["textblob"] = textblob_sentiment
            
            if llm_sentiment:
                sentiment_scores.append(llm_sentiment["score"] * weights["llm"])
                confidence_scores.append(llm_sentiment["confidence"] * weights["llm"])
                result["sources"]["llm"] = llm_sentiment
                
                # Use LLM summary if available
                if "summary" in llm_sentiment:
                    result["summary"] = llm_sentiment["summary"]
                
                # Use LLM keywords if available
                if "keywords" in llm_sentiment:
                    result["keywords"] = llm_sentiment["keywords"]
            
            # Calculate weighted average
            if sentiment_scores:
                result["sentiment_score"] = sum(sentiment_scores) / sum(weights.values())
                result["confidence"] = sum(confidence_scores) / sum(weights.values())
                
                # Determine sentiment label
                if result["sentiment_score"] > 0.2:
                    result["sentiment_label"] = "bullish"
                elif result["sentiment_score"] < -0.2:
                    result["sentiment_label"] = "bearish"
                else:
                    result["sentiment_label"] = "neutral"
            
            # Add source data
            result["sources"]["news"] = news_data
            result["sources"]["social"] = social_data
            
            # Generate summary if not already set by LLM
            if not result["summary"]:
                result["summary"] = self._generate_summary(symbol, result)
            
            # Cache result
            self.sentiment_cache[cache_key] = {
                "timestamp": time.time(),
                "data": result
            }
            
            self.logger.info(f"Sentiment analysis for {symbol}: {result['sentiment_label']} (score: {result['sentiment_score']:.2f}, confidence: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            result["summary"] = f"Error analyzing sentiment: {str(e)}"
            return result
    
    async def _fetch_news(self, symbol: str, lookback_hours: int) -> Dict[str, Any]:
        """
        Fetch news articles for a symbol.
        
        Args:
            symbol: Symbol to fetch news for
            lookback_hours: Hours to look back
            
        Returns:
            Dictionary with news data
        """
        self.logger.info(f"Fetching news for {symbol}")
        
        result = {
            "source": "news",
            "articles": [],
            "article_count": 0,
            "sources": []
        }
        
        try:
            # Mock implementation - in a real system, this would call news APIs
            # such as Alpha Vantage News API, NewsAPI, or Bloomberg
            
            # Simulate some news articles
            crypto_terms = {
                "BTC": ["Bitcoin", "BTC"],
                "ETH": ["Ethereum", "ETH"],
                "SOL": ["Solana", "SOL"],
                "DOGE": ["Dogecoin", "DOGE"],
                "XRP": ["Ripple", "XRP"]
            }
            
            terms = crypto_terms.get(symbol, [symbol])
            
            # Generate mock articles
            articles = []
            for term in terms:
                articles.append(f"Analysts predict {term} will see increased adoption in coming months.")
                articles.append(f"Market volatility continues to affect {term} price action.")
                articles.append(f"New developments in {term} ecosystem show promise for future growth.")
            
            result["articles"] = articles
            result["article_count"] = len(articles)
            result["sources"] = ["Mock News Source"]
            
            self.logger.info(f"Found {len(articles)} news articles for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
            return result
    
    async def _fetch_social_media(self, symbol: str, lookback_hours: int) -> Dict[str, Any]:
        """
        Fetch social media posts for a symbol.
        
        Args:
            symbol: Symbol to fetch posts for
            lookback_hours: Hours to look back
            
        Returns:
            Dictionary with social media data
        """
        self.logger.info(f"Fetching social media for {symbol}")
        
        result = {
            "source": "social",
            "posts": [],
            "post_count": 0,
            "sources": []
        }
        
        try:
            # Mock implementation - in a real system, this would call social media APIs
            # such as Twitter API, Reddit API, or StockTwits API
            
            # Simulate some social media posts
            crypto_terms = {
                "BTC": ["Bitcoin", "BTC", "#Bitcoin", "#BTC"],
                "ETH": ["Ethereum", "ETH", "#Ethereum", "#ETH"],
                "SOL": ["Solana", "SOL", "#Solana", "#SOL"],
                "DOGE": ["Dogecoin", "DOGE", "#Dogecoin", "#DOGE"],
                "XRP": ["Ripple", "XRP", "#Ripple", "#XRP"]
            }
            
            terms = crypto_terms.get(symbol, [symbol, f"#{symbol}"])
            
            # Generate mock posts
            posts = []
            for term in terms:
                posts.append(f"Just bought more {term}! Feeling bullish about the future.")
                posts.append(f"Not sure about {term} price action today. Market seems uncertain.")
                posts.append(f"{term} looking strong despite market conditions. Holding long term.")
            
            result["posts"] = posts
            result["post_count"] = len(posts)
            result["sources"] = ["Mock Social Media"]
            
            self.logger.info(f"Found {len(posts)} social media posts for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching social media for {symbol}: {e}")
            return result
    
    async def _analyze_with_nltk(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment using NLTK.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.sia:
            return None
            
        try:
            # Analyze each text
            scores = []
            for text in texts:
                sentiment = self.sia.polarity_scores(text)
                # Convert to -1 to 1 scale
                score = sentiment["compound"]
                scores.append(score)
            
            # Calculate average score
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Calculate confidence based on score distribution
            if scores:
                variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                # Higher variance means lower confidence
                confidence = max(0.0, 1.0 - min(1.0, variance * 2))
            else:
                confidence = 0.0
            
            return {
                "method": "nltk",
                "score": avg_score,
                "confidence": confidence,
                "raw_scores": scores
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing with NLTK: {e}")
            return None
    
    async def _analyze_with_textblob(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Analyze each text
            polarities = []
            subjectivities = []
            for text in texts:
                blob = TextBlob(text)
                polarities.append(blob.sentiment.polarity)
                subjectivities.append(blob.sentiment.subjectivity)
            
            # Calculate average polarity
            avg_polarity = sum(polarities) / len(polarities) if polarities else 0
            
            # Calculate average subjectivity
            avg_subjectivity = sum(subjectivities) / len(subjectivities) if subjectivities else 0
            
            # Calculate confidence based on subjectivity
            # Higher subjectivity means higher confidence
            confidence = avg_subjectivity
            
            return {
                "method": "textblob",
                "score": avg_polarity,
                "confidence": confidence,
                "subjectivity": avg_subjectivity,
                "raw_polarities": polarities,
                "raw_subjectivities": subjectivities
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing with TextBlob: {e}")
            return None
    
    async def _analyze_with_llm(self, symbol: str, texts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment using LLM (OpenAI or local).
        
        Args:
            symbol: Symbol being analyzed
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Combine texts, but limit to avoid token limits
        combined_text = "\n\n".join(texts[:20])  # Limit to 20 texts
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."
        
        prompt = f"""
        Analyze the sentiment of the following texts about {symbol} cryptocurrency.
        
        Texts:
        {combined_text}
        
        Please provide:
        1. A sentiment score between -1.0 (extremely bearish) and 1.0 (extremely bullish)
        2. A confidence score between 0.0 and 1.0
        3. A brief summary of the overall sentiment
        4. Key topics or keywords mentioned, with their sentiment (positive, negative, or neutral)
        
        Format your response as JSON:
        {{
            "sentiment_score": float,
            "confidence": float,
            "summary": "string",
            "keywords": {{
                "keyword1": "positive|negative|neutral",
                "keyword2": "positive|negative|neutral"
            }}
        }}
        """
        
        # Try OpenAI first if available
        if self.openai_client:
            try:
                response = await self._analyze_with_openai(prompt)
                if response:
                    return {
                        "method": "openai",
                        **response
                    }
            except Exception as e:
                self.logger.error(f"Error analyzing with OpenAI: {e}")
        
        # Fall back to local LLM if available
        if self.local_llm:
            try:
                response = self._analyze_with_local_llm(prompt)
                if response:
                    return {
                        "method": "local_llm",
                        **response
                    }
            except Exception as e:
                self.logger.error(f"Error analyzing with local LLM: {e}")
        
        return None
    
    async def _analyze_with_openai(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment using OpenAI.
        
        Args:
            prompt: Prompt for OpenAI
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.openai_client:
            return None
            
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content
            
            # Try to parse JSON
            try:
                # Find JSON in the response
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    return data
                else:
                    self.logger.warning("No JSON found in OpenAI response")
                    return None
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON from OpenAI response")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return None
    
    def _analyze_with_local_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment using local LLM.
        
        Args:
            prompt: Prompt for local LLM
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.local_llm:
            return None
            
        try:
            # Generate response
            response = self.local_llm(
                prompt,
                max_tokens=500,
                temperature=0.2,
                stop=["</s>", "\n\n\n"]
            )
            
            # Extract text
            content = response["choices"][0]["text"]
            
            # Try to parse JSON
            try:
                # Find JSON in the response
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    return data
                else:
                    self.logger.warning("No JSON found in local LLM response")
                    return None
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON from local LLM response")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calling local LLM: {e}")
            return None
    
    def _generate_summary(self, symbol: str, result: Dict[str, Any]) -> str:
        """
        Generate a summary of sentiment analysis results.
        
        Args:
            symbol: Symbol being analyzed
            result: Sentiment analysis results
            
        Returns:
            Summary string
        """
        score = result["sentiment_score"]
        confidence = result["confidence"]
        label = result["sentiment_label"]
        
        # Count sources
        news_count = len(result.get("sources", {}).get("news", {}).get("articles", []))
        social_count = len(result.get("sources", {}).get("social", {}).get("posts", []))
        
        if label == "bullish":
            strength = "strongly" if score > 0.5 else "moderately"
            summary = f"The sentiment for {symbol} is {strength} bullish with a score of {score:.2f} (confidence: {confidence:.2f})."
        elif label == "bearish":
            strength = "strongly" if score < -0.5 else "moderately"
            summary = f"The sentiment for {symbol} is {strength} bearish with a score of {score:.2f} (confidence: {confidence:.2f})."
        else:
            summary = f"The sentiment for {symbol} is neutral with a score of {score:.2f} (confidence: {confidence:.2f})."
        
        summary += f" Analysis based on {news_count} news articles and {social_count} social media posts."
        
        return summary
