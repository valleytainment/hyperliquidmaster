"""
Event-driven communication module for the hyperliquidmaster package.

This module provides event-driven communication between core and UI components.
"""

import asyncio
import queue
import logging
import threading
from typing import Dict, Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# Define event types
EVENT_MARKET_DATA = "market_data"
EVENT_ORDER_BOOK = "order_book"
EVENT_ACCOUNT_INFO = "account_info"
EVENT_TRADE_SIGNAL = "trade_signal"
EVENT_ORDER_UPDATE = "order_update"
EVENT_ERROR = "error"
EVENT_STATUS = "status"

class EventBus:
    """
    Event bus for communication between components.
    
    This class provides a simple event bus for publishing and subscribing to events.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = {}
        self.queue = asyncio.Queue()
        self.sync_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Callback function to be called when event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to event: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from event: {event_type}")
    
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publish an event asynchronously.
        
        Args:
            event_type: Type of event to publish
            data: Event data
        """
        event = {
            "type": event_type,
            "data": data
        }
        
        await self.queue.put(event)
    
    def publish_sync(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publish an event synchronously.
        
        Args:
            event_type: Type of event to publish
            data: Event data
        """
        event = {
            "type": event_type,
            "data": data
        }
        
        self.sync_queue.put(event)
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self.running:
            try:
                # Get event from queue
                event = await self.queue.get()
                
                # Process event
                event_type = event["type"]
                data = event["data"]
                
                # Notify subscribers
                if event_type in self.subscribers:
                    for callback in self.subscribers[event_type]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in event callback: {e}")
                
                # Mark as done
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _process_sync_events(self) -> None:
        """Process events from the synchronous queue."""
        while self.running:
            try:
                # Get event from queue with timeout
                event = self.sync_queue.get(block=True, timeout=0.1)
                
                # Process event
                event_type = event["type"]
                data = event["data"]
                
                # Notify subscribers
                if event_type in self.subscribers:
                    for callback in self.subscribers[event_type]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in event callback: {e}")
                
                # Mark as done
                self.sync_queue.task_done()
            except queue.Empty:
                # No event available, continue
                pass
            except Exception as e:
                logger.error(f"Error processing sync event: {e}")
    
    def start(self) -> None:
        """Start the event bus."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_sync_events, daemon=True)
        self.thread.start()
        logger.info("Event bus started")
    
    def stop(self) -> None:
        """Stop the event bus."""
        if not self.running:
            return
        
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        logger.info("Event bus stopped")

# Global event bus instance
event_bus = EventBus()
