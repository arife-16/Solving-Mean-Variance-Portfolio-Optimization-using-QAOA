import numpy as np
import networkx as nx
from typing import Dict, Any, Callable, Optional, List, Type, Tuple
import inspect
import importlib.util
from pathlib import Path
import weakref
import hashlib
import threading
import time
from collections import OrderedDict
from functools import wraps, lru_cache

class AdvancedCacheManager:
    """Cache manager with TTL, LRU and adaptive memory management."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600, cleanup_interval: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._last_cleanup = time.time()
    
    def _cleanup_expired(self) -> None:
        """Cleanup expired entries."""
        current_time = time.time()
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        expired_keys = [k for k, (_, timestamp) in self._cache.items() 
                       if current_time - timestamp > self.ttl]
        for key in expired_keys:
            del self._cache[key]
        self._last_cleanup = current_time
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._cleanup_expired()
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp <= self.ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cleanup_expired()
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

class ComponentRegistry:
    """Component registry with lazy loading."""
    
    def __init__(self):
        self._components: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}
        self._weak_refs: Dict[str, weakref.ref] = {}
    
    def register(self, name: str, component_class: Type, override: bool = False) -> bool:
        if name in self._components and not override:
            return False
        self._components[name] = component_class
        return True
    
    def get_class(self, name: str) -> Optional[Type]:
        return self._components.get(name)
    
    def get_instance(self, name: str, instance_id: str, init_params: Dict = None) -> Any:
        full_id = f"{name}_{instance_id}"
        
        # Check weak reference
        if full_id in self._weak_refs:
            instance = self._weak_refs[full_id]()
            if instance is not None:
                return instance
            del self._weak_refs[full_id]
        
        # Create new instance
        if name not in self._components:
            raise ValueError(f"Unknown component: {name}")
        
        init_params = init_params or {}
        if 'graph' in init_params and not isinstance(init_params['graph'], nx.Graph):
            init_params['graph'] = nx.Graph(init_params['graph'])
        
        instance = self._components[name](**init_params)
        self._instances[full_id] = instance
        self._weak_refs[full_id] = weakref.ref(instance)
        
        return instance
    
    def clear_instances(self, component: Optional[str] = None) -> None:
        if component:
            to_remove = [k for k in self._instances.keys() if k.startswith(f"{component}_")]
            for key in to_remove:
                del self._instances[key]
                self._weak_refs.pop(key, None)
        else:
            self._instances.clear()
            self._weak_refs.clear()
    
    @property
    def component_names(self) -> List[str]:
        return list(self._components.keys())
    
    @property
    def stats(self) -> Dict[str, int]:
        return {
            'components': len(self._components),
            'instances': len(self._instances),
            'active_refs': len([ref for ref in self._weak_refs.values() if ref() is not None])
        }

class EventSystem:
    """Event system with priorities."""
    
    def __init__(self):
        self._handlers: Dict[str, List[Tuple[Callable, int]]] = {}
        self._lock = threading.RLock()
    
    def register_handler(self, event: str, handler: Callable, priority: int = 0) -> None:
        with self._lock:
            if event not in self._handlers:
                self._handlers[event] = []
            self._handlers[event].append((handler, priority))
            self._handlers[event].sort(key=lambda x: x[1], reverse=True)
    
    def emit(self, event: str, data: Dict) -> None:
        with self._lock:
            handlers = self._handlers.get(event, [])
        
        for handler, _ in handlers:
            try:
                handler(data)
            except Exception as e:
                print(f"Error in event handler {event}: {e}")
    
    def get_handler_count(self, event: str) -> int:
        return len(self._handlers.get(event, []))

class Interconnect:
    """Universal interconnect for framework component integration."""
    
    def __init__(self, auto_discovery: bool = True, cache_size: int = 1000, cache_ttl: int = 3600):
        self.cache = AdvancedCacheManager(max_size=cache_size, ttl=cache_ttl)
        self.registry = ComponentRegistry()
        self.events = EventSystem()
        self._method_cache = {}
        
        if auto_discovery:
            self._discover_components()
    
    @lru_cache(maxsize=100)
    def _get_method_signature(self, component: str, method: str) -> Optional[inspect.Signature]:
        """Cache method signatures."""
        component_class = self.registry.get_class(component)
        if component_class and hasattr(component_class, method):
            return inspect.signature(getattr(component_class, method))
        return None
    
    def _discover_components(self) -> None:
        """Automatic component discovery."""
        base_path = Path("/content/drive/MyDrive/QOKit_enhanced_MLMVN")
        directories = ["Smart_qaoa_neural_solver", "Smart_qaoa_spectral_core", "MLMVN", "Rqaoa_agents", "Rqaoa_core"]
        
        for directory in directories:
            dir_path = base_path / directory
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    module_name = f"{directory}.{py_file.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if obj.__module__ == module_name and not name.startswith('_'):
                                self.registry.register(name.lower(), obj)
                except Exception:
                    continue
    
    def _generate_cache_key(self, component: str, method: str, params: Dict) -> str:
        """Generate hash for caching."""
        cache_data = {
            'method_params': params.get('method_params', {}),
            'instance_id': params.get('instance_id', 'default')
        }
        key_string = f"{component}_{method}_{str(sorted(cache_data.items()))}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def register_component(self, name: str, component_class: Type, override: bool = False) -> Dict[str, str]:
        """Register component."""
        if self.registry.register(name, component_class, override):
            self.events.emit('component_registered', {'name': name, 'class': component_class.__name__})
            return {'status': 'ok', 'message': f'Component {name} registered'}
        return {'status': 'error', 'error': f'Component {name} already exists'}
    
    def route_request(self, component: str, method: str, params: Dict = None) -> Dict[str, Any]:
        """Route requests between components."""
        params = params or {}
        
        try:
            # Check component existence
            if not self.registry.get_class(component):
                return {
                    'status': 'error',
                    'error': f'Unknown component: {component}. Available: {self.registry.component_names}'
                }
            
            # Get instance
            instance_id = params.get('instance_id', 'default')
            init_params = params.get('init_params', {})
            instance = self.registry.get_instance(component, instance_id, init_params)
            
            # Check method
            if not hasattr(instance, method) or not callable(getattr(instance, method)):
                available_methods = [name for name, _ in inspect.getmembers(instance, inspect.ismethod) 
                                   if not name.startswith('_')]
                return {
                    'status': 'error',
                    'error': f'Unknown method: {method}. Available: {available_methods}'
                }
            
            # Caching
            cache_key = self._generate_cache_key(component, method, params)
            if params.get('use_cache', True):
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return {'status': 'ok', 'data': cached_result, 'from_cache': True}
            
            # Execute method
            method_params = params.get('method_params', {})
            result = getattr(instance, method)(**method_params)
            
            # Save to cache
            if params.get('cache_result', True):
                self.cache.set(cache_key, result)
            
            self.events.emit('method_executed', {
                'component': component,
                'method': method,
                'instance_id': instance_id
            })
            
            return {'status': 'ok', 'data': result, 'from_cache': False}
            
        except Exception as e:
            error_msg = f"Error in {component}.{method}: {str(e)}"
            self.events.emit('method_error', {
                'component': component,
                'method': method,
                'error': error_msg
            })
            return {'status': 'error', 'error': error_msg}
    
    def on_event(self, event: str, handler: Callable, priority: int = 0) -> None:
        """Register event handler."""
        self.events.register_handler(event, handler, priority)
    
    def broadcast(self, event: str, data: Dict) -> Dict[str, Any]:
        """Broadcast event to all components."""
        responses = {}
        for component_name in self.registry.component_names:
            try:
                instance = self.registry.get_instance(component_name, 'default')
                if hasattr(instance, 'on_broadcast'):
                    responses[component_name] = instance.on_broadcast(event, data)
                else:
                    responses[component_name] = {'status': 'skipped'}
            except Exception as e:
                responses[component_name] = {'status': 'error', 'error': str(e)}
        return {'status': 'ok', 'responses': responses}
    
    def get_component_info(self, component: str) -> Dict[str, Any]:
        """Get component information."""
        component_class = self.registry.get_class(component)
        if not component_class:
            return {'status': 'error', 'error': f'Unknown component: {component}'}
        
        try:
            instance = self.registry.get_instance(component, 'default')
            methods = [name for name, _ in inspect.getmembers(instance, inspect.ismethod) 
                      if not name.startswith('_')]
        except Exception:
            methods = []
        
        return {
            'status': 'ok',
            'data': {
                'name': component,
                'class': component_class.__name__,
                'module': component_class.__module__,
                'methods': methods,
                'doc': component_class.__doc__ or 'No documentation'
            }
        }
    
    def clear_cache(self) -> Dict[str, str]:
        """Clear cache."""
        self.cache.clear()
        return {'status': 'ok', 'message': 'Cache cleared'}
    
    def clear_instances(self, component: Optional[str] = None) -> Dict[str, str]:
        """Clear component instances."""
        self.registry.clear_instances(component)
        message = f'{component} instances cleared' if component else 'All instances cleared'
        return {'status': 'ok', 'message': message}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        registry_stats = self.registry.stats
        return {
            'status': 'ok',
            'data': {
                'components': registry_stats['components'],
                'component_list': self.registry.component_names,
                'instances': registry_stats['instances'],
                'active_refs': registry_stats['active_refs'],
                'cache_size': len(self.cache._cache),
                'cache_hit_rate': self.cache.hit_rate,
                'event_handlers': {
                    event: self.events.get_handler_count(event) 
                    for event in self.events._handlers.keys()
                }
            }
        }

# Global instance
interconnect = Interconnect()

# Public API
def route_request(component: str, method: str, params: Dict = None) -> Dict[str, Any]:
    return interconnect.route_request(component, method, params)

def register_component(name: str, component_class: Type, override: bool = False) -> Dict[str, str]:
    return interconnect.register_component(name, component_class, override)

def on_event(event: str, handler: Callable, priority: int = 0) -> None:
    interconnect.on_event(event, handler, priority)

def broadcast(event: str, data: Dict) -> Dict[str, Any]:
    return interconnect.broadcast(event, data)

def get_component_info(component: str) -> Dict[str, Any]:
    return interconnect.get_component_info(component)

def clear_cache() -> bool:
    return interconnect.clear_cache()['status'] == 'ok'

def get_stats() -> Dict[str, Any]:
    return interconnect.get_stats()['data']