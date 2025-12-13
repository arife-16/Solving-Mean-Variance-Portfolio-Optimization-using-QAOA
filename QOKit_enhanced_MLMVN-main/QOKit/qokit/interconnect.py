import numpy as np
import networkx as nx
from typing import Dict, Any, Callable, Optional, List, Type
import inspect
import importlib
import os
import sys
from pathlib import Path

class CacheManager:
    """Простой кэш-менеджер"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
    
    def get(self, key: str):
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        if len(self._cache) >= self.max_size:
            # Удаляем первый элемент (FIFO)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value
    
    def clear(self):
        self._cache.clear()

class Interconnect:
    """Универсальный интерконект для связи между компонентами фреймворка"""
    
    def __init__(self):
        self.cache = CacheManager(max_size=1000)
        self.components = {}  # Классы компонентов
        self.instances = {}   # Экземпляры компонентов
        self.handlers = {}    # Обработчики событий
        self.auto_discovery = True
        self._discover_components()

    def _discover_components(self):
        """Автоматическое обнаружение компонентов в текущей директории"""
        if not self.auto_discovery:
            return
        
        current_dir = Path(__file__).parent
        for py_file in current_dir.glob("*.py"):
            if py_file.name in ["interconnect.py", "__init__.py"]:
                continue
            
            try:
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Ищем классы в модуле
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == module_name:
                        component_name = name.lower()
                        self.components[component_name] = obj
                        
            except Exception as e:
                print(f"Не удалось загрузить {py_file}: {e}")

    def register_component(self, name: str, component_class: Type, 
                         override: bool = False) -> Dict:
        """Регистрация компонента вручную"""
        if name in self.components and not override:
            return {
                'status': 'error', 
                'error': f'Component {name} already exists. Use override=True to replace.'
            }
        
        self.components[name] = component_class
        self._emit('component_registered', {'name': name, 'class': component_class.__name__})
        return {'status': 'ok', 'message': f'Component {name} registered successfully'}

    def route_request(self, component: str, method: str, params: Dict = None) -> Dict:
        """Маршрутизация запросов между компонентами"""
        params = params or {}
        
        try:
            # Проверка существования компонента
            if component not in self.components:
                available = list(self.components.keys())
                return {
                    'status': 'error', 
                    'error': f'Unknown component: {component}. Available: {available}'
                }

            # Получение экземпляра компонента
            instance = self._get_instance(component, params)
            
            # Проверка существования метода
            if not self._has_method(instance, method):
                available_methods = self._get_available_methods(instance)
                return {
                    'status': 'error', 
                    'error': f'Unknown method: {method}. Available: {available_methods}'
                }

            # Проверка кэша
            cache_key = self._generate_cache_key(component, method, params)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None and params.get('use_cache', True):
                return {'status': 'ok', 'data': cached_result, 'from_cache': True}

            # Выполнение метода
            method_params = params.get('method_params', {})
            result = getattr(instance, method)(**method_params)

            # Кэширование результата
            if params.get('cache_result', True):
                self.cache.set(cache_key, result)

            # Уведомление о выполнении
            self._emit('method_executed', {
                'component': component,
                'method': method,
                'instance_id': params.get('instance_id', f"{component}_default")
            })

            return {'status': 'ok', 'data': result, 'from_cache': False}

        except Exception as e:
            error_msg = f"Error in {component}.{method}: {str(e)}"
            self._emit('method_error', {
                'component': component,
                'method': method,
                'error': error_msg
            })
            return {'status': 'error', 'error': error_msg}

    def _get_instance(self, component: str, params: Dict) -> Any:
        """Получение или создание экземпляра компонента"""
        instance_id = params.get('instance_id', f"{component}_default")
        
        if instance_id not in self.instances:
            init_params = params.get('init_params', {})
            
            # Специальная обработка для NetworkX графов
            if 'graph' in init_params:
                graph = init_params['graph']
                if not isinstance(graph, nx.Graph):
                    init_params['graph'] = nx.Graph(graph)
            
            # Создание экземпляра
            instance = self.components[component](**init_params)
            self.instances[instance_id] = instance
            
            # Уведомление о создании
            self._emit('instance_created', {
                'component': component,
                'instance_id': instance_id
            })
        
        return self.instances[instance_id]

    def _has_method(self, instance: Any, method: str) -> bool:
        """Проверка существования метода у экземпляра"""
        return hasattr(instance, method) and callable(getattr(instance, method))

    def _get_available_methods(self, instance: Any) -> List[str]:
        """Получение списка доступных методов экземпляра"""
        return [name for name, method in inspect.getmembers(instance, inspect.ismethod)
                if not name.startswith('_')]

    def _generate_cache_key(self, component: str, method: str, params: Dict) -> str:
        """Генерация ключа для кэширования"""
        cache_params = {
            'method_params': params.get('method_params', {}),
            'instance_id': params.get('instance_id', f"{component}_default")
        }
        return f"{component}_{method}_{hash(str(sorted(cache_params.items())))}"

    def _emit(self, event: str, data: Dict) -> None:
        """Испускание событий"""
        for handler in self.handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                print(f"Error in event handler for {event}: {e}")

    def on_event(self, event: str, handler: Callable) -> None:
        """Регистрация обработчика событий"""
        self.handlers.setdefault(event, []).append(handler)

    def broadcast(self, event: str, data: Dict) -> Dict:
        """Рассылка события всем компонентам"""
        responses = {}
        for component_name, component_class in self.components.items():
            try:
                instance = self._get_instance(component_name, {})
                if hasattr(instance, 'on_broadcast'):
                    response = instance.on_broadcast(event, data)
                    responses[component_name] = response
            except Exception as e:
                responses[component_name] = {'error': str(e)}
        
        return {'status': 'ok', 'responses': responses}

    def get_component_info(self, component: str) -> Dict:
        """Получение информации о компоненте"""
        if component not in self.components:
            return {'status': 'error', 'error': f'Unknown component: {component}'}
        
        component_class = self.components[component]
        
        # Получение методов
        methods = []
        if component in [inst.split('_')[0] for inst in self.instances.keys()]:
            # Если экземпляр уже создан, получаем методы из него
            instance_id = f"{component}_default"
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                methods = self._get_available_methods(instance)
        else:
            # Получаем методы из класса
            methods = [name for name, method in inspect.getmembers(component_class, inspect.isfunction)
                      if not name.startswith('_')]
        
        return {
            'status': 'ok',
            'data': {
                'name': component,
                'class': component_class.__name__,
                'module': component_class.__module__,
                'methods': methods,
                'doc': component_class.__doc__
            }
        }

    def clear_cache(self) -> Dict:
        """Очистка кэша"""
        self.cache.clear()
        return {'status': 'ok', 'message': 'Cache cleared'}

    def clear_instances(self, component: str = None) -> Dict:
        """Очистка экземпляров компонентов"""
        if component:
            # Удаляем экземпляры конкретного компонента
            to_remove = [k for k in self.instances.keys() if k.startswith(f"{component}_")]
            for key in to_remove:
                del self.instances[key]
            return {'status': 'ok', 'message': f'Instances of {component} cleared'}
        else:
            # Удаляем все экземпляры
            self.instances.clear()
            return {'status': 'ok', 'message': 'All instances cleared'}

    def get_stats(self) -> Dict:
        """Получение статистики системы"""
        return {
            'status': 'ok',
            'data': {
                'components': len(self.components),
                'component_list': list(self.components.keys()),
                'instances': len(self.instances),
                'instance_list': list(self.instances.keys()),
                'cache_size': len(self.cache._cache),
                'event_handlers': {event: len(handlers) for event, handlers in self.handlers.items()}
            }
        }

# Глобальный экземпляр интерконекта
interconnect = Interconnect()

# Публичное API для компонентов
def route_request(component: str, method: str, params: Dict = None) -> Dict:
    """Маршрутизация запроса к компоненту"""
    return interconnect.route_request(component, method, params)

def register_component(name: str, component_class: Type, override: bool = False) -> Dict:
    """Регистрация компонента"""
    return interconnect.register_component(name, component_class, override)

def on_event(event: str, handler: Callable) -> None:
    """Регистрация обработчика событий"""
    interconnect.on_event(event, handler)

def broadcast(event: str, data: Dict) -> Dict:
    """Рассылка события всем компонентам"""
    return interconnect.broadcast(event, data)

def get_component_info(component: str) -> Dict:
    """Получение информации о компоненте"""
    return interconnect.get_component_info(component)

def clear_cache() -> bool:
    """Очистка кэша"""
    return interconnect.clear_cache()['status'] == 'ok'

def get_stats() -> Dict:
    """Получение статистики"""
    return interconnect.get_stats()['data']

# Примеры использования для ваших компонентов:

# Пример 1: Работа с MVN
# result = route_request('mvn', 'forward', {
#     'init_params': {'n_inputs': 10, 'k': 4},
#     'method_params': {'x': np.random.randn(10)}
# })

# Пример 2: Работа с MLMVN
# result = route_request('mlmvn', 'train', {
#     'init_params': {'layer_sizes': [10, 5, 2], 'k': 4, 'theta': 0.1},
#     'method_params': {
#         'X': np.random.randn(100, 10),
#         'y': np.random.randn(100, 2),
#         'max_iterations': 1000
#     }
# })

# Пример 3: Связь между компонентами
# Компонент A может обращаться к компоненту B через интерконект:
# def some_method_in_component_A(self):
#     result = route_request('mvn', 'forward', {'method_params': {'x': self.data}})
#     if result['status'] == 'ok':
#         return result['data']
#     else:
#         raise Exception(result['error'])