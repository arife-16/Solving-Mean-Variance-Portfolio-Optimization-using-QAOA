# enhanced_diagnostic_tool.py
import os
import json
import ast
from pathlib import Path
import inspect
import importlib.util
from typing import Dict, List, Tuple, Set, Optional
import re
from collections import defaultdict
import networkx as nx

# Добавляем путь к interconnect.py для импорта
import sys
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/')

try:
    from interconnect import Interconnect, get_component_info, get_stats
except ImportError:
    print("Предупреждение: Не удалось импортировать interconnect.py")
    Interconnect = None

class EnhancedDiagnosticTool:
    def __init__(self, framework_dirs: List[str], new_files_dirs: List[str]):
        self.interconnect = Interconnect() if Interconnect else None
        self.framework_dirs = [Path(d) for d in framework_dirs]
        self.new_files_dirs = [Path(d) for d in new_files_dirs]
        
        # Информация о компонентах
        self.framework_components = {}
        self.new_components = {}
        
        # Анализ зависимостей
        self.dependency_graph = nx.DiGraph()
        
        # Рекомендации и отчеты
        self.integration_recommendations = []
        self.compatibility_report = {}
        self.exclusion_list = []
        
        # Ключевые слова для анализа квантовых симуляторов
        self.quantum_keywords = {
            'qaoa': ['qaoa', 'quantum approximate optimization', 'variational quantum'],
            'parameters': ['theta', 'gamma', 'beta', 'params', 'parameters'],
            'optimization': ['optimize', 'minimize', 'cost', 'objective', 'energy'],
            'neural': ['neural', 'network', 'predict', 'train', 'model', 'mlp', 'mlmvn'],
            'quantum_gates': ['gate', 'circuit', 'qubit', 'hamiltonian'],
            'algorithms': ['algorithm', 'solver', 'iteration', 'convergence']
        }

    def scan_framework_components(self):
        """Расширенное сканирование компонентов фреймворка"""
        print("🔍 Сканирование компонентов фреймворка...")
        
        for directory in self.framework_dirs:
            if not directory.exists():
                print(f"⚠️  Директория не найдена: {directory}")
                continue
                
            print(f"📁 Сканирование: {directory}")
            
            for py_file in directory.rglob("*.py"):
                if py_file.name in ["__init__.py", "interconnect.py", "diagnostic_tool.py", "enhanced_diagnostic_tool.py"]:
                    continue
                    
                try:
                    component_info = self._analyze_python_file(py_file, is_framework=True)
                    if component_info:
                        self.framework_components[py_file.stem] = component_info
                        print(f"  ✅ Проанализирован: {py_file.name}")
                except Exception as e:
                    print(f"  ❌ Ошибка при анализе {py_file.name}: {e}")

    def scan_new_components(self):
        """Расширенное сканирование новых компонентов"""
        print("\n🔍 Сканирование новых компонентов...")
        
        for directory in self.new_files_dirs:
            if not directory.exists():
                print(f"⚠️  Директория не найдена: {directory}")
                continue
                
            print(f"📁 Сканирование: {directory}")
            
            for py_file in directory.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                try:
                    component_info = self._analyze_python_file(py_file, is_framework=False)
                    if component_info:
                        self.new_components[py_file.stem] = component_info
                        print(f"  ✅ Проанализирован: {py_file.name}")
                except Exception as e:
                    print(f"  ❌ Ошибка при анализе {py_file.name}: {e}")

    def _analyze_python_file(self, py_file: Path, is_framework: bool = True) -> Optional[Dict]:
        """Глубокий анализ Python файла"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Парсинг AST для более точного анализа
            try:
                tree = ast.parse(content)
            except SyntaxError:
                print(f"  ⚠️  Синтаксическая ошибка в {py_file.name}")
                return None
            
            # Анализ импортов
            imports = self._extract_imports(tree)
            
            # Анализ классов и функций
            classes = self._extract_classes(tree)
            functions = self._extract_functions(tree)
            
            # Анализ ключевых слов
            keywords_found = self._find_keywords(content)
            
            # Анализ зависимостей
            dependencies = self._analyze_dependencies(content, imports)
            
            return {
                'file_path': str(py_file),
                'relative_path': py_file.name,
                'directory': str(py_file.parent),
                'is_framework': is_framework,
                'imports': imports,
                'classes': classes,
                'functions': functions,
                'keywords': keywords_found,
                'dependencies': dependencies,
                'content_hash': hash(content),
                'file_size': py_file.stat().st_size,
                'category': self._categorize_component(classes, functions, keywords_found)
            }
            
        except Exception as e:
            print(f"  ❌ Ошибка при анализе {py_file}: {e}")
            return None

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Извлечение всех импортов из AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        return imports

    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Извлечение информации о классах"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    'docstring': ast.get_docstring(node) or ""
                })
        return classes

    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Извлечение информации о функциях"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                functions.append({
                    'name': node.name,
                    'args': args,
                    'docstring': ast.get_docstring(node) or ""
                })
        return functions

    def _find_keywords(self, content: str) -> Dict[str, List[str]]:
        """Поиск ключевых слов в содержимом файла"""
        content_lower = content.lower()
        found_keywords = {}
        
        for category, keywords in self.quantum_keywords.items():
            found = []
            for keyword in keywords:
                if keyword in content_lower:
                    found.append(keyword)
            if found:
                found_keywords[category] = found
                
        return found_keywords

    def _analyze_dependencies(self, content: str, imports: List[str]) -> Dict:
        """Анализ зависимостей компонента"""
        dependencies = {
            'numpy': any('numpy' in imp or 'np' in imp for imp in imports),
            'networkx': any('networkx' in imp or 'nx' in imp for imp in imports),
            'quantum_libs': any(lib in ' '.join(imports) for lib in ['qiskit', 'cirq', 'pennylane']),
            'ml_libs': any(lib in ' '.join(imports) for lib in ['torch', 'tensorflow', 'sklearn']),
            'optimization': any(lib in ' '.join(imports) for lib in ['scipy.optimize', 'cvxpy']),
            'interconnect': 'interconnect' in ' '.join(imports)
        }
        return dependencies

    def _categorize_component(self, classes: List[Dict], functions: List[Dict], keywords: Dict) -> str:
        """Категоризация компонента по его содержимому"""
        if 'neural' in keywords:
            return 'neural_network'
        elif 'qaoa' in keywords:
            return 'qaoa_algorithm'
        elif 'optimization' in keywords:
            return 'optimization'
        elif 'quantum_gates' in keywords:
            return 'quantum_circuit'
        elif any('test' in cls['name'].lower() for cls in classes):
            return 'test'
        elif any('benchmark' in func['name'].lower() for func in functions):
            return 'benchmark'
        else:
            return 'utility'

    def analyze_benchmark_file(self, benchmark_file: str) -> Set[str]:
        """Анализ файла бенчмарка для выявления активных компонентов"""
        print(f"\n📊 Анализ бенчмарка: {benchmark_file}")
        
        active_components = set()
        
        try:
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Поиск упоминаний компонентов фреймворка
            for component_name, component_info in self.framework_components.items():
                # Проверяем упоминание имени файла
                if component_name in content:
                    active_components.add(component_name)
                    
                # Проверяем упоминание классов
                for cls in component_info['classes']:
                    if cls['name'] in content:
                        active_components.add(component_name)
                        
                # Проверяем упоминание функций
                for func in component_info['functions']:
                    if func['name'] in content:
                        active_components.add(component_name)
            
            print(f"  ✅ Найдено активных компонентов: {len(active_components)}")
            return active_components
            
        except Exception as e:
            print(f"  ❌ Ошибка при анализе бенчмарка: {e}")
            return set()

    def generate_integration_recommendations(self, benchmark_file: str):
        """Генерация расширенных рекомендаций по интеграции"""
        print("\n🎯 Генерация рекомендаций по интеграции...")
        
        active_components = self.analyze_benchmark_file(benchmark_file)
        
        for new_name, new_info in self.new_components.items():
            recommendations = {
                'new_file': new_info['file_path'],
                'new_component': new_name,
                'category': new_info['category'],
                'recommended_integrations': [],
                'compatibility_score': {},
                'integration_type': self._determine_integration_type(new_info),
                'required_modifications': []
            }
            
            # Анализ совместимости с каждым компонентом фреймворка
            for fw_name, fw_info in self.framework_components.items():
                compatibility = self._calculate_compatibility(new_info, fw_info)
                
                if compatibility['score'] > 0.3:  # Порог совместимости
                    recommendations['recommended_integrations'].append({
                        'framework_file': fw_info['file_path'],
                        'component_name': fw_name,
                        'compatibility_score': compatibility['score'],
                        'integration_reasons': compatibility['reasons'],
                        'is_active_in_benchmark': fw_name in active_components
                    })
                    
                recommendations['compatibility_score'][fw_name] = compatibility['score']
            
            # Сортировка рекомендаций по совместимости
            recommendations['recommended_integrations'].sort(
                key=lambda x: (x['is_active_in_benchmark'], x['compatibility_score']), 
                reverse=True
            )
            
            # Определение необходимых модификаций
            recommendations['required_modifications'] = self._suggest_modifications(new_info)
            
            self.integration_recommendations.append(recommendations)

    def _calculate_compatibility(self, new_info: Dict, fw_info: Dict) -> Dict:
        """Расчет совместимости между новым компонентом и компонентом фреймворка"""
        score = 0.0
        reasons = []
        
        # Совместимость по зависимостям
        dep_overlap = set(new_info['dependencies'].keys()) & set(fw_info['dependencies'].keys())
        if dep_overlap:
            score += 0.2 * len(dep_overlap)
            reasons.append(f"Общие зависимости: {', '.join(dep_overlap)}")
        
        # Совместимость по ключевым словам
        for category, keywords in new_info['keywords'].items():
            if category in fw_info['keywords']:
                overlap = set(keywords) & set(fw_info['keywords'][category])
                if overlap:
                    score += 0.3 * len(overlap)
                    reasons.append(f"Общие ключевые слова ({category}): {', '.join(overlap)}")
        
        # Совместимость по категориям
        if new_info['category'] == 'neural_network' and fw_info['category'] in ['qaoa_algorithm', 'optimization']:
            score += 0.4
            reasons.append("Нейросеть для оптимизации квантовых алгоритмов")
        
        # Совместимость по методам/функциям
        new_methods = set(method['name'] for cls in new_info['classes'] for method in cls['methods'])
        fw_methods = set(method['name'] for cls in fw_info['classes'] for method in cls['methods'])
        
        method_overlap = new_methods & fw_methods
        if method_overlap:
            score += 0.1 * len(method_overlap)
            reasons.append(f"Общие методы: {', '.join(list(method_overlap)[:3])}")
        
        return {'score': min(score, 1.0), 'reasons': reasons}

    def _determine_integration_type(self, component_info: Dict) -> str:
        """Определение типа интеграции"""
        if component_info['category'] == 'neural_network':
            return 'predictor'  # Интеграция как предсказатель параметров
        elif component_info['category'] == 'optimization':
            return 'optimizer'  # Интеграция как оптимизатор
        elif component_info['category'] == 'qaoa_algorithm':
            return 'algorithm'  # Интеграция как алгоритм
        else:
            return 'utility'  # Интеграция как вспомогательный компонент

    def _suggest_modifications(self, component_info: Dict) -> List[str]:
        """Предложение необходимых модификаций для интеграции"""
        modifications = []
        
        if not component_info['dependencies'].get('interconnect', False):
            modifications.append("Добавить импорт interconnect")
        
        if component_info['category'] == 'neural_network':
            modifications.append("Добавить метод predict_parameters() для интеграции с QAOA")
            modifications.append("Добавить метод on_broadcast() для обработки событий")
        
        if not any('__init__' in method['name'] for cls in component_info['classes'] for method in cls['methods']):
            modifications.append("Добавить конструктор класса")
        
        return modifications

    def generate_exclusion_list(self):
        """Генерация списка файлов, с которыми НЕ нужно интегрировать"""
        print("\n🚫 Генерация списка исключений...")
        
        exclusion_categories = ['test', 'benchmark', 'utility']
        
        for fw_name, fw_info in self.framework_components.items():
            should_exclude = False
            reasons = []
            
            # Исключение по категории
            if fw_info['category'] in exclusion_categories:
                should_exclude = True
                reasons.append(f"Категория: {fw_info['category']}")
            
            # Исключение по имени файла
            if any(keyword in fw_info['relative_path'].lower() 
                   for keyword in ['test', 'benchmark', 'example', 'demo']):
                should_exclude = True
                reasons.append("Имя файла указывает на тест/бенчмарк/демо")
            
            # Исключение файлов без классов и функций
            if not fw_info['classes'] and not fw_info['functions']:
                should_exclude = True
                reasons.append("Нет классов и функций для интеграции")
            
            if should_exclude:
                self.exclusion_list.append({
                    'file': fw_info['file_path'],
                    'component': fw_name,
                    'reasons': reasons
                })

    def print_detailed_report(self):
        """Вывод подробного отчета"""
        print("\n" + "="*80)
        print("🔍 ПОДРОБНЫЙ ОТЧЕТ ПО ИНТЕГРАЦИИ КОМПОНЕНТОВ")
        print("="*80)
        
        # Статистика сканирования
        print(f"\n📊 СТАТИСТИКА СКАНИРОВАНИЯ:")
        print(f"  • Компонентов фреймворка: {len(self.framework_components)}")
        print(f"  • Новых компонентов: {len(self.new_components)}")
        print(f"  • Рекомендаций по интеграции: {len(self.integration_recommendations)}")
        print(f"  • Файлов в списке исключений: {len(self.exclusion_list)}")
        
        # Рекомендации по интеграции
        print(f"\n🎯 РЕКОМЕНДАЦИИ ПО ИНТЕГРАЦИИ:")
        print("-" * 60)
        
        for rec in self.integration_recommendations:
            print(f"\n📄 НОВЫЙ ФАЙЛ: {rec['new_component']}")
            print(f"   Путь: {rec['new_file']}")
            print(f"   Категория: {rec['category']}")
            print(f"   Тип интеграции: {rec['integration_type']}")
            
            if rec['recommended_integrations']:
                print(f"   \n   🔗 РЕКОМЕНДУЕМЫЕ ИНТЕГРАЦИИ:")
                for i, integration in enumerate(rec['recommended_integrations'][:5], 1):
                    status = "🔥 АКТИВЕН" if integration['is_active_in_benchmark'] else "💤 НЕАКТИВЕН"
                    print(f"     {i}. {Path(integration['framework_file']).name} "
                          f"({integration['compatibility_score']:.2f}) {status}")
                    print(f"        Причины: {'; '.join(integration['integration_reasons'])}")
            else:
                print(f"   ⚠️  НЕ НАЙДЕНО ПОДХОДЯЩИХ ИНТЕГРАЦИЙ")
            
            if rec['required_modifications']:
                print(f"   \n   ⚙️  НЕОБХОДИМЫЕ МОДИФИКАЦИИ:")
                for mod in rec['required_modifications']:
                    print(f"     • {mod}")
        
        # Список исключений
        print(f"\n🚫 ФАЙЛЫ ДЛЯ ИСКЛЮЧЕНИЯ ИЗ ИНТЕГРАЦИИ:")
        print("-" * 60)
        
        for exclusion in self.exclusion_list:
            print(f"• {Path(exclusion['file']).name}")
            print(f"  Причины: {'; '.join(exclusion['reasons'])}")
        
        # Краткое резюме
        print(f"\n📋 КРАТКОЕ РЕЗЮМЕ:")
        print("-" * 60)
        successful_integrations = sum(1 for rec in self.integration_recommendations 
                                    if rec['recommended_integrations'])
        print(f"• Успешных интеграций: {successful_integrations}/{len(self.integration_recommendations)}")
        print(f"• Файлов исключено: {len(self.exclusion_list)}")
        
        # Топ-5 рекомендаций
        print(f"\n🏆 ТОП-5 ИНТЕГРАЦИЙ:")
        print("-" * 60)
        all_integrations = []
        for rec in self.integration_recommendations:
            for integration in rec['recommended_integrations']:
                all_integrations.append({
                    'new_file': rec['new_component'],
                    'framework_file': Path(integration['framework_file']).name,
                    'score': integration['compatibility_score'],
                    'active': integration['is_active_in_benchmark']
                })
        
        all_integrations.sort(key=lambda x: (x['active'], x['score']), reverse=True)
        
        for i, integration in enumerate(all_integrations[:5], 1):
            status = "🔥" if integration['active'] else "💤"
            print(f"{i}. {integration['new_file']} → {integration['framework_file']} "
                  f"({integration['score']:.2f}) {status}")

    def save_report_to_file(self, output_file: str):
        """Сохранение отчета в файл"""
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'framework_components': len(self.framework_components),
            'new_components': len(self.new_components),
            'integration_recommendations': self.integration_recommendations,
            'exclusion_list': self.exclusion_list
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Отчет сохранен в файл: {output_file}")

    def run_full_analysis(self, benchmark_file: str, output_file: str = None):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА ИНТЕГРАЦИИ")
        print("="*80)
        
        # Сканирование компонентов
        self.scan_framework_components()
        self.scan_new_components()
        
        # Генерация рекомендаций
        self.generate_integration_recommendations(benchmark_file)
        
        # Генерация списка исключений
        self.generate_exclusion_list()
        
        # Вывод отчета
        self.print_detailed_report()
        
        # Сохранение отчета
        if output_file:
            self.save_report_to_file(output_file)
        
        print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")


def main():
    """Главная функция для запуска диагностики"""
    
    # Пути к директориям
    framework_dirs = [
        "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/",
        "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/fur",
        "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/fur/c"
    ]
    
    new_files_dirs = [
        "/content/drive/MyDrive/QOKit_enhanced_MLMVN/IterFree_neural_solver/",
        "/content/drive/MyDrive/QOKit_enhanced_MLMVN/IterFree_spectral_core/",
        "/content/drive/MyDrive/QOKit_enhanced_MLMVN/MLMVN/",
        "/content/drive/MyDrive/QOKit_enhanced_MLMVN/Rqaoa_agents/",
        "/content/drive/MyDrive/QOKit_enhanced_MLMVN/Rqaoa_core/"
    ]
    
    benchmark_file = "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/benchmark_new.ipynb"
    output_file = "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/integration_report.json"
    
    # Создание и запуск диагностического инструмента
    diagnostic = EnhancedDiagnosticTool(framework_dirs, new_files_dirs)
    diagnostic.run_full_analysis(benchmark_file, output_file)


if __name__ == "__main__":
    main()