import os
import ast
import inspect
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import re
import json
from collections import defaultdict, Counter
import numpy as np

class InterconnectDiagnostic:
    """
    Диагностический анализатор для автоматической интеграции файлов через Interconnect
    """
    
    def __init__(self):
        # Пути к директориям
        self.framework_paths = [
            "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/",
            "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/fur/",
            "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/fur/c/"
        ]
        
        self.new_files_paths = [
            "/content/drive/MyDrive/QOKit_enhanced_MLMVN/IterFree_neural_solver/",
            "/content/drive/MyDrive/QOKit_enhanced_MLMVN/IterFree_spectral_core/",
            "/content/drive/MyDrive/QOKit_enhanced_MLMVN/MLMVN/",
            "/content/drive/MyDrive/QOKit_enhanced_MLMVN/Rqaoa_agents/",
            "/content/drive/MyDrive/QOKit_enhanced_MLMVN/Rqaoa_core/"
        ]
        
        self.interconnect_path = "/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/qokit/"
        
        # Кэши для анализа
        self.framework_files = {}
        self.new_files = {}
        self.integration_map = {}
        self.exclusion_list = set()
        
        # Расширенная система ключевых слов
        self.compatibility_patterns = {
            'quantum_simulation': [
                'quantum', 'qubit', 'gate', 'circuit', 'hamiltonian', 'pauli',
                'unitary', 'density', 'state', 'measurement', 'operator',
                'eigenvalue', 'eigenvector', 'fidelity', 'entanglement'
            ],
            'neural_network': [
                'neural', 'network', 'train', 'predict', 'layer', 'neuron',
                'activation', 'loss', 'optimizer', 'gradient', 'backprop',
                'forward', 'batch', 'epoch', 'model', 'weights', 'bias'
            ],
            'optimization': [
                'optimize', 'minimize', 'maximize', 'cost', 'objective',
                'constraint', 'solver', 'algorithm', 'iteration', 'convergence'
            ],
            'data_processing': [
                'data', 'process', 'transform', 'preprocess', 'normalize',
                'feature', 'dataset', 'batch', 'pipeline', 'filter'
            ],
            'visualization': [
                'plot', 'graph', 'visualize', 'chart', 'display', 'render',
                'matplotlib', 'plotly', 'seaborn', 'figure', 'axis'
            ],
            'benchmarking': [
                'benchmark', 'performance', 'timing', 'profiling', 'metrics',
                'evaluate', 'test', 'score', 'comparison', 'analysis'
            ]
        }

    def scan_files(self, directory: str) -> Dict[str, Dict]:
        """Сканирование файлов в директории с извлечением метаданных"""
        files_data = {}
        
        if not os.path.exists(directory):
            return files_data
            
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    filepath = os.path.join(root, file)
                    try:
                        metadata = self._extract_file_metadata(filepath)
                        if metadata:
                            files_data[file] = metadata
                    except Exception as e:
                        files_data[file] = {'error': str(e), 'path': filepath}
        
        return files_data

    def _extract_file_metadata(self, filepath: str) -> Dict:
        """Извлечение метаданных из Python файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            metadata = {
                'path': filepath,
                'file_name': os.path.basename(filepath),
                'classes': [],
                'functions': [],
                'imports': [],
                'keywords': set(),
                'docstring': '',
                'has_init': False,
                'has_main': False,
                'complexity_score': 0,
                'interaction_points': [],
                'dependencies': set(),
                'parameters': [],
                'return_types': []
            }
            
            # Основной docstring
            metadata['docstring'] = ast.get_docstring(tree) or ''
            
            # Анализ импортов
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata['imports'].append(alias.name)
                        metadata['dependencies'].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metadata['imports'].append(node.module)
                        metadata['dependencies'].add(node.module)
            
            # Анализ классов
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [],
                        'attributes': [],
                        'docstring': ast.get_docstring(node) or '',
                        'bases': [self._get_node_name(base) for base in node.bases]
                    }
                    
                    # Анализ методов класса
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = self._analyze_function(item)
                            class_info['methods'].append(method_info)
                            if item.name == '__init__':
                                metadata['has_init'] = True
                    
                    metadata['classes'].append(class_info)
            
            # Анализ функций
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Проверяем, что это не метод класса
                    is_method = False
                    for class_node in ast.walk(tree):
                        if isinstance(class_node, ast.ClassDef):
                            if node in class_node.body:
                                is_method = True
                                break
                    
                    if not is_method:
                        func_info = self._analyze_function(node)
                        metadata['functions'].append(func_info)
                        if node.name == 'main':
                            metadata['has_main'] = True
            
            # Анализ ключевых слов
            content_lower = content.lower()
            for category, keywords in self.compatibility_patterns.items():
                found_keywords = [kw for kw in keywords if kw in content_lower]
                if found_keywords:
                    metadata['keywords'].add(category)
            
            # Поиск точек взаимодействия
            metadata['interaction_points'] = self._find_interaction_points(content)
            
            # Вычисление сложности
            metadata['complexity_score'] = self._calculate_complexity(tree)
            
            return metadata
            
        except Exception as e:
            return {'error': str(e), 'path': filepath}

    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Анализ функции"""
        func_info = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'docstring': ast.get_docstring(node) or '',
            'returns': None,
            'complexity': 0
        }
        
        # Анализ возвращаемого типа
        if node.returns:
            func_info['returns'] = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Подсчет сложности функции
        func_info['complexity'] = len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.Try))])
        
        return func_info

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Вычисление сложности кода"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
            elif isinstance(node, ast.ClassDef):
                complexity += 2
        return complexity

    def _find_interaction_points(self, content: str) -> List[str]:
        """Поиск точек взаимодействия в коде"""
        interaction_points = []
        
        patterns = [
            r'\.predict\(',
            r'\.train\(',
            r'\.optimize\(',
            r'\.simulate\(',
            r'\.execute\(',
            r'\.process\(',
            r'\.transform\(',
            r'\.fit\(',
            r'\.forward\(',
            r'\.backward\(',
            r'route_request\(',
            r'interconnect\.'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                interaction_points.extend(matches)
        
        return interaction_points

    def _get_node_name(self, node) -> str:
        """Получение имени узла AST"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        return str(node)

    def _functions_similarity(self, func1: str, func2: str) -> float:
        """Вычисление схожести названий функций"""
        words1 = set(re.findall(r'[a-zA-Z]+', func1.lower()))
        words2 = set(re.findall(r'[a-zA-Z]+', func2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def analyze_compatibility(self, framework_file: Dict, new_file: Dict) -> Dict:
        """Анализ совместимости между файлами фреймворка и новыми файлами"""
        if 'error' in framework_file or 'error' in new_file:
            return {'compatible': False, 'reason': 'parsing_error', 'score': 0.0}
        
        score = 0.0
        reasons = []
        
        # Совпадение по ключевым словам
        common_keywords = framework_file['keywords'] & new_file['keywords']
        if common_keywords:
            keyword_score = len(common_keywords) / len(framework_file['keywords'] | new_file['keywords'])
            score += keyword_score * 0.4
            reasons.append(f"common_keywords: {list(common_keywords)}")
        
        # Совпадение по зависимостям
        fw_deps = framework_file.get('dependencies', set())
        new_deps = new_file.get('dependencies', set())
        
        if fw_deps and new_deps:
            deps_intersection = len(fw_deps & new_deps)
            deps_union = len(fw_deps | new_deps)
            if deps_union > 0:
                deps_score = deps_intersection / deps_union
                score += deps_score * 0.3
                reasons.append(f"common_dependencies: {deps_intersection}")
        
        # Анализ функций
        fw_functions = [f['name'] for f in framework_file.get('functions', [])]
        new_functions = [f['name'] for f in new_file.get('functions', [])]
        
        similar_functions = 0
        for nf in new_functions:
            for ff in fw_functions:
                if self._functions_similarity(nf, ff) > 0.7:
                    similar_functions += 1
                    break
        
        if new_functions:
            func_score = similar_functions / len(new_functions)
            score += func_score * 0.2
            reasons.append(f"similar_functions: {similar_functions}")
        
        # Анализ классов
        if framework_file.get('classes') and new_file.get('classes'):
            score += 0.15
            reasons.append("both_have_classes")
        
        # Точки взаимодействия
        fw_interactions = set(framework_file.get('interaction_points', []))
        new_interactions = set(new_file.get('interaction_points', []))
        
        if fw_interactions & new_interactions:
            score += 0.1
            reasons.append("interaction_points_match")
        
        # Штраф за большую разницу в сложности
        complexity_diff = abs(framework_file['complexity_score'] - new_file['complexity_score'])
        if complexity_diff > 50:
            score -= 0.1
            reasons.append("complexity_mismatch")
        
        return {
            'compatible': score > 0.25,
            'score': score,
            'reasons': reasons,
            'common_keywords': list(common_keywords) if common_keywords else [],
            'common_dependencies': list(fw_deps & new_deps) if fw_deps and new_deps else []
        }

    def generate_exclusion_list(self) -> Set[str]:
        """Генерация списка файлов фреймворка для исключения из интеграции"""
        exclusions = set()
        
        service_patterns = [
            '__init__', '__pycache__', 'test_', '_test', 'benchmark', 'example', 
            'demo', 'tutorial', 'setup', 'config', 'settings', 'constants'
        ]
        
        for framework_file, metadata in self.framework_files.items():
            if 'error' in metadata:
                exclusions.add(framework_file)
                continue
                
            # Исключаем по паттернам
            for pattern in service_patterns:
                if pattern in framework_file.lower():
                    exclusions.add(framework_file)
                    break
            
            # Исключаем файлы только с импортами
            if (not metadata.get('classes') and 
                not metadata.get('functions') and 
                len(metadata.get('imports', [])) < 3):
                exclusions.add(framework_file)
        
        return exclusions

    def generate_integration_recommendations(self) -> Dict[str, List[Tuple[str, float]]]:
        """Генерация рекомендаций по интеграции"""
        recommendations = {}
        
        for new_file, new_metadata in self.new_files.items():
            if 'error' in new_metadata:
                continue
                
            best_matches = []
            
            for framework_file, framework_metadata in self.framework_files.items():
                if framework_file in self.exclusion_list:
                    continue
                    
                compatibility = self.analyze_compatibility(framework_metadata, new_metadata)
                
                if compatibility['compatible']:
                    best_matches.append((framework_file, compatibility['score']))
            
            best_matches.sort(key=lambda x: x[1], reverse=True)
            recommendations[new_file] = best_matches[:5]
        
        return recommendations

    def run_diagnostic(self) -> Dict:
        """Основной диагностический анализ"""
        print("Запуск диагностического анализа...")
        
        # Сканирование файлов фреймворка
        print("Сканирование файлов фреймворка...")
        for path in self.framework_paths:
            files = self.scan_files(path)
            self.framework_files.update(files)
        
        # Сканирование новых файлов
        print("Сканирование новых файлов...")
        for path in self.new_files_paths:
            files = self.scan_files(path)
            self.new_files.update(files)
        
        # Генерация списка исключений
        self.exclusion_list = self.generate_exclusion_list()
        
        # Генерация рекомендаций
        print("Анализ совместимости файлов...")
        recommendations = self.generate_integration_recommendations()
        
        # Формирование отчета
        report = {
            'summary': {
                'framework_files_total': len(self.framework_files),
                'framework_files_valid': len([f for f in self.framework_files.values() if 'error' not in f]),
                'new_files_total': len(self.new_files),
                'new_files_valid': len([f for f in self.new_files.values() if 'error' not in f]),
                'excluded_files': len(self.exclusion_list),
                'integration_candidates': len(recommendations)
            },
            'integration_recommendations': recommendations,
            'excluded_files': list(self.exclusion_list),
            'framework_analysis': {
                name: {
                    'classes': len(meta.get('classes', [])),
                    'functions': len(meta.get('functions', [])),
                    'keywords': list(meta.get('keywords', set())),
                    'complexity': meta.get('complexity_score', 0),
                    'interaction_points': meta.get('interaction_points', [])
                }
                for name, meta in self.framework_files.items()
                if 'error' not in meta
            },
            'new_files_analysis': {
                name: {
                    'classes': len(meta.get('classes', [])),
                    'functions': len(meta.get('functions', [])),
                    'keywords': list(meta.get('keywords', set())),
                    'complexity': meta.get('complexity_score', 0),
                    'interaction_points': meta.get('interaction_points', [])
                }
                for name, meta in self.new_files.items()
                if 'error' not in meta
            }
        }
        
        return report

    def generate_integration_code(self, new_file: str, framework_files: List[str]) -> str:
        """Генерация кода для интеграции через Interconnect"""
        code_template = f"""
# Интеграция {new_file} через Interconnect
from interconnect import route_request, register_component, on_event

# Регистрация компонента
# register_component('{new_file.replace('.py', '')}', YourComponentClass)

# Примеры интеграции с файлами фреймворка:
"""
        for fw_file in framework_files:
            code_template += f"""
# Интеграция с {fw_file}
# result = route_request('{fw_file.replace('.py', '')}', 'method_name', {{
#     'method_params': {{'param': 'value'}}
# }})
"""
        
        return code_template

    def save_report(self, report: Dict, filename: str = "integration_report.json"):
        """Сохранение отчета в файл"""
        output_path = os.path.join(self.interconnect_path, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Отчет сохранен: {output_path}")

    def export_integration_config(self, output_path: str):
        """Экспорт конфигурации интеграции в JSON"""
        recommendations = self.generate_integration_recommendations()
        
        config = {
            'framework_analysis': {
                file_path: {
                    'file_name': analysis['file_name'],
                    'classes': [c['name'] for c in analysis.get('classes', [])],
                    'functions': [f['name'] for f in analysis.get('functions', [])],
                    'keywords': list(analysis.get('keywords', set())),
                    'complexity_score': analysis.get('complexity_score', 0)
                }
                for file_path, analysis in self.framework_files.items()
                if 'error' not in analysis
            },
            'new_files_analysis': {
                file_path: {
                    'file_name': analysis['file_name'],
                    'classes': [c['name'] for c in analysis.get('classes', [])],
                    'functions': [f['name'] for f in analysis.get('functions', [])],
                    'keywords': list(analysis.get('keywords', set())),
                    'complexity_score': analysis.get('complexity_score', 0)
                }
                for file_path, analysis in self.new_files.items()
                if 'error' not in analysis
            },
            'integration_recommendations': {
                new_file: [
                    {'framework_file': match[0], 'score': match[1]} 
                    for match in matches
                ]
                for new_file, matches in recommendations.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Конфигурация интеграции сохранена в: {output_path}")

    def print_integration_summary(self, report: Dict):
        """Вывод краткой сводки интеграции"""
        print("\n" + "="*80)
        print("РЕКОМЕНДАЦИИ ПО ИНТЕГРАЦИИ ЧЕРЕЗ INTERCONNECT")
        print("="*80)
        
        print(f"\nСтатистика:")
        print(f"• Файлов фреймворка: {report['summary']['framework_files_valid']}")
        print(f"• Новых файлов: {report['summary']['new_files_valid']}")
        print(f"• Исключенных файлов: {report['summary']['excluded_files']}")
        print(f"• Кандидатов для интеграции: {report['summary']['integration_candidates']}")
        
        print(f"\nИСКЛЮЧЕННЫЕ ФАЙЛЫ ФРЕЙМВОРКА:")
        for excluded in sorted(report['excluded_files']):
            print(f"• {excluded}")
        
        print(f"\nРЕКОМЕНДАЦИИ ПО ИНТЕГРАЦИИ:")
        for new_file, matches in report['integration_recommendations'].items():
            if matches:
                print(f"\n* {new_file} -> рекомендуется интегрировать с:")
                for match in matches[:3]:
                    print(f"  - {match[0]} (score: {match[1]:.3f})")
            else:
                print(f"\n* {new_file} -> НЕ НАЙДЕНЫ совместимые файлы фреймворка")
        
        print(f"\nПРИМЕРЫ КОДА ДЛЯ ИНТЕГРАЦИИ:")
        for new_file, matches in list(report['integration_recommendations'].items())[:3]:
            if not matches:
                continue
                
            new_file_name = new_file.replace('.py', '')
            best_match = matches[0]
            framework_file_name = best_match[0].replace('.py', '')
            
            print(f"\n# Пример для {new_file_name}:")
            print(f"from interconnect import route_request")
            print(f"result = route_request('{framework_file_name}', 'method_name', {{")
            print(f"    'method_params': {{'data': your_data}}")
            print(f"}})")

def main():
    """Основная функция диагностики"""
    diagnostic = InterconnectDiagnostic()
    
    try:
        report = diagnostic.run_diagnostic()
        diagnostic.save_report(report)
        diagnostic.export_integration_config(
            os.path.join(diagnostic.interconnect_path, 'integration_config.json')
        )
        diagnostic.print_integration_summary(report)
        
        print(f"\n✅ Диагностика завершена успешно!")
        print(f"📁 Полный отчет сохранен в: {diagnostic.interconnect_path}integration_report.json")
        print(f"⚙️  Конфигурация сохранена в: {diagnostic.interconnect_path}integration_config.json")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении диагностики: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()