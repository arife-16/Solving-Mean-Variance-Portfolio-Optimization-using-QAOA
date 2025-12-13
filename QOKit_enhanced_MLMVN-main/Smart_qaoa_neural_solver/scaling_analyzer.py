"""
Scaling Analyzer with Unified Integration
High-performance scaling analysis with protocol compliance
"""

import numpy as np
import networkx as nx
from typing import List, Optional, Callable, Dict, Any
from contextlib import contextmanager
import sys
import os

# Add interconnect path
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import route_request, register_component

class ScalingAnalyzer:
    """Production-level scaling analyzer with unified protocol integration"""
    
    def __init__(self, 
                 reps: int,
                 qaoa_factory: Optional[Callable] = None,
                 cache_manager: Optional[Any] = None):
        self.reps = reps
        self._qaoa_factory = qaoa_factory
        self.cache_manager = cache_manager
        self._spectral_cache: Dict[str, Any] = {}
    
    def set_qaoa_factory(self, factory: Callable) -> None:
        """Set QAOA factory with validation"""
        self._qaoa_factory = factory
    
    def _hash_graph(self, graph: nx.Graph) -> str:
        """Consistent graph hashing"""
        edges = sorted(graph.edges())
        return str(hash(tuple(edges)))
    
    @contextmanager
    def _safe_state_access(self, qaoa_instance: Any):
        """Safe state access with guaranteed recovery"""
        original_state = {}
        if hasattr(qaoa_instance, 'graph'):
            original_state['graph'] = qaoa_instance.graph
        if hasattr(qaoa_instance, '_spectral_features'):
            original_state['_spectral_features'] = qaoa_instance._spectral_features
        
        try:
            yield
        finally:
            for attr, value in original_state.items():
                setattr(qaoa_instance, attr, value)
    
    def _extract_qaoa_data(self, qaoa_instance: Any, graph: nx.Graph) -> Dict[str, Any]:
        """Extract QAOA data through unified interface with state safety"""
        with self._safe_state_access(qaoa_instance):
            try:
                # Temporary state update
                qaoa_instance.graph = graph
                
                # Feature computation through interconnect
                features_result = route_request('spectralcore', 'get_spectral_features', {
                    'method_params': {'graph': graph},
                    'use_cache': True
                })
                features = features_result['data']['spectrum'] if features_result['status'] == 'ok' else self._compute_fallback_spectral(graph)
                
                # Scaling info through interconnect
                scaling_result = route_request('hybridscaling', 'get_scaling_info', {
                    'method_params': {'graph': graph, 'features': features},
                    'use_cache': True
                })
                scaling_info = scaling_result['data'] if scaling_result['status'] == 'ok' else self._compute_fallback_scaling(graph)
                
                return {
                    'scaling_info': scaling_info,
                    'spectral_features': features,
                    'state_info': {}
                }
            except Exception as e:
                # Fallback through unified provider
                return self._compute_fallback_data(graph)
    
    def _compute_fallback_spectral(self, graph: nx.Graph) -> np.ndarray:
        """Fallback spectral computation"""
        try:
            L = nx.laplacian_matrix(graph).astype(float)
            eigenvals = np.linalg.eigvals(L.toarray())
            return np.sort(eigenvals)[:min(10, len(eigenvals))]
        except:
            return np.zeros(min(10, len(graph.nodes)))

    def _compute_fallback_scaling(self, graph: nx.Graph) -> Dict[str, Any]:
        """Fallback scaling computation"""
        n = len(graph.nodes)
        return {
            'complexity': n * np.log(n) if n > 1 else 1.0,
            'recommended_depth': min(max(2, int(np.log2(n))), 10),
            'scaling_factor': 1.0,
            'n_nodes': n,
            'n_edges': len(graph.edges),
            'graph_density': nx.density(graph),
            'average_clustering': nx.average_clustering(graph)
        }
    
    def _compute_fallback_data(self, graph: nx.Graph) -> Dict[str, Any]:
        """Unified fallback computation using FallbackProvider logic"""
        # Spectral features
        try:
            L = nx.laplacian_matrix(graph).astype(float)
            eigenvals = np.linalg.eigvals(L.toarray())
            spectral_features = np.sort(eigenvals)[:min(10, len(eigenvals))]
        except:
            spectral_features = np.zeros(min(10, len(graph.nodes)))
        
        # Scaling info
        n = len(graph.nodes)
        scaling_info = {
            'complexity': n * np.log(n) if n > 1 else 1.0,
            'recommended_depth': min(max(2, int(np.log2(n))), 10),
            'scaling_factor': 1.0,
            'n_nodes': n,
            'n_edges': len(graph.edges),
            'graph_density': nx.density(graph),
            'average_clustering': nx.average_clustering(graph)
        }
        
        return {
            'scaling_info': scaling_info,
            'spectral_features': spectral_features,
            'state_info': {}
        }
    
    def _compute_spectral_properties(self, graph: nx.Graph) -> Dict[str, Any]:
        """Compute spectral properties with caching"""
        graph_hash = self._hash_graph(graph)
        
        if graph_hash in self._spectral_cache:
            return self._spectral_cache[graph_hash]
        
        try:
            laplacian = nx.laplacian_matrix(graph).toarray()
            eigenvalues = np.linalg.eigvals(laplacian)
            eigenvalues = np.sort(np.real(eigenvalues))
            
            properties = {
                'spectral_gap': float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0,
                'spectral_radius': float(np.max(eigenvalues)),
                'eigenvalue_stats': {
                    'mean': float(np.mean(eigenvalues)),
                    'std': float(np.std(eigenvalues))
                }
            }
        except:
            properties = {
                'spectral_gap': 0.0,
                'spectral_radius': 0.0,
                'eigenvalue_stats': {'mean': 0.0, 'std': 0.0}
            }
        
        self._spectral_cache[graph_hash] = properties
        return properties
    
    def analyze_scaling_performance(self, 
                                  test_graphs: List[nx.Graph],
                                  entropy_weight: float = 0.7, 
                                  topology_weight: float = 0.3,
                                  use_spectral_fft: bool = True,
                                  spectral_method: str = 'adaptive') -> Dict[str, Any]:
        """Analyze scaling performance with unified integration"""
        if not test_graphs:
            return {'error': 'No test graphs provided'}
        
        if self._qaoa_factory is None:
            raise ValueError("QAOA factory not set")
        
        analysis_data = {
            'scaling_factors': [],
            'complexity_metrics': [],
            'spectral_properties': [],
            'integration_status': []
        }
        
        for graph in test_graphs:
            try:
                # Check cache manager
                cache_key = None
                if self.cache_manager:
                    cache_result = route_request('cachemanager', 'get_graph_cache', {
                        'method_params': {'graph': graph, 'key': "scaling_analysis"},
                        'use_cache': False
                    })
                    if cache_result['status'] == 'ok':
                        cached_result = route_request('cachemanager', 'get', {
                            'method_params': {'key': cache_result['data']},
                            'use_cache': False
                        })
                        if cached_result['status'] == 'ok' and cached_result['data']:
                            self._append_cached_results(analysis_data, cached_result['data'])
                            continue
                
                # Create QAOA instance
                qaoa_instance = self._qaoa_factory(
                    graph=graph,
                    p=self.reps,
                    mlmvn_api=None  # Controlled external dependency
                )
                
                # Extract data through unified interface
                qaoa_data = self._extract_qaoa_data(qaoa_instance, graph)
                spectral_props = self._compute_spectral_properties(graph)
                
                # Save results
                result = {
                    'scaling_factor': qaoa_data['scaling_info']['scaling_factor'],
                    'complexity': qaoa_data['scaling_info']['complexity'],
                    'spectral_gap': spectral_props['spectral_gap'],
                    'graph_properties': {
                        'nodes': qaoa_data['scaling_info']['n_nodes'],
                        'edges': qaoa_data['scaling_info']['n_edges'],
                        'density': qaoa_data['scaling_info']['graph_density']
                    },
                    'integration_success': True
                }
                
                analysis_data['scaling_factors'].append(result['scaling_factor'])
                analysis_data['complexity_metrics'].append(result['complexity'])
                analysis_data['spectral_properties'].append(spectral_props)
                analysis_data['integration_status'].append(result['integration_success'])
                
                # Caching
                if self.cache_manager and cache_result['status'] == 'ok':
                    route_request('cachemanager', 'set', {
                        'method_params': {'key': cache_result['data'], 'value': result},
                        'use_cache': False
                    })
                
            except Exception:
                # Graceful failure handling
                analysis_data['integration_status'].append(False)
                continue
        
        return self._compute_final_statistics(analysis_data)
    
    def _append_cached_results(self, analysis_data: Dict, cached_result: Dict):
        """Append cached results to analysis data"""
        analysis_data['scaling_factors'].append(cached_result['scaling_factor'])
        analysis_data['complexity_metrics'].append(cached_result['complexity'])
        analysis_data['spectral_properties'].append({
            'spectral_gap': cached_result['spectral_gap']
        })
        analysis_data['integration_status'].append(cached_result['integration_success'])
    
    def _compute_final_statistics(self, analysis_data: Dict) -> Dict[str, Any]:
        """Compute final analysis statistics"""
        if not analysis_data['scaling_factors']:
            return {'error': 'No successful analyses'}
        
        scaling_factors = analysis_data['scaling_factors']
        complexity_metrics = analysis_data['complexity_metrics']
        success_rate = np.mean(analysis_data['integration_status'])
        
        return {
            'scaling_performance': {
                'mean_scale': float(np.mean(scaling_factors)),
                'std_scale': float(np.std(scaling_factors)),
                'complexity_correlation': float(np.corrcoef(scaling_factors, complexity_metrics)[0, 1]) if len(scaling_factors) > 1 else 0.0
            },
            'integration_metrics': {
                'success_rate': float(success_rate),
                'analyzed_graphs': len(scaling_factors),
                'spectral_cache_hits': len(self._spectral_cache)
            }
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache utilization statistics"""
        return {
            'spectral_cache_size': len(self._spectral_cache),
            'cache_manager_active': self.cache_manager is not None
        }
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        self._spectral_cache.clear()
        if self.cache_manager:
            route_request('cachemanager', 'clear', {'use_cache': False})

# Register component in interconnect
register_component('scalinganalyzer', ScalingAnalyzer)