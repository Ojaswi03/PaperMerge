"""Fresh-process dependency/startup checks; no research training or cloud runs."""
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', [
    'basil_core.lazy_imports', 'basil_core.models', 'basil_core.class_registry',
    'basil_core.research_protocol', 'basil_core.research_audit', 'basil_core.strict_cart',
    'gui.app', 'gui.config_library', 'gui.network_view', 'gui.services',
    'reporting.baseline_study_plots', 'reporting.adaptive_study_plots',
    'reporting.iid_study_plots', 'reporting.noniid_study_plots',
    'reporting.sequential_basil_plots', 'reporting.basil_plots',
    'reporting.comparison_plots', 'reporting.experiment_plots',
    'scripts.run_strict_cart_diagnostic', 'scripts.report_strict_cart_diagnostic',
    'basil_core.forgetting_audit', 'scripts.run_forgetting_mechanism_audit',
    'scripts.report_forgetting_mechanism_audit',
    'scripts.download_cart_literature',
    'basil_core.cart_feature_memory',
    'scripts.run_cart_feature_diagnostic',
    'scripts.report_cart_feature_diagnostic',
    'basil_core.cart_representation',
    'scripts.run_cart_representation_diagnostic',
    'scripts.report_cart_representation_diagnostic',
    'scripts.calibrate_attack', 'scripts.calibrate_final', 'scripts.calibrate_recovery',
    'scripts.calibrate_ring', 'scripts.check_ebm_rigorous', 'scripts.run_single_config',
    'tests.test_convergence',
    'noise_comm.ebm', 'basil_core.data.n_mnist',
])
def test_import_does_not_load_numerical_plotting_or_cloud_libraries(name):
    code = f"import importlib,sys; importlib.import_module({name!r}); " \
           "assert not ({'numpy','tensorflow','matplotlib','wandb','torch','tonic'} & sys.modules.keys()), " \
           "({'numpy','tensorflow','matplotlib','wandb','torch','tonic'} & sys.modules.keys())"
    result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr


def test_lazy_module_is_thread_safe_and_propagates_missing_dependency():
    from concurrent.futures import ThreadPoolExecutor
    from basil_core.lazy_imports import lazy_module
    module = lazy_module('json')
    assert module.__dict__['_target'] is None
    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(lambda _: module.loads('{"x":1}'), range(12)))
    assert values == [{'x':1}]*12
    assert module.__dict__['_target'] is sys.modules['json']
    with pytest.raises(ModuleNotFoundError):lazy_module('papermerge_nonexistent_dependency').anything


def test_deferred_tensorflow_function_caches_same_compiled_function():
    import tensorflow as tf
    from basil_core.lazy_imports import lazy_tf_function
    @lazy_tf_function(reduce_retracing=True)
    def square(x):return x*x
    assert square._compiled is None
    assert float(square(tf.constant(3.))) == 9.
    compiled = square._compiled
    assert float(square(tf.constant(4.))) == 16.
    assert square._compiled is compiled
    assert square.get_concrete_function(tf.TensorSpec([],tf.float32)) is not None


def test_heavy_libraries_are_not_imported_unconditionally_at_module_scope():
    import ast
    from scripts.audit_lazy_imports import source_files, HEAVY
    unexpected=[]
    for path in source_files():
        for node in ast.parse(path.read_text()).body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                if any(m.split('.')[0] in HEAVY for m in modules):unexpected.append(f'{path}:{node.lineno}')
    assert not unexpected, unexpected


def test_diagnostic_help_starts_without_tensorflow():
    code = "import runpy,sys; sys.argv=['run_strict_cart_diagnostic.py','--help']; " \
           "\ntry:runpy.run_path('scripts/run_strict_cart_diagnostic.py',run_name='__main__')" \
           "\nexcept SystemExit as e:assert e.code==0" \
           "\nassert 'tensorflow' not in sys.modules and 'numpy' not in sys.modules"
    result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', ['basil_plots.py','noisy_channel_plots.py',
    'comparison_plots.py','experiment_plots.py','baseline_study_plots.py',
    'adaptive_study_plots.py','iid_study_plots.py','noniid_study_plots.py',
    'sequential_basil_plots.py'])
def test_standalone_plot_module_import_without_project_on_initial_path(name):
    path=ROOT/'reporting'/name
    code=f"import runpy,sys; runpy.run_path({str(path)!r},run_name='import_check'); " \
         "assert not ({'tensorflow','numpy','matplotlib','wandb'} & sys.modules.keys())"
    result=subprocess.run([sys.executable,'-I','-c',code],cwd='/tmp',capture_output=True,text=True,timeout=30)
    assert result.returncode==0, result.stderr
