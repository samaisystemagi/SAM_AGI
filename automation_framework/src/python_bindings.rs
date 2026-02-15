use pyo3::prelude::*;
use crate::{AutomationFramework, FrameworkConfig};

#[pyclass(name = "AutomationFramework")]
pub struct PyAutomationFramework {
    inner: AutomationFramework,
}

#[pymethods]
impl PyAutomationFramework {
    #[new]
    #[pyo3(signature = (max_concurrent_subagents=10, enable_resource_tracking=true, enable_race_detection=true, billing_threshold=100.0))]
    fn new(
        max_concurrent_subagents: usize,
        enable_resource_tracking: bool,
        enable_race_detection: bool,
        billing_threshold: f64
    ) -> PyResult<Self> {
        let config = FrameworkConfig {
            max_concurrent_subagents,
            enable_resource_tracking,
            enable_race_detection,
            billing_threshold,
            ..Default::default()
        };

        // Initialize a new framework instance
        // Note: In a real async binding, we usually don't block_on here unless strictly necessary for setup.
        // AutomationFramework::new is async, so we block briefly to initialize the Arcs.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let framework = rt.block_on(async {
            AutomationFramework::new(config).await.unwrap()
        });

        Ok(PyAutomationFramework { inner: framework })
    }

    fn select_model<'p>(&self, py: Python<'p>, task: String, context: Option<String>) -> PyResult<&'p PyAny> {
        let framework = self.inner.clone();
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            match framework.select_model(&task, context.as_deref()).await {
                Ok(model) => Ok(model),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
            }
        })
    }

    fn auto_switch_model<'p>(&self, py: Python<'p>, task: String, context: Option<String>) -> PyResult<&'p PyAny> {
        let framework = self.inner.clone();
        
        pyo3_asyncio::tokio::future_into_py(py, async move {
            match framework.auto_switch_model(&task, context.as_deref()).await {
                Ok(model) => Ok(model),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())),
            }
        })
    }
}
