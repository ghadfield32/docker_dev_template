"""MLflow model registry utilities."""
import mlflow
from typing import Optional, Dict, Any
from .config import MODEL_NAME, MODEL_STAGE_PRODUCTION


def register_model(model_uri: str, 
                   model_name: Optional[str] = None,
                   description: Optional[str] = None) -> str:
    """
    Register a model in the MLflow model registry using the fluent client API.
    
    Args:
        model_uri: URI of the model to register
        model_name: Name for the registered model
        description: Optional description
        
    Returns:
        Model version
    """
    name = model_name or MODEL_NAME
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Create registered model if it doesn't exist
        if not client.get_registered_model(name, silent=True):
            client.create_registered_model(name)
            print(f"Created new registered model: {name}")
            
        # Create new version
        mv = client.create_model_version(
            name=name,
            source=model_uri,
            description=description
        )
        print(f"Created version {mv.version} of model {name}")
        return mv.version
        
    except Exception as e:
        print(f"Failed to register model: {e}")
        raise


def promote_model_to_stage(model_name: Optional[str] = None,
                           version: Optional[str] = None,
                           stage: str = MODEL_STAGE_PRODUCTION) -> None:
    """
    Promote a model version to a specific stage using the fluent client.
    
    Args:
        model_name: Name of the registered model
        version: Version to promote (if None, promotes latest)
        stage: Target stage
    """
    name = model_name or MODEL_NAME
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get latest version if not specified
        if version is None:
            latest = client.get_latest_versions(name, stages=["None"])
            if not latest:
                raise ValueError(f"No versions found for model {name}")
            version = latest[0].version
        
        # Transition to stage
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
        print(f"Promoted model {name} version {version} to {stage}")
        
    except Exception as e:
        print(f"Failed to promote model: {e}")
        raise


def load_model_from_registry(model_name: Optional[str] = None,
                             stage: str = MODEL_STAGE_PRODUCTION):
    """
    Load a model from the registry by name and stage.
    
    Args:
        model_name: Name of the registered model
        stage: Stage to load from
        
    Returns:
        Loaded model
    """
    name = model_name or MODEL_NAME
    model_uri = f"models:/{name}/{stage}"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model {name} from {stage} stage")
        return model
    except Exception as e:
        print(f"Failed to load model from registry: {e}")
        raise


def load_model_from_run(run_id: str, artifact_path: str = "model"):
    """
    Load a model from a specific run.
    
    Args:
        run_id: MLflow run ID
        artifact_path: Path to the model artifact
        
    Returns:
        Loaded model
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model from run {run_id}")
        return model
    except Exception as e:
        print(f"Failed to load model from run: {e}")
        raise


def get_model_info(model_name: Optional[str] = None,
                   stage: str = MODEL_STAGE_PRODUCTION) -> Dict[str, Any]:
    """
    Get information about a registered model using the fluent client.
    
    Args:
        model_name: Name of the registered model
        stage: Stage to get info for
        
    Returns:
        Model information dictionary
    """
    name = model_name or MODEL_NAME
    client = mlflow.tracking.MlflowClient()
    
    try:
        model_version = client.get_latest_versions(name, stages=[stage])[0]
        
        return {
            "name": model_version.name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "description": model_version.description,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "run_id": model_version.run_id
        }
    except Exception as e:
        print(f"Failed to get model info: {e}")
        raise 
