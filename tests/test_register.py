import types


def test_register_model_calls_mlflow(monkeypatch):
    calls = {}

    class DummyClient:
        def __init__(self):
            calls['created'] = False

        def list_registered_models(self):
            return []

        def create_registered_model(self, name):
            calls['created'] = name

        def create_model_version(self, name, source, run_id):
            calls['version'] = (name, source, run_id)
            mv = types.SimpleNamespace()
            mv.version = "1"
            return mv

        def get_model_version(self, name, version):
            return types.SimpleNamespace(status="READY")

        def transition_model_version_stage(self, name, version, stage):
            calls['stage'] = (name, version, stage)

    from importlib.machinery import SourceFileLoader
    R = SourceFileLoader("register_model", "scripts/register_model.py").load_module()

    monkeypatch.setattr(R, "MlflowClient", lambda: DummyClient())

    ver = R.register_model("fake_run", "avazu_ctr", artifact_path="model/x.joblib", stage="Staging")

    assert ver == "1"
    assert calls['version'][0] == "avazu_ctr"
    assert calls['stage'][2] == "Staging"
