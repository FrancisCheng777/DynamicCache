"""The simulator client can use the upstream wire format without model imports."""
import builtins
import importlib.util
from pathlib import Path
import threading

import numpy as np
from websockets.sync.server import serve

from evaluation.libero.evaluate_c3ache import BenchmarkClient


def test_numpy_websocket_roundtrip_without_importing_model_packages(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "wan_va/utils/Simple_Remote_Infer/deploy/msgpack_numpy.py"
    spec = importlib.util.spec_from_file_location("test_wire", path)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)

    def handler(connection):
        connection.send(wire.packb({}))
        request = wire.unpackb(connection.recv())
        connection.send(wire.packb({"array": request["array"] * 2, "seed": request["seed"]}))

    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"wan_va", "torch", "diffusers", "transformers"}
        return original_import(name, *args, **kwargs)

    with serve(handler, "127.0.0.1", 0) as websocket_server:
        thread = threading.Thread(target=websocket_server.serve_forever, daemon=True)
        thread.start()
        client = None
        try:
            monkeypatch.setattr(builtins, "__import__", checked_import)
            client = BenchmarkClient("127.0.0.1", websocket_server.socket.getsockname()[1], timeout=5)
            values = np.arange(6, dtype=np.float32).reshape(2, 3)
            reply = client.infer({"array": values, "seed": 42})
            np.testing.assert_array_equal(reply["array"], values * 2)
            assert reply["seed"] == 42
        finally:
            if client is not None:
                client.close()
            websocket_server.shutdown()
            thread.join(timeout=5)
