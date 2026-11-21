"""Vérifie forward, backward et mise à jour sur une convolution connue."""
import tempfile
from pathlib import Path


def main():
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import aidge_core as ac
    import aidge_backend_cpu
    import aidge_onnx
    x = np.array([1, 2, 3, 4], np.float32).reshape(1, 1, 2, 2)
    graph = helper.make_graph([helper.make_node('Conv', ['image', 'weight', 'bias'], ['output'], kernel_shape=[1, 1])],
        'flowsense_check', [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])],
        [numpy_helper.from_array(np.full((1, 1, 1, 1), 2, np.float32), 'weight'),
         numpy_helper.from_array(np.array([.5], np.float32), 'bias')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    def tensor(value):
        result = ac.Tensor(np.ascontiguousarray(value, dtype=np.float32))
        result.to_backend('cpu')
        result.set_data_format(ac.dformat.nchw)
        return result
    with tempfile.TemporaryDirectory(prefix='flowsense-aidge-') as folder:
        path = Path(folder) / 'check.onnx'
        onnx.save(model, path)
        network = aidge_onnx.load_onnx(str(path))
        ac.expand_metaops(network)
        network.compile('cpu', ac.dtype.float32, dims=[[1, 1, 2, 2]])
        scheduler = ac.SequentialScheduler(network)
        output = scheduler.forward(data=[tensor(x)])[0]
        raw = np.array(output, copy=True)
        np.testing.assert_allclose(raw, 2*x+.5, rtol=1e-5, atol=1e-5)
        output.grad = tensor(raw / raw.size)
        scheduler.backward()
        parameters = {node.name(): node.get_operator().get_output(0) for node in ac.producers(network)}
        for name, expected in [('weight', 16.25), ('bias', 5.5)]:
            np.testing.assert_allclose(np.array(parameters[name].grad, copy=True), expected, rtol=1e-4, atol=1e-4)
        before = float(np.square(raw).mean())
        for parameter in parameters.values():
            view = np.array(parameter, copy=False)
            view -= .001*np.array(parameter.grad, copy=True)
            parameter.grad.zeros()
        after = np.array(scheduler.forward(data=[tensor(x)])[0], copy=True)
        if float(np.square(after).mean()) >= before:
            raise RuntimeError('La mise à jour ne diminue pas la loss.')
    print('Aidge CPU : sorties, gradients et mise à jour vérifiés. Tester ensuite le vrai réseau avec --check-only.')


if __name__ == '__main__':
    main()
