from logzero import logger
import argparse
import numpy as np
import pandas as pd
import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers
from chainer.training import extensions
from chainer.datasets import tuple_dataset


class Network(chainer.Chain):

    def __init__(self, n_out):
        super(Network, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, 100)  # n_in -> 100
            self.l2 = L.Linear(None, n_out)  # 100 -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return F.softmax(self.l2(h1))


class MLP:
    def __init__(self, config):
        self._process = config.process
        self._batch_size = config.batch_size
        self._epochs = config.epochs
        self._learning_rate = config.learning_rate
        self._model_params_path = config.model_params_path
        self._n_labels = 26
        if hasattr(config, 'n_labels') :
            self._n_labels = config.n_labels
        self._training_dataset_path = None
        if hasattr(config, 'training_dataset_path') :
            self._training_dataset_path = config.training_dataset_path
        self._validation_dataset_path = None
        if hasattr(config, 'validation_dataset_path') :
            self._validation_dataset_path = config.validation_dataset_path
        self._evaluation_dataset_path = None
        if hasattr(config, 'evaluation_dataset_path') :
            self._evaluation_dataset_path = config.evaluation_dataset_path 
        self._frequency = -1
        if hasattr(config, 'frequency') :
            self._frequency = config.frequency
        self._gpu = -1
        if hasattr(config, 'gpu') :
            self._gpu = config.gpu
        self._out = ''
        if hasattr(config, 'out') :
            self._out = config.out
        self._resume = ''
        if hasattr(config, 'resume') :
            self._resume = config.resume
        self._plot = ''
        if hasattr(config, 'plot') :
            self._plot = config.plot
        self._model = None

    def create_model(self, n_out):
        return L.Classifier(Network(n_out))

    def load_dataset(self, dataset_path, x_col_name='x', y_col_name='y'):
        logger.info("Load a dataset from {}.".format(dataset_path))
        dataset_dirpath = os.path.dirname(dataset_path)
        xlist = []
        ylist = []
        indexcsv = pd.read_csv(dataset_path)
        for cell in indexcsv[x_col_name]:
            df = pd.read_csv(os.path.join(dataset_dirpath, cell), header=None)
            xlist.append(np.float32(df.as_matrix().flatten()))
        for cell in indexcsv[y_col_name]:
            ylist.append(np.int32(cell))
        return tuple_dataset.TupleDataset(xlist, ylist)

    # Load the model
    def _load_model(self, model_params_path='model.npz'):
        serializers.load_npz(model_params_path, self._model)
        return self._model

    # Save the model
    def _save_model(self, model_params_path='model.npz'):
        return serializers.save_npz(model_params_path, self._model)

    # Training
    def train(self):
        self._model = self.create_model(self._n_labels)
        if self._gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(self._gpu).use()
            self._model.to_gpu()  # Copy the model to the GPU

        optimizer = chainer.optimizers.Adam(alpha=self._learning_rate)
        optimizer.setup(self._model)

        train_dataset = self.load_dataset(self._training_dataset_path)
        valid_dataset = self.load_dataset(self._validation_dataset_path)
        train_iter = chainer.iterators.SerialIterator(train_dataset, self._batch_size)
        valid_iter = chainer.iterators.SerialIterator(valid_dataset, self._batch_size,
                                                 repeat=False, shuffle=False)

        # Set up a trainer
        updater = training.updaters.StandardUpdater(train_iter, optimizer, device=self._gpu)
        trainer = training.Trainer(updater, (self._epochs, 'epoch'), out=self._out)
        trainer.extend(extensions.Evaluator(valid_iter, self._model, device=self._gpu))
        trainer.extend(extensions.dump_graph('main/loss'))
        frequency = self._epochs if self._frequency == -1 else max(1, self._frequency)
        trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
        trainer.extend(extensions.LogReport())
        if self._plot and extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'epoch', file_name='accuracy.png'))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())
        if self._resume:
            chainer.serializers.load_npz(self._resume, trainer)
        trainer.run()
        self._save_model(self._model_params_path)

    # Evaluation
    def evaluate(self):
        test_dataset = self.load_dataset(self._evaluation_dataset_path)
        if self._gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(self._gpu).use()
            self._model.to_gpu()  # Copy the model to the GPU
        self._model = self.create_model(self._n_labels)
        self._model = self._load_model(self._model_params_path)

        test_iter = chainer.iterators.SerialIterator(test_dataset, self._batch_size,
                                                 repeat=False, shuffle=False)
        test_evaluator = extensions.Evaluator(test_iter, self._model, device=self._gpu)
        results = test_evaluator()
        print("test accuracy : ", results['main/accuracy'])

    # Initializer for inference
    def init_for_infer(self):
        self._model = self.create_model(self._n_labels)
        if self._gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(self._gpu).use()
            self._model.to_gpu()  # Copy the model to the GPU
        self._model = self._load_model(self._model_params_path)

    # Inference
    def infer(self, x):
        x = np.float32(x.flatten())
        x = self._model.xp.asarray(x.reshape(1, x.shape[0]))
        with chainer.using_config('train', False), \
                    chainer.using_config('enable_backprop', False):
            y = self._model.predictor(x)
        result = chainer.backends.cuda.to_cpu(y.data)
        return result.argmax(1)[0]

def get_args(model_params_path='model.npz', training_dataset_path="trining.csv",
        validation_dataset_path="validation.csv", evaluation_dataset_path="evaluation.csv",
        epochs=100, learning_rate=0.001, batch_size=100, n_labels=26, gpu=-1,
        process="train", description=None):
    if description is None:
        description = "MLP"
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-r",
                        type=float, default=learning_rate)
    parser.add_argument("--epochs", "-e", type=int, default=epochs,
                        help='Epochs of training.')
    parser.add_argument("--model-params-path", "-m",
                        type=str, default=model_params_path,
                        help='Path of the model parameters file.')
    parser.add_argument("--training-dataset-path", "-dt",
                        type=str, default=training_dataset_path,
                        help='Path of the training dataset.')
    parser.add_argument("--validation-dataset-path", "-dv",
                        type=str, default=validation_dataset_path,
                        help='Path of the validation dataset.')
    parser.add_argument("--evaluation-dataset-path", "-de",
                        type=str, default=evaluation_dataset_path,
                        help='Path of the evaluation dataset.')
    parser.add_argument('--process', '-p', type=str,
                        default='train', help="(train|evaluate|infer).")
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=gpu,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='chainer_mlp_result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-re', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n-labels', '-l', type=int, default=n_labels,
                        help='Number of labels')
    parser.add_argument('--plot', action='store_true',
                        help='Enable PlotReport extension')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_args()

    model = MLP(config)
    if config.process == 'train':
        model.train()
    elif config.process == 'evaluate':
        model.evaluate()
    elif config.process == 'infer':
        logger.info("Load a dataset from {}.".format(config.evaluation_dataset_path))
        model.init_for_infer()
        test_dataset = model.load_dataset(config.evaluation_dataset_path)
        for i in range(len(test_dataset)):
            x = test_dataset[i]
            result = model.infer(x[0])
            print("inference result = {}, true label = {}".format(result, x[1]))
