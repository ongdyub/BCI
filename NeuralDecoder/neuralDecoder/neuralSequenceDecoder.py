import os
import copy
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.io
import scipy.special
import tensorflow as tf
import pickle
from jiwer import wer

# import tensorflow_probability as tfp
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from tensorflow.keras.optimizers import Adam

import neuralDecoder.lrSchedule as lrSchedule
import neuralDecoder.models as models
import neuralDecoder.transformermodels as transformermodels
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from neuralDecoder.datasets import getDataset
from scipy.ndimage.filters import gaussian_filter1d

record_train_loss = []
record_gradNorm = []
record_cer = []


@tf.function(experimental_relax_shapes=True)
def gaussSmooth(inputs, kernelSD=2, padding="SAME"):
    """
    Applies a 1D gaussian smoothing operation with tensorflow to smooth the data along the time axis.

    Args:
        inputs (tensor : B x T x N): A 3d tensor with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel

    Returns:
        smoothedData (tensor : B x T x N): A smoothed 3d tensor with batch size B, time steps T, and number of features N
    """

    # get gaussian smoothing kernel
    inp = np.zeros([100], dtype=np.float32)
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Apply depth_wise convolution
    B, T, C = inputs.shape.as_list()
    filters = tf.tile(gaussKernel[None, :, None, None], [1, 1, C, 1])  # [1, W, C, 1]
    inputs = inputs[:, None, :, :]  # [B, 1, T, C]
    smoothedInputs = tf.nn.depthwise_conv2d(
        inputs, filters, strides=[1, 1, 1, 1], padding=padding
    )
    smoothedInputs = tf.squeeze(smoothedInputs, 1)

    return smoothedInputs


class NeuralSequenceDecoder(object):
    """
    This class encapsulates all the functionality needed for training, loading and running the neural sequence decoder RNN.
    To use it, initialize this class and then call .train() or .inference(). It can also be run from the command line (see bottom
    of the script). The args dictionary passed during initialization is used to configure all aspects of its behavior.
    """

    def __init__(self, args):
        self.args = args

        if not os.path.isdir(self.args["outputDir"]):
            os.mkdir(self.args["outputDir"])

        # record these parameters
        if self.args["mode"] == "train":
            with open(os.path.join(args["outputDir"], "args.yaml"), "w") as f:
                OmegaConf.save(config=self.args, f=f)

        # random variable seeding
        if self.args["seed"] == -1:
            self.args["seed"] = datetime.now().microsecond
        np.random.seed(self.args["seed"])
        tf.random.set_seed(self.args["seed"])
        random.seed(self.args["seed"])
        
        # Hyperparameters
        d_model = 256

        self.model = transformermodels.Transformer(
                                                    num_layers=6, d_model=256, num_heads=8, dff=2048,
                                                    input_vocab_size=8500, target_vocab_size=150,
                                                    pe_input=10000, pe_target=6000
                                                    )

        # Compile the model with an appropriate optimizer and loss function
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.learning_rate = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
                                            epsilon=1e-9)
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        self._prepareForTraining()

    def _buildInputNetworks(self, isTraining):
        # Build day transformation and normalization layers (FCNs)
        self.nInputLayers = np.max(self.args["dataset"]["datasetToLayerMap"]) + 1
        self.inputLayers = []
        self.normLayers = []
        for layerIdx in range(self.nInputLayers):
            datasetIdx = np.argwhere(
                np.array(self.args["dataset"]["datasetToLayerMap"]) == layerIdx
            )
            datasetIdx = datasetIdx[0, 0]
            nInputFeatures = self.args["dataset"]["nInputFeatures"]

            normLayer = tf.keras.layers.experimental.preprocessing.Normalization(
                input_shape=[nInputFeatures]
            )

            if isTraining and self.args["normLayer"]:
                normLayer.adapt(self.tfAdaptDatasets[datasetIdx].take(-1))

            inputModel = tf.keras.Sequential()
            inputModel.add(tf.keras.Input(shape=(None, nInputFeatures)))

            for i in range(self.args["model"]["inputNetwork"]["nInputLayers"]):
                if i == 0:
                    if (
                        self.args["model"]["inputNetwork"]["inputLayerSizes"][0]
                        == nInputFeatures
                    ):
                        kernelInit = tf.keras.initializers.identity()
                    else:
                        kernelInit = "glorot_uniform"
                else:
                    if (
                        self.args["model"]["inputNetwork"]["inputLayerSizes"][i]
                        == self.args["model"]["inputNetwork"]["inputLayerSizes"][i - 1]
                    ):
                        kernelInit = tf.keras.initializers.identity()
                    else:
                        kernelInit = "glorot_uniform"

                inputModel.add(
                    tf.keras.layers.Dense(
                        self.args["model"]["inputNetwork"]["inputLayerSizes"][i],
                        activation=self.args["model"]["inputNetwork"]["activation"],
                        kernel_initializer=kernelInit,
                        kernel_regularizer=tf.keras.regularizers.L2(
                            self.args["model"]["weightReg"]
                        ),
                    )
                )
                inputModel.add(
                    tf.keras.layers.Dropout(
                        rate=self.args["model"]["inputNetwork"]["dropout"]
                    )
                )

            inputModel.trainable = self.args["model"]["inputNetwork"].get(
                "trainable", True
            )
            # inputModel.summary()

            self.inputLayers.append(inputModel)
            self.normLayers.append(normLayer)

    def _buildInputLayers(self, isTraining):
        # Build day transformation and normalization layers
        self.nInputLayers = np.max(self.args["dataset"]["datasetToLayerMap"]) + 1
        self.inputLayers = []
        self.normLayers = []
        for layerIdx in range(self.nInputLayers):
            datasetIdx = np.argwhere(
                np.array(self.args["dataset"]["datasetToLayerMap"]) == layerIdx
            )
            datasetIdx = datasetIdx[0, 0]

            nInputFeatures = self.args["dataset"]["nInputFeatures"]

            # Adapt normalization layer with all data.
            normLayer = tf.keras.layers.experimental.preprocessing.Normalization(
                input_shape=[nInputFeatures]
            )
            if isTraining and self.args["normLayer"]:
                normLayer.adapt(self.tfAdaptDatasets[datasetIdx].take(-1))

            inputLayerSize = self.args["model"].get("inputLayerSize", nInputFeatures)
            if inputLayerSize == nInputFeatures:
                kernelInit = tf.keras.initializers.identity()
            else:
                kernelInit = "glorot_uniform"
            linearLayer = tf.keras.layers.Dense(
                inputLayerSize,
                kernel_initializer=kernelInit,
                kernel_regularizer=tf.keras.regularizers.L2(
                    self.args["model"]["weightReg"]
                ),
            )
            linearLayer.build(input_shape=[nInputFeatures])

            self.inputLayers.append(linearLayer)
            self.normLayers.append(normLayer)

    def _prepareForTraining(self):
        # build the dataset pipelines
        self.tfAdaptDatasets = []
        self.tfTrainDatasets = []
        self.tfValDatasets = []
        subsetChans = self.args["dataset"].get("subsetChans", -1)
        lastDaySubsetChans = self.args["dataset"].get("lastDaySubsetChans", -1)
        TXThreshold = self.args["dataset"].get("TXThreshold", True)
        spkPower = self.args["dataset"].get("spkPower", True)
        nInputFeatures = self.args["dataset"]["nInputFeatures"]
        if subsetChans > 0:
            if TXThreshold and spkPower:
                # nInputFeatures = 2*subsetChans
                chanIndices = np.random.permutation(128)[:subsetChans]
                chanIndices = np.concatenate((chanIndices, chanIndices + 128))
            else:
                # nInputFeatures = subsetChans
                if TXThreshold:
                    chanIndices = np.random.permutation(128)[:subsetChans]
                else:
                    chanIndices = np.random.permutation(128)[:subsetChans] + 128
        else:
            chanIndices = None
            if "chanIndices" in self.args["dataset"]:
                chanIndices = np.array(
                    list(
                        range(
                            self.args["dataset"]["chanIndices"][0],
                            self.args["dataset"]["chanIndices"][1],
                        )
                    )
                )
            nInputFeatures = self.args["dataset"]["nInputFeatures"]

        for i, (thisDataset, thisDataDir) in enumerate(
            zip(self.args["dataset"]["sessions"], self.args["dataset"]["dataDir"])
        ):
            trainDir = os.path.join(thisDataDir, thisDataset, "train")
            syntheticDataDir = None
            if (
                self.args["dataset"]["syntheticMixingRate"] > 0
                and self.args["dataset"]["syntheticDataDir"] is not None
            ):
                if isinstance(self.args["dataset"]["syntheticDataDir"], ListConfig):
                    if self.args["dataset"]["syntheticDataDir"][i] is not None:
                        syntheticDataDir = os.path.join(
                            self.args["dataset"]["syntheticDataDir"][i],
                            f"{thisDataset}_syntheticSentences",
                        )
                else:
                    syntheticDataDir = os.path.join(
                        self.args["dataset"]["syntheticDataDir"],
                        f"{thisDataset}_syntheticSentences",
                    )

            datasetName = self.args["dataset"]["name"]
            labelDir = None
            labelDirs = self.args["dataset"].get("labelDir", None)
            if labelDirs is not None and labelDirs[i] is not None:
                labelDir = os.path.join(labelDirs[i], thisDataset)

            lastDaySubsetSize = self.args["dataset"].get("lastDaySubsetSize", -1)
            if (
                i == (len(self.args["dataset"]["sessions"]) - 1)
                and lastDaySubsetSize != -1
            ):
                subsetSize = lastDaySubsetSize
            else:
                subsetSize = self.args["dataset"].get("subsetSize", -1)

            newTrainDataset = getDataset(datasetName)(
                trainDir,
                nInputFeatures,
                self.args["dataset"]["nClasses"],
                self.args["dataset"]["maxSeqElements"],
                self.args["dataset"]["bufferSize"],
                syntheticDataDir,
                0
                if syntheticDataDir is None
                else self.args["dataset"]["syntheticMixingRate"],
                subsetSize,
                labelDir,
                self.args["dataset"].get("timeWarpSmoothSD", 0),
                self.args["dataset"].get("timeWarpNoiseSD", 0),
                chanIndices=chanIndices,
            )

            newTrainDataset, newDatasetForAdapt = newTrainDataset.build(
                self.args["batchSize"], isTraining=True
            )

            testOnTrain = self.args["dataset"].get("testOnTrain", False)
            if "testDir" in self.args.keys():
                testDir = self.args["testDir"]
            else:
                testDir = "test"
            valDir = os.path.join(
                thisDataDir, thisDataset, testDir if not testOnTrain else "train"
            )

            newValDataset = getDataset(datasetName)(
                valDir,
                nInputFeatures,
                self.args["dataset"]["nClasses"],
                self.args["dataset"]["maxSeqElements"],
                self.args["dataset"]["bufferSize"],
                chanIndices=chanIndices,
            )
            newValDataset = newValDataset.build(
                self.args["batchSize"], isTraining=False
            )

            self.tfAdaptDatasets.append(newDatasetForAdapt)
            self.tfTrainDatasets.append(newTrainDataset)
            self.tfValDatasets.append(newValDataset)

        # Define input layers, including feature normalization which is adapted on the training data
        if "inputNetwork" in self.args["model"]:
            self._buildInputNetworks(isTraining=True)
        else:
            self._buildInputLayers(isTraining=True)

        # Train dataset selector. Used for switch between different day's data during training.
        self.trainDatasetSelector = {}
        self.trainDatasetIterators = [iter(d) for d in self.tfTrainDatasets]
        for x in range(len(self.args["dataset"]["sessions"])):
            self.trainDatasetSelector[x] = lambda x=x: self._datasetLayerTransform(
                self.trainDatasetIterators[x].get_next(),
                self.normLayers[self.args["dataset"]["datasetToLayerMap"][x]],
                self.args["dataset"]["whiteNoiseSD"],
                self.args["dataset"]["constantOffsetSD"],
                self.args["dataset"]["randomWalkSD"],
                self.args["dataset"]["staticGainSD"],
                self.args["dataset"].get("randomCut", 0),
            )

        # clear old checkpoints
        ckptFiles = [str(x) for x in Path(self.args["outputDir"]).glob("ckpt-*")]
        for file in ckptFiles:
            os.remove(file)

        if os.path.isfile(self.args["outputDir"] + "/checkpoint"):
            os.remove(self.args["outputDir"] + "/checkpoint")

        # saving/loading
        ckptVars = {}
        ckptVars["net"] = self.model
        for x in range(len(self.normLayers)):
            ckptVars["normLayer_" + str(x)] = self.normLayers[x]
            ckptVars["inputLayer_" + str(x)] = self.inputLayers[x]

        # Resume if checkpoint exists in outputDir
        resume = os.path.exists(os.path.join(self.args["outputDir"], "checkpoint"))
        if resume:
            # Resume training, so we need to load optimizer and step from checkpoint.
            ckptVars["step"] = tf.Variable(0)
            ckptVars["bestValCer"] = tf.Variable(1.0)
            ckptVars["optimizer"] = self.optimizer
            self.checkpoint = tf.train.Checkpoint(**ckptVars)
            ckptPath = tf.train.latest_checkpoint(self.args["outputDir"])
            # If in infer mode, we may want to load a particular checkpoint idx
            if self.args["mode"] == "infer":
                if self.args["loadCheckpointIdx"] is not None:
                    ckptPath = os.path.join(
                        self.args["outputDir"], f'ckpt-{self.args["loadCheckpointIdx"]}'
                    )
            print("Loading from : " + ckptPath)
            self.checkpoint.restore(ckptPath).expect_partial()
        else:
            if self.args["loadDir"] != None and os.path.exists(
                os.path.join(self.args["loadDir"], "checkpoint")
            ):
                if self.args["loadCheckpointIdx"] is not None:
                    ckptPath = os.path.join(
                        self.args["loadDir"], f'ckpt-{self.args["loadCheckpointIdx"]}'
                    )
                else:
                    ckptPath = tf.train.latest_checkpoint(self.args["loadDir"])

                print("Loading from : " + ckptPath)
                self.checkpoint = tf.train.Checkpoint(**ckptVars)
                self.checkpoint.restore(ckptPath)

                if (
                    "copyInputLayer" in self.args["dataset"]
                    and self.args["dataset"]["copyInputLayer"] is not None
                ):
                    print(self.args["dataset"]["copyInputLayer"].items())
                    for t, f in self.args["dataset"]["copyInputLayer"].items():
                        for vf, vt in zip(
                            self.inputLayers[int(f)].variables,
                            self.inputLayers[int(t)].variables,
                        ):
                            vt.assign(vf)

                # After loading, we need to put optimizer and step back to checkpoint in order to save them.
                ckptVars["step"] = tf.Variable(0)
                ckptVars["bestValCer"] = tf.Variable(1.0)
                ckptVars["optimizer"] = self.optimizer
                self.checkpoint = tf.train.Checkpoint(**ckptVars)
            else:
                # Nothing to load.
                ckptVars["step"] = tf.Variable(0)
                ckptVars["bestValCer"] = tf.Variable(1.0)
                ckptVars["optimizer"] = self.optimizer
                self.checkpoint = tf.train.Checkpoint(**ckptVars)

        self.ckptManager = tf.train.CheckpointManager(
            self.checkpoint,
            self.args["outputDir"],
            max_to_keep=None if self.args["batchesPerSave"] > 0 else 10,
        )

        # Tensorboard summary
        if self.args["mode"] == "train":
            self.summary_writer = tf.summary.create_file_writer(self.args["outputDir"])

    # train에서 그 data 들 dictionary 원본
    def _datasetLayerTransform(
        self,
        dat,
        normLayer,
        whiteNoiseSD,
        constantOffsetSD,
        randomWalkSD,
        staticGainSD,
        randomCut,
    ):
        features = dat["inputFeatures"]
        features = normLayer(dat["inputFeatures"])

        featShape = tf.shape(features)
        batchSize = featShape[0]
        featDim = featShape[2]
        if staticGainSD > 0:
            warpMat = tf.tile(
                tf.eye(features.shape[2])[tf.newaxis, :, :], [batchSize, 1, 1]
            )
            warpMat += tf.random.normal(tf.shape(warpMat), mean=0, stddev=staticGainSD)
            features = tf.linalg.matmul(features, warpMat)

        if whiteNoiseSD > 0:
            features += tf.random.normal(featShape, mean=0, stddev=whiteNoiseSD)

        if constantOffsetSD > 0:
            features += tf.random.normal(
                [batchSize, 1, featDim], mean=0, stddev=constantOffsetSD
            )

        if randomWalkSD > 0:
            features += tf.math.cumsum(
                tf.random.normal(featShape, mean=0, stddev=randomWalkSD),
                axis=self.args["randomWalkAxis"],
            )

        if randomCut > 0:
            cut = np.random.randint(0, randomCut)
            features = features[:, cut:, :]
            dat["nTimeSteps"] = dat["nTimeSteps"] - cut

        if self.args["smoothInputs"]:
            features = gaussSmooth(features, kernelSD=self.args["smoothKernelSD"])

        if self.args["lossType"] == "ctc":
            outDict = {
                "inputFeatures": features,
                #'classLabelsOneHot': dat['classLabelsOneHot'],
                "newClassSignal": dat["newClassSignal"],
                "seqClassIDs": dat["seqClassIDs"],
                "nTimeSteps": dat["nTimeSteps"],
                "nSeqElements": dat["nSeqElements"],
                "ceMask": dat["ceMask"],
                "transcription": dat["transcription"],
            }
        elif self.args["lossType"] == "ce":
            outDict = {
                "inputFeatures": features,
                "classLabelsOneHot": dat["classLabelsOneHot"],
                "newClassSignal": dat["newClassSignal"],
                "seqClassIDs": dat["seqClassIDs"],
                "nTimeSteps": dat["nTimeSteps"],
                "nSeqElements": dat["nSeqElements"],
                "ceMask": dat["ceMask"],
                "transcription": dat["transcription"],
            }

        return outDict

    def train(self):
        perBatchData_train = np.zeros([self.args["nBatchesToTrain"] + 1, 6])
        perBatchData_val = np.zeros([self.args["nBatchesToTrain"] + 1, 6])

        # Restore snapshot
        restoredStep = int(self.checkpoint.step)
        if restoredStep > 0:
            outputSnapshot = scipy.io.loadmat(
                self.args["outputDir"] + "/outputSnapshot"
            )
            perBatchData_train = outputSnapshot["perBatchData_train"]
            perBatchData_val = outputSnapshot["perBatchData_val"]

        saveBestCheckpoint = self.args["batchesPerSave"] == 0
        bestValCer = self.checkpoint.bestValCer
        print("bestVal-WER: " + str(bestValCer))
        for batchIdx in range(restoredStep, self.args["nBatchesToTrain"] + 1):
            # --training--
            if self.args["dataset"]["datasetProbability"] is None:
                nSessions = len(self.args["dataset"]["sessions"])
                self.args["dataset"]["datasetProbability"] = [
                    1.0 / nSessions
                ] * nSessions
            datasetIdx = int(
                np.argwhere(
                    np.random.multinomial(1, self.args["dataset"]["datasetProbability"])
                )[0][0]
            )
            
            layerIdx = self.args["dataset"]["datasetToLayerMap"][datasetIdx]
            
            dtStart = datetime.now()
            try:
                self._trainStep(
                    tf.constant(datasetIdx, dtype=tf.int32),
                    tf.constant(layerIdx, dtype=tf.int32),
                )

                self.checkpoint.step.assign_add(1)
                totalSeconds = (datetime.now() - dtStart).total_seconds()
                # self._addRowToStatsTable(
                #     perBatchData_train, batchIdx, totalSeconds, trainOut, True
                # )
                print(
                    f"Train batch {batchIdx}: "
                    + f'loss: {self.train_loss.result():.2f} '
                    + f'Accuracy: {self.train_accuracy.result():.2f} '
                    + f"time {totalSeconds:.2f}"
                )
                record_train_loss.append(self.train_loss.result())       
                
            except tf.errors.InvalidArgumentError as e:
                print(e)

            # --validation--
            if batchIdx % self.args["batchesPerVal"] == 0:
                dtStart = datetime.now()
                valOutputs = self.inference()
                
                avg_wer = np.average(valOutputs["wer"])
                # print(avg_wer)

                totalSeconds = (datetime.now() - dtStart).total_seconds()
                
                rn_1 = random.randint(0, len(valOutputs["targetSentences"]))
                rn_2 = random.randint(0, 5)
                                
                print(
                    f"Val batch {batchIdx}: "
                    + f'WER: {avg_wer:.8f} '
                    + f"time {totalSeconds:.2f}"
                )
                # print(valOutputs["targetSentences"])
                # print(valOutputs["decodedSentences"])
                print("-------------------- EXAMPLE -------------------")
                print("Target : " + valOutputs["targetSentences"][rn_1][rn_2])
                print("Output : " + valOutputs["decodedSentences"][rn_1][rn_2])
                
                if saveBestCheckpoint and avg_wer < bestValCer:
                    bestValCer = avg_wer
                    self.checkpoint.bestValCer.assign(bestValCer)
                    savedCkpt = self.ckptManager.save(checkpoint_number=batchIdx)
                    print(f"Checkpoint saved {savedCkpt}")

            if (
                self.args["batchesPerSave"] > 0
                and batchIdx % self.args["batchesPerSave"] == 0
            ):
                savedCkpt = self.ckptManager.save(checkpoint_number=batchIdx)
                print(f"Checkpoint saved {savedCkpt}")
                
        # with open('../record_train_loss.pkl', 'wb') as file:
        #     pickle.dump(record_train_loss, file)
        # with open('../record_cer.pkl', 'wb') as file:
        #     pickle.dump(record_cer, file)
            
        return float(bestValCer)

    def inference(self, returnData=False):
        # run through the specified dataset a single time and return the outputs
        infOut = {}
        infOut["logits"] = []
        infOut["inferSeqs"] = []
        infOut["transcription"] = []
        infOut["targetSentences"] = []
        infOut["targetLength"] = []
        infOut["decodedSentences"] = []
        infOut["wer"] = []
        allData = []

        for datasetIdx, valProb in enumerate(
            self.args["dataset"]["datasetProbabilityVal"]
        ):
            if valProb <= 0:
                continue

            layerIdx = self.args["dataset"]["datasetToLayerMap"][datasetIdx]

            for data in self.tfValDatasets[datasetIdx]:
                
                out = self._valStep(data, layerIdx)

                infOut["logits"].append(out["logits"].numpy())
                infOut["transcription"].append(out["transcription"].numpy())
                infOut["inferSeqs"].append(out["inferSeqs"].numpy())
                infOut["targetSentences"].append(out["targetSentences"])
                infOut["targetLength"].append(out["targetLength"])
                infOut["decodedSentences"].append(out["decodedSentences"])
                infOut["wer"].append(out["wer"])

        if returnData:
            return infOut, allData
        else:
            return infOut

    def _addRowToStatsTable(
        self, currentTable, batchIdx, computationTime, minibatchOutput, isTrainBatch
    ):
        currentTable[batchIdx, :] = np.array(
            [
                batchIdx,
                computationTime,
                minibatchOutput["predictionLoss"] if isTrainBatch else 0.0,
                minibatchOutput["regularizationLoss"] if isTrainBatch else 0.0,
                tf.reduce_mean(minibatchOutput["seqErrorRate"]),
                minibatchOutput["gradNorm"] if isTrainBatch else 0.0,
            ],
            dtype=object,
        )

        prefix = "train" if isTrainBatch else "val"

        with self.summary_writer.as_default():
            if isTrainBatch:
                tf.summary.scalar(
                    f"{prefix}/predictionLoss",
                    minibatchOutput["predictionLoss"],
                    step=batchIdx,
                )
                tf.summary.scalar(
                    f"{prefix}/regLoss",
                    minibatchOutput["regularizationLoss"],
                    step=batchIdx,
                )
                tf.summary.scalar(
                    f"{prefix}/gradNorm", minibatchOutput["gradNorm"], step=batchIdx
                )
            tf.summary.scalar(
                f"{prefix}/seqErrorRate",
                tf.reduce_mean(minibatchOutput["seqErrorRate"]),
                step=batchIdx,
            )
            tf.summary.scalar(
                f"{prefix}/computationTime", computationTime, step=batchIdx
            )
            # if isTrainBatch:
            #    tf.summary.scalar(
            #        f'{prefix}/lr', self.optimizer._decayed_lr(tf.float32), step=batchIdx)

    @tf.function()
    def _trainStep(self, datasetIdx, layerIdx):
        
        data = tf.switch_case(datasetIdx, self.trainDatasetSelector)
        
        inputTransformSelector = {}
        for x in range(self.nInputLayers):
            inputTransformSelector[x] = lambda x=x: self.inputLayers[x](
                data["inputFeatures"], training=True
            )

        regLossSelector = {}
        for x in range(self.nInputLayers):
            regLossSelector[x] = lambda x=x: self.inputLayers[x].losses
            
        with tf.GradientTape() as tape:
            
            inputTransformedFeatures = tf.switch_case(layerIdx, inputTransformSelector)
            
            predictions, _ = self.model([inputTransformedFeatures, data["transcription"]],
                                    training = True)
            loss = loss_function(data["transcription"], predictions)

        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(data["transcription"], predictions))

    def _valStep(self, data, layerIdx):
        data = self._datasetLayerTransform(
            data, self.normLayers[layerIdx], 0, 0, 0, 0, 0
        )

        # channel zeroing
        if "channelMask" in self.args.keys():
            maskedFeatures = data["inputFeatures"] * tf.constant(
                np.array(self.args["channelMask"])[np.newaxis, np.newaxis, :],
                dtype=tf.float32,
            )
            print("masking")
        else:
            maskedFeatures = data["inputFeatures"]

        inputTransformedFeatures = self.inputLayers[layerIdx](
            maskedFeatures, training=False
        )
        
        predictions, _ = self.model([inputTransformedFeatures, data["transcription"]],
                                 training = True)
        
        outputLabels = tf.argmax(predictions, axis=2)
        
        target_sent = []
        target_length = []
        for seq in data["transcription"]:
            endIdx = tf.argmax(tf.cast(tf.equal(seq, 0), tf.int32)).numpy()
            target_length.append(endIdx)
            characters = [chr(value) for value in seq]
            result_string = ''.join(characters)
            removed = result_string[:endIdx]
            target_sent.append(removed)
            
        output_sent = []
        for idx, seq in enumerate(outputLabels):
            characters = [chr(value) for value in seq]
            result_string = ''.join(characters)
            removed = result_string[:target_length[idx]]
            output_sent.append(removed)
        
        s_wer = 0
        for idx in range(len(target_sent)):
            s_wer += wer(target_sent, output_sent)
        s_wer /= len(target_sent)

        output = {}
        output["logits"] = predictions
        output["inferSeqs"] = outputLabels
        
        output["transcription"] = data["transcription"]
        output["targetSentences"] = target_sent
        output["targetLength"] = target_length
        
        output["decodedSentences"] = output_sent
        
        output["wer"] = s_wer

        return output


def timeWarpDataElement(dat, timeScalingRange):
    warpDat = {}
    warpDat["seqClassIDs"] = dat["seqClassIDs"]
    warpDat["nSeqElements"] = dat["nSeqElements"]
    warpDat["transcription"] = dat["transcription"]

    # nTimeSteps, inputFeatures need to be modified
    globalTimeFactor = (
        1 + (tf.random.uniform(shape=[], dtype=tf.float32) - 0.5) * timeScalingRange
    )
    warpDat["nTimeSteps"] = tf.cast(
        tf.cast(dat["nTimeSteps"], dtype=tf.float32) * globalTimeFactor, dtype=tf.int64
    )

    b = tf.shape(dat["inputFeatures"])[0]
    t = tf.cast(tf.shape(dat["inputFeatures"])[1], dtype=tf.int32)
    warppedT = tf.cast(tf.cast(t, dtype=tf.float32) * globalTimeFactor, dtype=tf.int32)
    newIdx = tf.linspace(
        tf.zeros_like(dat["nTimeSteps"], dtype=tf.int32),
        tf.ones_like(dat["nTimeSteps"], dtype=tf.int32) * (t - 1),
        warppedT,
        axis=1,
    )
    newIdx = tf.cast(newIdx, dtype=tf.int32)
    batchIdx = tf.tile(tf.range(b)[:, None, None], [1, warppedT, 1])
    newIdx = tf.concat([batchIdx, newIdx[..., None]], axis=-1)
    warpDat["inputFeatures"] = tf.gather_nd(dat["inputFeatures"], newIdx)
    # warpDat['classLabelsOneHot'] = tf.gather(
    #    dat['classLabelsOneHot'], newIdx, axis=0)
    warpDat["newClassSignal"] = tf.gather_nd(dat["newClassSignal"], newIdx)
    warpDat["ceMask"] = tf.gather_nd(dat["ceMask"], newIdx)

    return warpDat

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def _wer(reference, hypothesis):
    # Create a 2D matrix to store the distances
    distance = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]

    # Initialize the matrix with the distances
    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == 0:
                distance[i][j] = j
            elif j == 0:
                distance[i][j] = i
            else:
                cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
                distance[i][j] = min(
                    distance[i - 1][j] + 1,      # Deletion
                    distance[i][j - 1] + 1,      # Insertion
                    distance[i - 1][j - 1] + cost  # Substitution
                )

    # Return the WER
    return distance[len(reference)][len(hypothesis)] / len(reference)
