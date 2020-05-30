import React, { useEffect } from "react";
import {
  sequential,
  layers,
  tensor2d,
  util,
  train,
  tidy,
  browser,
  image,
  argMax,
} from "@tensorflow/tfjs";
// import { show } from "@tensorflow/tfjs-vis";

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const TRAIN_TEST_RATIO = 5 / 6;

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
  "https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png";
const MNIST_LABELS_PATH =
  "https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8";

class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  load = async () => {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = "";
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer = new ArrayBuffer(
          NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4
        );

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize
          );
          ctx.drawImage(
            img,
            0,
            i * chunkSize,
            img.width,
            chunkSize,
            0,
            0,
            img.width,
            chunkSize
          );

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages = this.datasetImages.slice(
      0,
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS
    );
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels = this.datasetLabels.slice(
      0,
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
    this.testLabels = this.datasetLabels.slice(
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
  };

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image = data[0].slice(
        idx * IMAGE_SIZE,
        idx * IMAGE_SIZE + IMAGE_SIZE
      );
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label = data[1].slice(
        idx * NUM_CLASSES,
        idx * NUM_CLASSES + NUM_CLASSES
      );
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
  }
}

var canvas, ctx, saveButton, clearButton;
var pos = { x: 0, y: 0 };
var rawImage;
var model;

const getModel = () => {
  model = sequential();

  model.add(
    layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 3,
      filters: 8,
      activation: "relu",
    })
  );
  model.add(layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(layers.conv2d({ filters: 16, kernelSize: 3, activation: "relu" }));
  model.add(layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(layers.flatten());
  model.add(layers.dense({ units: 128, activation: "relu" }));
  model.add(layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
};

const trainModel = async (model, data) => {
  // const metrics = ["loss", "val_loss", "accuracy", "val_accuracy"];
  // const container = { name: "Model Training", styles: { height: "640px" } };
  // const fitCallbacks = show.fitCallbacks(container, metrics);
  const fitCallbacks = [];

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 20,
    shuffle: true,
    callbacks: fitCallbacks,
  });
};

const setPosition = (e) => {
  pos.x = e.clientX - 100;
  pos.y = e.clientY - 150;
};

const draw = (e) => {
  if (e.buttons !== 1) return;
  ctx.beginPath();
  ctx.lineWidth = 24;
  ctx.lineCap = "round";
  ctx.strokeStyle = "white";
  ctx.moveTo(pos.x, pos.y);
  setPosition(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  rawImage.src = canvas.toDataURL("image/png");
};

const erase = () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 280, 280);
};

const save = () => {
  var raw = browser.fromPixels(rawImage, 1);
  var resized = image.resizeBilinear(raw, [28, 28]);
  var tensor = resized.expandDims(0);
  var prediction = model.predict(tensor);
  var pIndex = argMax(prediction, 1).dataSync();
  var classNames = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
  ];

  alert(classNames[pIndex]);
};

const init = () => {
  canvas = document.getElementById("canvas");
  rawImage = document.getElementById("canvasimg");
  ctx = canvas.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 280, 280);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mousedown", setPosition);
  canvas.addEventListener("mouseenter", setPosition);
  saveButton = document.getElementById("sb");
  saveButton.addEventListener("click", save);
  clearButton = document.getElementById("cb");
  clearButton.addEventListener("click", erase);
};

const run = async () => {
  const data = new MnistData();
  await data.load();
  const model = getModel();
  // show.modelSummary({ name: "Model Architecture" }, model);
  await trainModel(model, data);
  init();
  await model.save("downloads://my_model");
  alert("Training is done, try classifying your drawings!");
};

const Exercise2 = () => {
  useEffect(() => {
    run();
  }, []);
  return (
    <React.Fragment>
      <h1>Browser-based Models with TensorFlow.js | Exercise 2</h1>
      <h1>Fashion MNIST Classifier!</h1>
      <canvas
        id="canvas"
        width="280"
        height="280"
        style={{
          position: "absolute",
          top: "150px",
          left: "100px",
          border: "8px solid",
        }}
      ></canvas>
      <img
        id="canvasimg"
        style={{
          position: "absolute",
          top: "150px",
          left: "100px",
          width: "280",
          height: "280",
          display: "none",
        }}
        alt="canvasimage"
      />
      <input
        type="button"
        value="classify"
        id="sb"
        size="48"
        style={{ position: "fixed", top: "450px", left: "100px" }}
      />
      <input
        type="button"
        value="clear"
        id="cb"
        size="23"
        style={{ position: "fixed", top: "450px", left: "180px" }}
      ></input>
    </React.Fragment>
  );
};

export default Exercise2;
