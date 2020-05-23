import React, { useEffect } from "react";
import {
  sequential,
  layers,
  tensor2d,
  data,
  train,
  argMax,
} from "@tensorflow/tfjs";
import iris from "../datasets/iris.csv";

const run = async () => {
  const trainingData = data.csv(iris, {
    columnConfigs: {
      species: {
        isLabel: true,
      },
    },
  });

  const numOfFeatures = (await trainingData.columnNames()).length - 1;

  const convertedData = trainingData
    .map(({ xs, ys }) => {
      const labels = [
        ys.species === "setosa" ? 1 : 0,
        ys.species === "virginica" ? 1 : 0,
        ys.species === "versicolor" ? 1 : 0,
      ];

      return {
        xs: Object.values(xs),
        ys: Object.values(labels),
      };
    })
    .batch(10);

  const model = sequential();
  model.add(
    layers.dense({
      inputShape: [numOfFeatures],
      activation: "sigmoid",
      units: 5,
    })
  );
  model.add(layers.dense({ activation: "softmax", units: 3 }));
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: train.adam(0.06),
  });

  await model.fitDataset(convertedData, {
    epochs: 100,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch: " + epoch + ", Loss: " + logs.loss);
      },
    },
  });

  var predictions = [];
  const classNames = ["Setosa", "Virginica", "Versicolor"];

  // Setosa
  var testVal = tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);
  var prediction = model.predict(testVal);
  var pIndex = argMax(prediction, 1).dataSync();
  predictions.push(pIndex);

  // Versicolor
  testVal = tensor2d([6.4, 3.2, 4.5, 1.5], [1, 4]);
  prediction = model.predict(testVal);
  pIndex = argMax(prediction, 1).dataSync();
  predictions.push(pIndex);

  // Virginica
  testVal = tensor2d([5.8, 2.7, 5.1, 1.9], [1, 4]);
  prediction = model.predict(testVal);
  pIndex = argMax(prediction, 1).dataSync();
  predictions.push(pIndex);

  predictions.forEach((prediction) => {
    console.log(classNames[prediction]);
  });

  return 0;
};

export const Example2 = () => {
  useEffect(() => {
    run();
  }, []);
  return <h1>Browser-based Models with TensorFlow.js | Example 2</h1>;
};

export default Example2;
