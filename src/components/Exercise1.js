import React, { useEffect } from "react";
import { sequential, layers, data, train } from "@tensorflow/tfjs";
import wdbcTrain from "../datasets/wdbc-train.csv";
import wdbcTest from "../datasets/wdbc-test.csv";

const run = async () => {
  const trainingData = data.csv(wdbcTrain, {
    columnConfigs: {
      diagnosis: {
        isLabel: true,
      },
    },
  });

  const convertedTrainingData = trainingData
    .map(({ xs, ys }) => {
      return {
        xs: Object.values(xs),
        ys: Object.values(ys),
      };
    })
    .batch(10);

  const testingData = data.csv(wdbcTest, {
    columnConfigs: {
      diagnosis: {
        isLabel: true,
      },
    },
  });

  const convertedTestingData = testingData
    .map(({ xs, ys }) => {
      return {
        xs: Object.values(xs),
        ys: Object.values(ys),
      };
    })
    .batch(10);

  const numOfFeatures = (await trainingData.columnNames()).length - 1;

  const model = sequential();
  model.add(
    layers.dense({
      inputShape: [numOfFeatures],
      activation: "relu",
      units: 5,
    })
  );
  model.add(
    layers.dense({
      activation: "relu",
      units: 10,
    })
  );
  model.add(layers.dense({ activation: "sigmoid", units: 1 }));
  model.compile({
    loss: "binaryCrossentropy",
    optimizer: train.adam(0.06),
    metrics: ["accuracy"],
  });

  await model.fitDataset(convertedTrainingData, {
    epochs: 100,
    validationData: convertedTestingData,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch: " + epoch + ", Loss: " + logs.loss);
      },
    },
  });

  await model.save("downloads://my_model");
};

const Exercise1 = () => {
  useEffect(() => {
    run();
  }, []);
  return <h1>Browser-based Models with TensorFlow.js | Exercise 1</h1>;
};

export default Exercise1;
