import React, { useEffect } from "react";
import { sequential, layers, tensor2d } from "@tensorflow/tfjs";

const doTraining = async (model, xs, ys) => {
  const history = await model.fit(xs, ys, {
    epochs: 500,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch: " + epoch + ", Loss: " + logs.loss);
      },
    },
  });
};

const Example1 = () => {
  useEffect(() => {
    console.log("Starting...");
    const model = sequential();
    model.add(layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
    model.summary();
    const xs = tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
    const ys = tensor2d([-3.0, -1.0, 2.0, 3.0, 5.0, 7.0], [6, 1]);

    doTraining(model, xs, ys).then(() => {
      alert(model.predict(tensor2d([10], [1, 1])));
    });
  }, []);
  return (
      <h1>Browser-based Models with TensorFlow.js | Example 1</h1>
  );
};

export default Example1;
