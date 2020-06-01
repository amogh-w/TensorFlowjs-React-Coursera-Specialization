import React, { useEffect } from "react";
import { loadLayersModel, tensor2d } from "@tensorflow/tfjs";

const Example6 = () => {
  useEffect(() => {
    const run = async () => {
      const MODEL_URL = "http://127.0.0.1:8887/model.json";
      const model = await loadLayersModel(MODEL_URL);
      console.log(model.summary());
      const input = tensor2d([10.0], [1, 1]);
      const result = model.predict(input);
      alert(result);
    };

    run();
  }, []);

  return (
    <div>
      <h1>Browser-based Models with TensorFlow.js | Example 6</h1>
    </div>
  );
};

export default Example6;
