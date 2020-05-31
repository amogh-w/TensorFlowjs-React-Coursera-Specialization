import React, { useEffect } from "react";

const toxicity = require("@tensorflow-models/toxicity");

const Example4 = () => {
  useEffect(() => {
    const threshold = 0.9;

    toxicity.load(threshold).then((model) => {
      const sentences = ["you suck"];

      model.classify(sentences).then((predictions) => {
        console.log(predictions);
      });
    });
  }, []);

  return <h1>Browser-based Models with TensorFlow.js | Example 4</h1>;
};

export default Example4;
