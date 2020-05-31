import React, { useEffect } from "react";
import dogImage from "../datasets/dog.png";
const mobilenet = require("@tensorflow-models/mobilenet");

const Example5 = () => {
  useEffect(() => {
    const img = document.getElementById("dogImage");

    mobilenet.load().then((model) => {
      model.classify(img).then((predictions) => {
        console.log(predictions);
      });
    });
  }, []);

  return (
    <React.Fragment>
      <h1>Browser-based Models with TensorFlow.js | Example 5</h1>
      <img id="dogImage" src={dogImage} alt="Dog" />;
    </React.Fragment>
  );
};

export default Example5;
