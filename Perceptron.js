
const math = require('./math.js');
/**
* The Perceptron class allows the HMI to classify linearly seperable input data from a device so that it can make predictions
* as to if an alarm that is triggered under certain conditions by the input data will output true.
*
* @author Brendon Schultz
* @since 01/20/2022
*/
class Perceptron {

/**
 * The constructor for a Perceptron takes the inputs, outputs, threshold, learning rate, and number of epochs as parameters
 * to initialize the Perceptron object.
 *
 * @param inputs The two dimensional array of inputs
 * @param outputs The array of ouputs corresponding to each input
 * @param threshold The set threshold
 * @param learning The learning rate
 * @param epochs The number of epochs
 */
 constructor(inputs, outputs, threshold, learning, epochs) {
     this.inputs = inputs;
     this.outputs = outputs;
     this.threshold = threshold;
     this.learning = learning;
     this.epochs = epochs;
     this.weights = new Array(inputs[0].length);
     this.weights.fill(0);
     this.predictions = new Array(outputs.length);
     this.predictions.fill(1);
     this.errors = new Array(outputs.length);
     this.errors.fill(1);
     this.sumSquaredError = new Array();
 }

 /**
 * The train() method allows the Perceptron to calculate the weights needed to accurately predict the output.
 *
 * @returns The weights calculated within this method
 */
 train() {
     for (let iteration = 0; iteration < this.epochs; iteration++) {
         for (let i = 0; i < this.inputs.length; i++) {
             let dot = math.dot(this.inputs[i], this.weights);
             this.predictions[i] = (dot >= this.threshold) ? 1.0 : 0.0;
             for (let j = 0; j < this.weights.length; j++) {
                 this.weights[j] = this.weights[j] + this.learning * (this.outputs[i] - this.predictions[i]) * this.inputs[i][j];
             }
         }
         for (let i = 0; i < this.outputs.length; i++) {
             this.errors[i] = Math.pow((this.outputs[i] - this.predictions[i]), 2);
         }
         this.sumSquaredError[iteration] = math.sum(this.errors);
     }
     console.log("Weights: " + this.weights);
     console.log("The errors are: " + this.sumSquaredError);
     return this.weights;
 }

 /**
 * The predict function will predict if an alarm is going to occur based on a comparison between the dot product of
 * the current data and stored weights. This function only returns true or false and not a probability.
 *
 * @returns True if the model determines that an alarm will occur.
 */
 predict(data, weights) {
     let dot = math.dot(data, weights);
     return dot >= this.threshold;
 }

 /**
 * The didConverge() method may be used to determine if the Perceptron did converge on a set of weights that result in a sum squared
 * error value of 0. This method will determine if at least the last half of the sumSquaredError array is filled with 0s.
 *
 * @returns True if the last half of the sumSquaredError array is filled with 0.0
 */
 didConverge() {
     let sum = 0;
     if (this.epochs %2 == 0) {
         for (var i = (this.epochs / 2); i < this.epochs; i++) {
             sum += this.sumSquaredError[i];
         }
     } else {
         for (var i = ((this.epochs + 1) / 2); i < this.epochs; i++) {
             sum += this.sumSquaredError[i];
         }
     }
     return sum == 0.0;
 }
}
