var data = [
    {input: [0, 0],output: [0, 0, 0]},
    {input: [1, 0], output: [0.5, 0, 0]},
    {input: [2, 0],output: [0, 0, 0]},
    {input: [0, 1], output: [1, 1, 1]},
    {input: [1, 1], output: [0.5, 0.5, 0.5]},
    {input: [2, 1], output: [0, 0.5, 0]},
    {input: [0, 2], output: [0, 0, 0]},
    {input: [1, 2],output: [0, 0, 0.5]},
    {input: [2, 2],output: [0, 0, 0]},

];

var activation_sigmoid = x => 1 / (1 + Math.exp(-x));
var derivative_sigmoid = x => {
    const fx = activation_sigmoid(x);
    return fx * (1 - fx);
};

var weights = {
    i1_h1: Math.random(),
    i2_h1: Math.random(),
    bias_h1: Math.random(),
    i1_h2: Math.random(),
    i2_h2: Math.random(),
    bias_h2: Math.random(),
    i1_h3: Math.random(),
    i2_h3: Math.random(),
    bias_h3: Math.random(),
    h1_o1: Math.random(),
    h2_o1: Math.random(),
    h3_o1: Math.random(),
    bias_o1: Math.random(),
    h1_o2: Math.random(),
    h2_o2: Math.random(),
    h3_o2: Math.random(),
    bias_o2: Math.random(),
    h1_o3: Math.random(),
    h2_o3: Math.random(),
    h3_o3: Math.random(),
    bias_o3: Math.random(),
};

function nn(i1, i2) {
    var h1_input =
        weights.i1_h1 * i1 +
        weights.i2_h1 * i2 +
        weights.bias_h1;
    var h1 = activation_sigmoid(h1_input);

    var h2_input =
        weights.i1_h2 * i1 +
        weights.i2_h2 * i2 +
        weights.bias_h2;
    var h2 = activation_sigmoid(h2_input);

    var h3_input =
        weights.i1_h3 * i1 +
        weights.i2_h3 * i2 +
        weights.bias_h3;
    var h3 = activation_sigmoid(h3_input);

    var o1_input =
        weights.h1_o1 * h1 +
        weights.h2_o1 * h2 +
        weights.h3_o1 * h3 +
        weights.bias_o1;
    var o1 = activation_sigmoid(o1_input);

    var o2_input =
        weights.h1_o2 * h1 +
        weights.h2_o2 * h2 +
        weights.h3_o2 * h3 +
        weights.bias_o2;
    var o2 = activation_sigmoid(o2_input);

    var o3_input =
        weights.h1_o3 * h1 +
        weights.h2_o3 * h2 +
        weights.h3_o3 * h3 +
        weights.bias_o3;
    var o3 = activation_sigmoid(o3_input);

    return {o1, o2, o3};
}

var outputResults = () =>
    data.forEach(({
            input: [i1, i2],
            output: y
        }) =>
        console.log(`${i1} XOR ${i2} => ${nn(i1, i2)} (expected ${y})`)
    );

var train = () => {
    const weight_deltas = {
        i1_h1: 0,
        i2_h1: 0,
        bias_h1: 0,
        i1_h2: 0,
        i2_h2: 0,
        bias_h2: 0,
        i1_h3: 0,
        i2_h3: 0,
        bias_h3: 0,
        h1_o1: 0,
        h2_o1: 0,
        h3_o1: 0,
        bias_o1: 0,
        h1_o2: 0,
        h2_o2: 0,
        h3_o2: 0,
        bias_o2: 0,
        h1_o3: 0,
        h2_o3: 0,
        h3_o3: 0,
        bias_o3: 0,
    };

    for (var {
            input: [i1, i2],
            output
        } of data) {
        var h1_input =
            weights.i1_h1 * i1 +
            weights.i2_h1 * i2 +
            weights.bias_h1;
        var h1 = activation_sigmoid(h1_input);

        var h2_input =
            weights.i1_h2 * i1 +
            weights.i2_h2 * i2 +
            weights.bias_h2;
        var h2 = activation_sigmoid(h2_input);

        var h3_input =
            weights.i1_h3 * i1 +
            weights.i2_h3 * i2 +
            weights.bias_h3;
        var h3 = activation_sigmoid(h3_input);

        var o1_input =
            weights.h1_o1 * h1 +
            weights.h2_o1 * h2 +
            weights.h3_o1 * h3 +
            weights.bias_o1;
        var o1 = activation_sigmoid(o1_input);

        var o2_input =
            weights.h1_o2 * h1 +
            weights.h2_o2 * h2 +
            weights.h3_o2 * h3 +
            weights.bias_o2;
        var o2 = activation_sigmoid(o2_input);

        var o3_input =
            weights.h1_o3 * h1 +
            weights.h2_o3 * h2 +
            weights.h3_o3 * h3 +
            weights.bias_o3;
        var o3 = activation_sigmoid(o3_input);

        var delta1 = output[0] - o1;
        var delta2 = output[1] - o2;
        var delta3 = output[2] - o3;

        var o1_delta = delta1 * derivative_sigmoid(o1_input);
        var o2_delta = delta2 * derivative_sigmoid(o2_input);
        var o3_delta = delta3 * derivative_sigmoid(o3_input);

        weight_deltas.h1_o1 += h1 * o1_delta;
        weight_deltas.h2_o1 += h2 * o1_delta;
        weight_deltas.h3_o1 += h3 * o1_delta;
        weight_deltas.bias_o1 += o1_delta;

        weight_deltas.h1_o2 += h1 * o2_delta;
        weight_deltas.h2_o2 += h2 * o2_delta;
        weight_deltas.h3_o2 += h3 * o2_delta;
        weight_deltas.bias_o2 += o2_delta;

        weight_deltas.h1_o3 += h1 * o3_delta;
        weight_deltas.h2_o3 += h2 * o3_delta;
        weight_deltas.h3_o3 += h3 * o3_delta;
        weight_deltas.bias_o3 += o3_delta;

        var h1_delta =
            (weights.h1_o1 * o1_delta +
                weights.h1_o2 * o2_delta +
                weights.h1_o3 * o3_delta) *
            derivative_sigmoid(h1_input);
        var h2_delta =
            (weights.h2_o1 * o1_delta +
                weights.h2_o2 * o2_delta +
                weights.h2_o3 * o3_delta) *
            derivative_sigmoid(h2_input);
        var h3_delta =
            (weights.h3_o1 * o1_delta +
                weights.h3_o2 * o2_delta +
                weights.h3_o3 * o3_delta) *
            derivative_sigmoid(h3_input);

        weight_deltas.i1_h1 += i1 * h1_delta;
        weight_deltas.i2_h1 += i2 * h1_delta;
        weight_deltas.bias_h1 += h1_delta;

        weight_deltas.i1_h2 += i1 * h2_delta;
        weight_deltas.i2_h2 += i2 * h2_delta;
        weight_deltas.bias_h2 += h2_delta;

        weight_deltas.i1_h3 += i1 * h3_delta;
        weight_deltas.i2_h3 += i2 * h3_delta;
        weight_deltas.bias_h3 += h3_delta;
    }

    return weight_deltas;
}

var applyTrainUpdate = (weight_deltas = train()) =>
    Object.keys(weights).forEach((key) => (weights[key] += weight_deltas[key]));

// for (let i = 0; i < 1000; i++) {
//     applyTrainUpdate();
// }

outputResults();

console.log(nn(1, 1));

const canvas = document.querySelector('canvas')
canvas.width = 100;
canvas.height = 100;
const ctx = canvas.getContext('2d')


setInterval(() => {
  applyTrainUpdate()
  for(let x = 0; x < 3; x++) {
      for (let y = 0; y < 3; y++) {
          const {o1, o2, o3} = nn(x, y)
          ctx.fillStyle = `rgb(${o1 * 255}, ${o2 * 255}, ${o3 * 255})`
          ctx.fillRect(x * 10, y * 10, 10, 10)
      }
  }
}, 100)
