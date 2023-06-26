let data = [
    {input: [1, 10], output: 1},
    {input: [2, 10], output: 1},
    {input: [0.9, 10], output: 0},
    {input: [0.5, 5], output: 1},
    {input: [0.4, 5], output: 0},
];

const weights = {
    i1_h1: Math.random(),
    i1_h2: Math.random(),
    i2_h1: Math.random(),
    i2_h2: Math.random(),
    h1_o1: Math.random(),
    h2_o1: Math.random(),
    offset_h1: Math.random(),
    offset_h2: Math.random(),
    offset_o1: Math.random()
}

const sigmoid = x => 1 / (1 + Math.exp(-x))
const derivative_sigmoid = x => {
    const fx = sigmoid(x)
    return fx * (1 - fx)
}

const NN = (i1, i2) => {
    const h1_input = weights.i1_h1* i1 + weights.i2_h1* i2 + weights.offset_h1
    const h1 = sigmoid(h1_input)

    const h2_input = weights.i1_h2* i1 + weights.i2_h2* i2 + weights.offset_h2
    const h2 = sigmoid(h2_input)

    const o1_input = weights.h1_o1*h1 + weights.h2_o1* h2 + weights.offset_o1
    const o1 = sigmoid(o1_input)

    return o1;
}

const showResult = () => {
    data.forEach(({input: [i1, i2], output: y}) => {
        console.log(${i1} and ${i2} => ${NN(i1, i2)} expected ${y})
    })
}

// showResult()

const train = () => {
    const weights_deltas = {
        i1_h1: 0,
        i1_h2: 0,
        i2_h1: 0,
        i2_h2: 0,
        h1_o1: 0,
        h2_o1: 0,
        offset_h1: 0,
        offset_h2: 0,
        offset_o1: 0
    }

    for(const {input:[i1,i2], output} of data) {
        const h1_input = weights.i1_h1*i1 + weights.i2_h1*i2 + weights.offset_h1
        const h1 = sigmoid(h1_input)

        const h2_input = weights.i1_h2*i1 + weights.i2_h2*i2 + weights.offset_h2
        const h2 = sigmoid(h2_input)

        const o1_input = weights.h1_o1*h1 + weights.h2_o1*h2 + weights.offset_o1
        const o1 = sigmoid(o1_input)

        const delta = output - o1
        const o1_delta = delta * derivative_sigmoid(o1_input)

        weights_deltas.h1_o1 += h1 * o1_delta
        weights_deltas.h2_o1 += h2 * o1_delta
        weights_deltas.offset_o1 += o1_delta

        const h1_delta = o1_delta * derivative_sigmoid(h1_input)
        const h2_delta = o1_delta * derivative_sigmoid(h2_input)

        weights_deltas.i1_h1 += i1 * h1_delta
        weights_deltas.i2_h1 += i2 * h1_delta
        weights_deltas.offset_h1 += h1_delta

        weights_deltas.i1_h2 += i1 * h2_delta
        weights_deltas.i2_h2 += i2 * h2_delta
        weights_deltas.offset_h2 += h2_delta
    }

    return weights_deltas
}

const applyTrainUpdate = (deltas = train()) => 
    Object.keys(weights).forEach((key) => {
        weights[key] += deltas[key]
    });

for(let r = 0; r < 20000; r++) {
    applyTrainUpdate()
    showResult();
    console.log(r)
}

