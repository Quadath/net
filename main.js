const canvas = document.querySelector('.canvas')
canvas.width = 100;
canvas.height = 100;
const ctx = canvas.getContext("2d")
const weightsCounter = document.querySelector('.weights')

ctx.fillStyle = "black";

function showWeights() {
    weightsCounter.innerHTML =
    Object.keys(weights).reduce((total = "", current) => {
        return total += `${current} : ${weights[current]} <br>`;
    })
}


let data = [
    [1, 1, 1, 0.5, 1],
    [1, 0.25, 1, 1, 0.5],
    [1, 1, 1, 1, 0.20],
    [1, 0.40, 1, 1, 0.5],
    [1, 1, 1, 0.5, 1]
];



const weights = {
    x_h1: Math.random(),
    x_h2: Math.random(),
    y_h1: Math.random(),
    y_h2: Math.random(),
    offset_h1: Math.random(),
    offset_h2: Math.random(),


    h1_o: Math.random(),
    h2_o: Math.random(),
    offset_o: Math.random()
}

const sigmoid = x => 1 / (1 + Math.exp(-x))
const derivative_sigmoid = x => {
    const fx = sigmoid(x)
    return fx * (1 - fx)
}

const NN = (x, y) => {
    const h1 = sigmoid(weights.x_h1 * x + weights.y_h1 * y + weights.offset_h1);
    const h2 = sigmoid(weights.x_h2 * x + weights.y_h2 * y + weights.offset_h2);

    const o = sigmoid(weights.h1_o * h1 + weights.h2_o * h2 + weights.offset_o)
    return o;
}

const showResult = () => {
    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
            console.log(`${x} and ${y} => ${NN(x, y)} expected ${data[x][y]}`)
        }
    }
}

// showResult()

const train = () => {
    const weights_deltas = {
        x_h1: 0,
        x_h2: 0,
        y_h1: 0,
        y_h2: 0,
        offset_h1: 0,
        offset_h2: 0,


        h1_o: 0,
        h2_o: 0,
        offset_o: 0
    }

    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {

            const h1_input = weights.x_h1 * x + weights.y_h1 * y + weights.offset_h1
            const h1 = sigmoid(h1_input)

            const h2_input = weights.x_h2 * x + weights.y_h2 * y + weights.offset_h2
            const h2 = sigmoid(h2_input)

            const o_input = weights.h1_o * h1 + weights.h2_o * h2 + weights.offset_o
            const o = sigmoid(o_input);

            const delta = data[x][y] - o;
            const o_delta = delta * derivative_sigmoid(o_input)

            weights_deltas.h1_o += h1 * o_delta
            weights_deltas.h2_o += h2 * o_delta
            weights_deltas.offset_o += o_delta

            const h1_delta = o_delta * derivative_sigmoid(h1_input)
            const h2_delta = o_delta * derivative_sigmoid(h2_input)

            weights_deltas.x_h1 += x * h1_delta
            weights_deltas.y_h1 += y * h1_delta
            weights_deltas.offset_h1 += h1_delta


            weights_deltas.x_h2 += x * h2_delta
            weights_deltas.y_h2 += y * h2_delta
            weights_deltas.offset_h2 += h2_delta
        }
    }
    console.log(weights_deltas)
    return weights_deltas;
}


const applyTrainUpdate = (deltas = train()) =>
    Object.keys(weights).forEach((key) => {
        weights[key] += deltas[key]
    });

// for(let i = 0; i < 10000; i++) {
//     applyTrainUpdate()
// }

setInterval(() => {
    applyTrainUpdate()
    showWeights()
    for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 5; x++) {
                    const o = NN(x, y)
                    console.log(o)
                    ctx.fillStyle = `rgb(${o * 255}, ${o * 255}, ${o * 255})`
                    ctx.fillRect(x * 20, y * 20, x * 20 + 20, y * 20 + 20);
                }
    }
}, 100)
        
showResult()

console.log(NN(4, 1))
