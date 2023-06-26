const canvas = document.querySelector('.canvas')
canvas.width = 900;
canvas.height = 900;
const ctx = canvas.getContext("2d")

ctx.fillStyle = "black";




let data = [

];

const img = document.createElement("img");
img.src = './image.png'
img.addEventListener("load", e => {
    const cvs = document.createElement("canvas");
    const c = cvs.getContext("2d");

    c.drawImage(img, 0, 0);
    const colors = c.getImageData(0, 0, img.width, img.height).data
    console.log(colors)
    for (let y = 0; y < 32; y++) {
        for (let x = 0; x < 32; x++) {
            data[x, y] = {
                r: colors[y * 32 * 4 + x * 4]
            }
        }
    }
    ctx.drawImage(img, 0, 0)
})


const weights = {
    x_h1: Math.random(),
    x_h2: Math.random(),
    x_h3: Math.random(),
    x_h4: Math.random(),
    y_h1: Math.random(),
    y_h2: Math.random(),
    y_h3: Math.random(),
    y_h4: Math.random(),

    h1_r: Math.random(),
    h2_r: Math.random(),
    h3_r: Math.random(),
    h4_r: Math.random(),
    // h1_g: Math.random(),
    // h2_g: Math.random(),
    // h3_g: Math.random(),
    // h4_g: Math.random(),
    // h1_b: Math.random(),
    // h2_b: Math.random(),
    // h3_b: Math.random(),
    // h4_b: Math.random(),

    offset: Math.random()
}

const sigmoid = x => 1 / (1 + Math.exp(-x))
const derivative_sigmoid = x => {
    const fx = sigmoid(x)
    return fx * (1 - fx)
}

const NN = (x, y) => {
    // console.log(weights)
    const h1 = sigmoid(weights.x_h1 * x + weights.y_h1 * y);
    const h2 = sigmoid(weights.x_h2 * x + weights.y_h2 * y);
    const h3 = sigmoid(weights.x_h3 * x + weights.y_h3 * y);
    const h4 = sigmoid(weights.x_h4 * x + weights.y_h4 * y);

    const r = sigmoid(weights.h1_r * h1 + weights.h2_r * h2 + weights.h3_r * h3 + weights.h4_r * h4);
    // const g = sigmoid(weights.h1_g * h1 + weights.h2_g * h2 + weights.h3_g * h3 + weights.h4_g * h4);
    // const b = sigmoid(weights.h1_b * h1 + weights.h2_b * h2 + weights.h3_b * h3 + weights.h4_b * h4);
    return r;
}

// const showResult = () => {
//     data.forEach(({input: [i1, i2], output: y}) => {
//         console.log(`${i1} and ${i2} => ${NN(i1, i2)} expected ${y}`)
//     })
// }

// showResult()

const train = () => {
    const weights_deltas = {
        x_h1: 0,
        x_h2: 0,
        x_h3: 0,
        x_h4: 0,
        y_h1: 0,
        y_h2: 0,
        y_h3: 0,
        y_h4: 0,

        h1_r: 0,
        h2_r: 0,
        h3_r: 0,
        h4_r: 0,
        // h1_g: 0,
        // h2_g: 0,
        // h3_g: 0,
        // h4_g: 0,
        // h1_b: 0,
        // h2_b: 0,
        // h3_b:0,
        // h4_b: 0,

        offset: 0
    }

    for (let y = 0; y < 32; y++) {
        for (let x = 0; x < 32; x++) {

            const h1_input = weights.x_h1 * x + weights.y_h1 * y
            const h1 = sigmoid(h1_input)
            const h2_input = weights.x_h2 * x + weights.y_h2 * y
            const h2 = sigmoid(h2_input)
            const h3_input = weights.x_h3 * x + weights.y_h3 * y
            const h3 = sigmoid(h3_input)
            const h4_input = weights.x_h4 * x + weights.y_h4 * y
            const h4 = sigmoid(h4_input)

            const r_input = weights.h1_r * h1 + weights.h2_r * h2 + weights.h3_r * h3 + weights.h4_r * h4
            const r = sigmoid(r_input);
            // const g = sigmoid(weights.h1_g * h1 + weights.h2_g * h2 + weights.h3_g * h3 + weights.h4_g * h4)
            // const b = sigmoid(weights.h1_b * h1 + weights.h2_b * h2 + weights.h3_b * h3 + weights.h4_b * h4)
            // console.log(r)
            let r_delta = (data[x, y] / 255) - r;
            console.log(data[x, y])
            r_delta = r_delta * derivative_sigmoid(r_input)

            weights_deltas.h1_r += h1 * r_delta
            weights_deltas.h2_r += h2 * r_delta
            weights_deltas.h3_r += h3 * r_delta
            weights_deltas.h4_r += h4 * r_delta

            const h1_delta = r_delta * derivative_sigmoid(h1_input)
            const h2_delta = r_delta * derivative_sigmoid(h2_input)
            const h3_delta = r_delta * derivative_sigmoid(h3_input)
            const h4_delta = r_delta * derivative_sigmoid(h4_input)

            weights_deltas.x_h1 += x * h1_delta
            weights_deltas.y_h1 += y * h1_delta
            // weights_deltas.offset += h1_delta

            weights_deltas.x_h2 += x * h2_delta
            weights_deltas.y_h2 += y * h2_delta

            weights_deltas.x_h3 += x * h3_delta
            weights_deltas.y_h3 += y * h3_delta

            weights_deltas.x_h4 += x * h4_delta
            weights_deltas.y_h4 += y * h4_delta

        }
    }
    console.log(weights_deltas)
    return weights_deltas;
}




const applyTrainUpdate = (deltas = train()) =>
    Object.keys(weights).forEach((key) => {
        weights[key] += deltas[key]
    });

setInterval(() => {
    applyTrainUpdate()
    for (let y = 0; y < 32; y++) {
        for (let x = 0; x < 32; x++) {
            const r = NN(x, y)
            // console.log(r)
            ctx.fillStyle = `rgb(${r * 255}, ${0}, ${0})`
            ctx.fillRect(x + 100, y, x + 1, y + 1);
        }
    }
}, 3000)


const inputArray = [0, 1, 2, 3, 4, 5];

const outputArray = [];



console.log(outputArray);
