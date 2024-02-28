var BOARD = "vg";

function thread_link(board, id) {
	return `https://a.4cdn.org/${board}/thread/${id}.json`;
}
function choice(array) {
	return array[Math.floor(Math.random() * array.length) - 1]
}
const { convert } = require('html-to-text');
// Import library 
var tf = require("@tensorflow/tfjs-node")
const fs = require('fs');
var LENGTH = 256;
// Create embedding layer 
const embeddingLayer = tf.layers.embedding({ 
inputDim: 1024, 
outputDim: 1, 
inputLength: LENGTH 
}); 

function isASCII(str) {
    return /^[\x00-\x7F]*$/.test(str);
}


function encodeString(str) {
	var a = [];
	for (var i = 0; i < str.length; i++) {
		a.push(str.charCodeAt(i));
	}
	return a
}
function decodeString(array) {
	var a = "";
	array.forEach(char => {
		a += String.fromCharCode(char)
		
	})
	return a
}


var tmodel = tf.sequential();
tmodel.add(tf.layers.inputLayer({inputShape: [1], activation: "linear" }));
tmodel.add(tf.layers.dense({units: 2048, activation: "linear"}));
tmodel.add(tf.layers.dense({units: 2048*2, activation: "linear"}));
tmodel.add(tf.layers.dense({units: 2048, activation: "linear"}));
tmodel.add(tf.layers.dense({units: 1, activation: "linear"}));


(async () => {
tmodel = await tf.loadLayersModel('file://transformer/model.json');
tmodel.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(0.00004)});

var model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [LENGTH, 1], activation: "linear" }));
model.add(tf.layers.dense({units: 2048, activation: "linear"}));
model.add(tf.layers.dense({units: 2048, activation: "linear"}));
model.add(tf.layers.flatten())
model.add(tf.layers.dense({units: LENGTH, activation: "linear"}));
model = await tf.loadLayersModel('file://decoder/model.json');
model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(0.00004)});



// Get random thread
fetch(`https://a.4cdn.org/${BOARD}/threads.json`, {})
.then(r => r.json())
.then(json => {
	var pages = json.map(page => page.threads)
	var threads = []
	pages.forEach(page => {
		page.forEach(t => {
			threads.push(t.no);
		})
	})
	var no = choice(threads);
	
	var tl = thread_link(BOARD, no);

fetch(tl, { } )
.then(r => r.json())
.then(async json => {
		var a = convert(json.posts[0].com.substring(0, LENGTH))
		var b = convert((choice(json.posts)).com.substring(0, LENGTH))
		
		console.log(a);
		console.log("-->");
		console.log(b);
		
		var ta = encodeString(a.padEnd(LENGTH, "\0"));
		var tb = encodeString(b.padEnd(LENGTH, "\0"));
		const ax = tf.tensor([ta])
		const bx = tf.tensor([tb])
		
		var ae = embeddingLayer.apply(ax); 
		var be = embeddingLayer.apply(bx);
		ae = ae.squeeze().squeeze();
		be = be.squeeze().squeeze(); 
		
		//console.log(ae.dataSync(), be.dataSync());
		
		console.log("Training e2e...");
		await tmodel.fit(ae, be, { verbose: false, epochs: 16 })
		
		
		var p = model.predict(ae);
		p = tmodel.predict(p);
		
		const res = await tmodel.save('file://transformer'); 
		var txt = decodeString(p.dataSync().map( n => Math.round(n)));
		console.log("==============")
		console.log(a);
		console.log("--->");
		console.log(txt)
		console.log("==============")
		
		ax.dispose();
		bx.dispose();
		ae.dispose();
		be.dispose();
	})
	.catch(e => {
		console.log(e)
	})
})


var words = fs.readFileSync('wordlist.txt', 'utf8').toString().split("\n");
var a = words[Math.floor(Math.random()*words.length)-1]
var b = "how are you";

var ta = encodeString(a.padEnd(LENGTH, "\0"));

const input = tf.tensor([ta]) 

// Apply embedding to input 
const output = embeddingLayer.apply(input); 

console.log("Training decoder...");
await model.fit(output, input, { epochs: 120, verbose: false,  } )

var p = model.predict(output);
console.log( p.dataSync().map( n => Math.round(n)) )
var txt = decodeString(p.dataSync().map( n => Math.round(n)));

// Print the output 
console.log(txt)
const res = await model.save('file://decoder'); 
input.dispose();
output.dispose();
p.dispose();




})()




