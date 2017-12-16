import numpy
from flask import Flask, jsonify, render_template, request
from PIL import Image
# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((numpy.array(request.json, dtype=numpy.uint8))).reshape(28, 28)
    print(input.shape)
    #input : (28,28) ndarray
    #output : list 10 double numbers
    return jsonify(results=[0.1,0.1,0.2,0.05,0.05,0.1,0.1,0.1,0.1,0.1])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()