#!/bin/bash
whoami

# set -e

modelName=$1

# run the container
# containerName=$2
# sudo docker start ${containerName}
# sudo docker attach ${containerName}

#activate the conda virtual environment: onnx
conda activate onnx

python onnxModelParser.py -h
python onnxModelParser.py -i ${modelName} -p
python onnxModelParser.py -i ${modelName} -gi
python onnxModelParser.py -i ${modelName} -go
python onnxModelParser.py -i ${modelName} -ga ""/model.0/conv/Conv""
python onnxModelParser.py -i ${modelName} -gs "images"
python onnxModelParser.py -i ${modelName} -gs "/model.0/conv/Conv_output_0"
python onnxModelParser.py -i ${modelName} -gs "model.0.conv.weight"
python onnxModelParser.py -i ${modelName} -gap
python onnxModelParser.py -i ${modelName} -gfep
python onnxModelParser.py -i ${modelName} -om
python onnxModelParser.py -i ${modelName} -om -ol 0
python onnxModelParser.py -i ${modelName} -om -ol 1
python onnxModelParser.py -i ${modelName} -om -ol 2
python onnxModelParser.py -i ${modelName} -om -ol 3
python onnxModelParser.py -i ${modelName} -si
python onnxModelParser.py -i ${modelName} -ir '[3, 320, 320]'

#deactivate the conda virtual environment: onnx
conda deactivate

# exit the container
# exit
# stop the container
# sudo docker stop ${containerName}

