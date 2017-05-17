#!/usr/bin/env sh

CAFFE_ROOT=/home/leo/project/application/caffe
DEEP_OCR_ROOT=/home/leo/project/code/deep_ocr
EXAMPLE=$DEEP_OCR_ROOT
DATA=$DEEP_OCR_ROOT/workspace/caffe_dataset
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/caffe train --solver=$DEEP_OCR_ROOT/data/caffe_nets/all/lenet_solver.prototxt $@
