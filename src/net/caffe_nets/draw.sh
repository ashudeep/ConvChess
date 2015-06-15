#! /bin/bash
net_file=$1.prototxt
dest=drawings/$1.png
echo "Drawing the network in "$net_file" to "$dest
python ~/caffe/python/draw_net.py $net_file $dest
