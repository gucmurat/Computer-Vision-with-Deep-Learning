#!/bin/bash
if [ ! -d "coco_captioning" ]; then
    sh get_datasets.sh
fi
