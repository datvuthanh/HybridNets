import numpy as np

template = """#include <cstdint>
#include <cstdlib>
#include <vector>

#include "hybridnets_cpp/prior_bbox.h"


/*** Global variable ***/
const float PRIOR_BBOX::VARIANCE[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

const std::vector<float> PRIOR_BBOX::BBOX =
{
"""
prior = np.load('anchors_256x384.npy')[0]

with open('src/prior_bbox_256x384.cpp', 'w') as file:
    file.write(template)
    for prior_box in prior[:,]:
        file.write(str(prior_box[0]) + ", " + str(prior_box[1]) + ", " + str(prior_box[2]) + ", " + str(prior_box[3]) + ",\n")
    file.write("};")
