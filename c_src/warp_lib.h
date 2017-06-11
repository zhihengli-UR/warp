int warpForward(THFloatTensor *inputImage, THFloatTensor *flow, THFloatTensor *recImage);
int warpBackward(THFloatTensor *inputImage, THFloatTensor *flow, THFloatTensor *gradFlow, THFloatTensor *gradOutput);
