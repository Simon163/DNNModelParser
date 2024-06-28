import numpy as np


class IModelParser():
    '''
    interface of model parser
    '''

    def __init__(self):
        self.ops_info = []
        self.max_input_length = 0
        self.max_output_length = 0

    def _is_conv_or_convdw(self, op_info):
        '''
        check if the op is convolution layer or depthwise convolution layer,
        if yes, return True, else return False
        '''
        if self._is_conv(op_info) or self._is_convdw(op_info):
            return True
        else:
            return False

    def _is_conv(self, op_info):
        '''
        check if the op is convolution layer,
        if yes, return True, else return False
        '''
        if op_info['Op_Type'] in [
                "Convolution",  # caffe
                "Conv2D",  # tf
                "Conv",  #onnx
                "ConvTranspose",  # onnx
        ]:
            return True
        else:
            return False

    def _is_convdw(self, op_info):
        '''
        check if the op is depthwise convolution layer,
        if yes, return True, else return False
        '''
        if op_info['Op_Type'] in [
                "ConvolutionDepthwise",  # caffe
                "DepthwiseConv2dNative",  # tf
        ]:
            return True
        else:
            return False

    def _get_sum_flops(self):
        '''
        get the sum of FLOPs of all ops in the model
        '''
        sum_flops = 0
        for op_info in self.ops_info:
            sum_flops += op_info["FLOPs"]
        return sum_flops

    def _get_empty_op_info(self):
        return {
            "Op_ID": "",
            "Op_Type": "",
            "Op_Name": "",
            "FLOPs": "",
            "Input_Dimension_X": [],
            "Input_Dimension_Y": [],
            "Input_Dimension_C": [],
            "Output_Dimension_X": [],
            "Output_Dimension_Y": [],
            "Output_Dimension_C": [],
            "Kernel_Dimension_X": "",
            "Kernel_Dimension_Y": "",
            'Kernel_Outliers': "",
            "Pad_Dimension_X": "",
            "Pad_Dimension_Y": "",
            "Stride_Dimension_X": "",
            "Stride_Dimension_Y": "",
            "Dilate_Dimension_X": "",
            "Dilate_Dimension_Y": "",
            "Pad_Mode": "",
            "Group": "",
            "Pooling_Mode": "",
            "Activate_Mode": ""
        }

    def _get_op_info(self):
        pass

    def _check_outliers(self, array):
        ''' check if array has nan or inf, if so, set to 0,
            return the number of nan and inf, and max, min, mean of array
            Args:
                array: numpy array
            Returns:
                list: the number of nan and inf, and max, min, mean of array
        '''
        # check nan
        where_nan = np.where(np.isnan(array))
        num_nan = len(where_nan[0])
        for k in range(len(where_nan[0])):
            array[[index[k] for index in where_nan]] = 0

        #check inf
        where_inf = np.where(np.isinf(array))
        num_inf = len(where_inf[0])
        for k in range(len(where_inf[0])):
            array[[index[k] for index in where_inf]] = 0

        mean = array.mean()
        max = array.max()
        min = array.min()
        return [num_nan, num_inf, max, min, mean]
