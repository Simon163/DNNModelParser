import sys
import os
import argparse
import onnx
import onnxoptimizer
import onnx.helper
import onnx.checker
import onnx.utils
import onnx.shape_inference
import onnx.numpy_helper
import onnxruntime as onnxrt
import numpy as np
from IModelParser import IModelParser


class OnnxModelParser(IModelParser):
    '''
    Parse the onnx model, for example, get the input and output of the model, etc.
    '''

    def __init__(self, input_model):
        IModelParser.__init__(self)

        self.input_model = input_model
        print("input model: ", self.input_model)
        if not os.path.exists(self.input_model):
            print("error: the input model {} doesn't exists!".format(
                self.input_model))
            sys.exit(1)

        if not self.input_model.endswith('.onnx'):
            print("error: the input model doesn't end with '.onnx'!".format(
                self.input_model))
            sys.exit(1)

        model_path, model_name = os.path.split(self.input_model)
        model_name, _ = os.path.splitext(model_name)
        self.model_name = model_name
        self.model_path = model_path

        # layers info
        self.ops_info = []

        # output excel file name, used to save the model info
        self.output_xlsx_name = os.path.join(self.model_path,
                                             self.model_name + ".xlsx")
        # load the ONNX model
        self.model = onnx.load(self.input_model)

        # Check that the IR is well formed
        try:
            onnx.checker.check_model(self.model)
        except onnx.checker.ValidationError:
            print("error: check the model failed")
            sys.exit(1)

        # calculate the shape for each layer
        self.shape_inference()
        # parse the shaped model
        self.__parse()

    def __parse(self):
        '''
        Parse the onnx model, and get the model info
        '''
        self.max_input_length = 1
        self.max_output_length = 1
        # Get the ir version
        self.ir_version = self.model.ir_version
        # Get the opset
        self.opset = self.model.opset_import
        # Get the producer_name
        self.producer_name = self.model.producer_name
        # Get the graph
        self.graph = self.model.graph
        # Get the node
        self.node = self.graph.node
        # Get the initializer
        self.initializer = self.graph.initializer
        # Get the inputs
        self.input = self.graph.input
        # Get the outputs
        self.output = self.graph.output
        # Get value info
        self.value_info = self.graph.value_info
        # Get the model info
        self.doc_string = self.graph.doc_string
        # build inference session
        self.session = onnxrt.InferenceSession(self.model.SerializeToString())

    def print_model(self):
        '''
        Print the details of the model info
        '''
        print("---------------model info:---------------")
        print('ir_version: {}'.format(self.model.ir_version))
        print('producer_name: {}'.format(self.model.producer_name))
        print('doc_string: {}'.format(self.model.doc_string))
        print('opset: {}'.format(self.model.opset_import))
        print('opset: {}'.format(type(self.model.opset_import[0])))

        print("---------------printable_graph:---------------")
        # Print a human readable representation of the graph
        onnx.helper.printable_graph(self.graph)
        with open(
                os.path.join(self.model_path,
                             self.model_name + "_printable_graph.txt"),
                'w') as f:
            f.write(str(self.graph))

        print("---------------all nodes:---------------")
        for i in range(len(self.node)):
            print(self.node[i])

        # print("---------------initialize names:---------------")
        # for i in range(len(self.initializer)):
        #     print('\t{}'.format(self.initializer[i].name))

        print("---------------all inputs:---------------")
        for i in range(len(self.input)):
            print('\t{}'.format(self.input[i].name))
            print('\t{}'.format(self.input[i].type.tensor_type.shape.dim))

        print("---------------all outputs:---------------")
        for i in range(len(self.output)):
            print('\t{}'.format(self.output[i].name))
            print('\t{}'.format(self.output[i].type.tensor_type.shape.dim))

    def get_available_passes(self):
        '''
        Get the available passes
        '''
        return onnxoptimizer.get_available_passes()

    def get_fuse_and_elimination_passes(self):
        '''
        Get the fuse and elimination passes
        '''
        return onnxoptimizer.get_fuse_and_elimination_passes()

    def __optimize_model_level(self, optimization_level):
        '''
        optimize the model by level and save the optimized model.
        optimization_level should be in the list [0, 1, 2, 3]

        '''
        sess_options = onnxrt.SessionOptions()

        # Set graph optimization level
        sess_options.graph_optimization_level = onnxrt.GraphOptimizationLevel(
            optimization_level)

        optimized_model_name = os.path.join(
            self.model_path, self.model_name + "_optimized_level_" +
            str(optimization_level) + ".onnx")
        # To enable model serialization after graph optimization set this
        sess_options.optimized_model_filepath = optimized_model_name

        onnxrt.InferenceSession(self.input_model, sess_options)
        print("optimization level {} is used to optimize the model: ".format(
            optimization_level))
        print('save optimized model to {}, done'.format(optimized_model_name))
        return True

    def __optimize_model_pass(self, passes=None):
        '''
        optimize the model by passes.
        runs model checker, optimizer, shape inference engine on the model,
        and also strips the doc_string.
        by default, fuse and elimination passes are applied.
        This function can also optimize the model with the specified passes.
        '''
        optimized_model_name = os.path.join(
            self.model_path, self.model_name + "_optimized_model.onnx")

        # Check that the IR is well formed
        try:
            onnx.checker.check_model(self.model)
        except onnx.checker.ValidationError:
            print("error: check the model failed")
            return False

        self.model = onnx.shape_inference.infer_shapes(self.model)

        used_passes = self.get_fuse_and_elimination_passes()
        if passes is None:
            used_passes = used_passes
        else:
            used_passes = passes

        print("The following optimization passes are used: ")
        for p in used_passes:
            print(p)
        print()

        self.model = onnxoptimizer.optimize(self.model, used_passes)
        if self.model is None:
            print('optimize model with passes failed')
            return False

        # Check that the IR is well formed
        try:
            onnx.checker.check_model(self.model)
        except onnx.checker.ValidationError:
            print("error: check the model failed")
            return False

        # save new model
        onnx.save(self.model, optimized_model_name)
        print('save optimized model to {}, done'.format(optimized_model_name))
        return True

    def optimize_model(self, optimization_level=-1, passes=None):
        '''
        optimize the model by the optimization level or the passes
        '''
        # if optimization level is valid, optimize the model by the optimization level
        if optimization_level in range(0, 4):
            self.__optimize_model_level(optimization_level)
            return True

        # otherwise, optimize the model by the passes
        return self.__optimize_model_pass(passes)

    def shape_inference(self):
        '''
        generate the shape information of the model and save it to the model file
        '''
        self.model = onnx.shape_inference.infer_shapes(self.model)
        # Check that the IR is well formed
        try:
            onnx.checker.check_model(self.model)
        except onnx.checker.ValidationError:
            print("error: check the model failed")
            return False

        shape_inference_model_name = os.path.join(
            self.model_path, self.model_name + "_shape_inference.onnx")
        # save the model to the file
        onnx.save(self.model, shape_inference_model_name)
        print('save shaped inference model to {}, done'.format(
            shape_inference_model_name))
        return True

    def get_inputs(self):
        '''
        get the name and shape of the inputs of the model
        '''
        inputs = self.session.get_inputs()
        input_names = [str(input.name) for input in inputs]
        input_shape = [input.shape for input in inputs]
        return input_names, input_shape

    def get_outputs(self):
        '''
        get the name and shape of the outputs of the model
        '''
        outputs = self.session.get_outputs()
        output_names = [str(output.name) for output in outputs]
        output_shape = [output.shape for output in outputs]
        return output_names, output_shape

    def get_attributes(self, node_name):
        '''
        get all attribute values of the specified node
        '''
        op_num = len(self.node)
        for idx in range(op_num):
            op = self.node[idx]
            if op.name == node_name:
                return op.attribute
        return False

    def __get_attribute(self, node, attribute_name):
        '''
        get the value of the specified node and attribute name
        '''
        for idx in range(len(node.attribute)):
            if node.attribute[idx].name == attribute_name:
                return node.attribute[idx]
        return False

    def get_shape(self, tensor_name):
        '''
        get the shape of the specified tensor
        '''
        # the specified tensor is an input
        input_names, input_shape = self.get_inputs()
        for idx, name in enumerate(input_names):
            if name == tensor_name:
                return input_shape[idx]

        # the specified tensor is an output
        output_names, output_shape = self.get_outputs()
        for idx, name in enumerate(output_names):
            if name == tensor_name:
                return output_shape[idx]
        # the specified tensor is either input or output of a layer
        for idx in range(len(self.value_info)):
            if self.value_info[idx].name == tensor_name:
                return [
                    self.value_info[idx].type.tensor_type.shape.dim[kk].
                    dim_value for kk in range(
                        len(self.value_info[idx].type.tensor_type.shape.dim))
                ]

        # the specified tensor is an initializer
        for idx in range(len(self.initializer)):
            if self.initializer[idx].name == tensor_name:
                return self.initializer[idx].dims

            return False

    def get_initialize(self, initializer_name):
        for idx in range(len(self.initializer)):
            if self.initializer[idx].name == initializer_name:
                return self.initializer[idx]
        return False

    def input_reshape(self, new_input_shape):
        '''
        modify the input shape of the model
        '''
        if not isinstance(new_input_shape, list) or len(new_input_shape) != 4:
            print("input shape should be a list of integers with 3 dimensions")
            return False

        input_tensor_new = onnx.helper.make_tensor_value_info(
            name=self.input[0].name,
            elem_type=self.input[0].type.tensor_type.elem_type,
            shape=[
                self.input[0].type.tensor_type.shape.dim[0].dim_value,
                new_input_shape[0], new_input_shape[1], new_input_shape[2],
                new_input_shape[3]
            ])
        self.graph.input.remove(self.input[0])
        self.graph.input.insert(0, input_tensor_new)

        # create the model
        self.model = onnx.helper.make_model(self.graph)
        # Check that the IR is well formed
        try:
            onnx.checker.check_model(self.model)
        except onnx.checker.ValidationError:
            print("error: check the model failed")
            return False

        input_reshaped_model_name = os.path.join(
            self.model_path, self.model_name + "_input_reshaped_model.onnx")
        # save input reshaped model
        onnx.save(self.model, input_reshaped_model_name)
        print('save input reshaped model to {}, done'.format(
            input_reshaped_model_name))
        return True

    def _get_op_info(self, idx, op, op_info):
        layer = op
        layer_name = layer.name
        layer_Type = layer.op_type
        # print("idx: {}, layer_name: {}".format(idx, layer_name))

        op_info["Op_ID"] = idx
        op_info["Op_Type"] = layer_Type
        op_info["Op_Name"] = layer_name

        layer_inputs_name = list(layer.input)
        layer_outputs_name = list(layer.output)
        layer_inputs_shape = [
            self.get_shape(name) for name in layer_inputs_name
        ]
        layer_outputs_shape = [
            self.get_shape(name) for name in layer_outputs_name
        ]
        self.max_input_length = max(self.max_input_length,
                                    len(layer_inputs_name))
        self.max_output_length = max(self.max_output_length,
                                     len(layer_outputs_name))

        op_info["FLOPs"] = 0
        if self._is_conv(op_info):
            op_info["Input_Dimension_X"] = [
                layer_inputs_shape[0][3]
                if len(layer_inputs_shape[0]) == 4 else 0
            ]
            op_info["Input_Dimension_Y"] = [
                layer_inputs_shape[0][2]
                if len(layer_inputs_shape[0]) == 4 else 0
            ]
            op_info["Input_Dimension_C"] = [
                layer_inputs_shape[0][1]
                if len(layer_inputs_shape[0]) == 4 else 0
            ]
            op_info["Output_Dimension_X"] = [
                layer_outputs_shape[0][3]
                if len(layer_outputs_shape[0]) == 4 else 0
            ]
            op_info["Output_Dimension_Y"] = [
                layer_outputs_shape[0][2]
                if len(layer_outputs_shape[0]) == 4 else 0
            ]
            op_info["Output_Dimension_C"] = [
                layer_outputs_shape[0][1]
                if len(layer_outputs_shape[0]) == 4 else 0
            ]

            param_blob_w = self.get_initialize(layer_inputs_name[1])
            try:
                param_blob_b = self.get_initialize(layer_inputs_name[2])
            except:
                param_blob_b = None

            if type(param_blob_w) is onnx.TensorProto:
                op_info['Kernel_Outliers'] = self._check_outliers(
                    onnx.numpy_helper.to_array(param_blob_w).reshape(
                        onnx.numpy_helper.to_array(param_blob_w).size))

            else:
                op_info['Kernel_Outliers'] = self._check_outliers(
                    np.array(param_blob_w.float_data))

            if param_blob_b:
                if type(param_blob_b) is onnx.TensorProto:
                    op_info['Kernel_Outliers'] += self._check_outliers(
                        onnx.numpy_helper.to_array(param_blob_b).reshape(
                            onnx.numpy_helper.to_array(param_blob_b).size))

                else:
                    op_info['Kernel_Outliers'] += self._check_outliers(
                        np.array(param_blob_b.float_data))
            else:
                op_info['Kernel_Outliers'] += ["", "", "", "", ""]

            attribute = self.__get_attribute(layer, "kernel_shape")
            if attribute:
                op_info['Kernel_Dimension_X'] = list(attribute.ints)[0]
                op_info['Kernel_Dimension_Y'] = list(attribute.ints)[1]
            else:
                op_info['Kernel_Dimension_X'] = ""
                op_info['Kernel_Dimension_Y'] = ""

            attribute = self.__get_attribute(layer, "pads")
            if attribute:
                op_info['Pad_Dimension_X'] = list(attribute.ints)[0]
                op_info['Pad_Dimension_Y'] = list(attribute.ints)[1]
            else:
                op_info['Pad_Dimension_X'] = 0
                op_info['Pad_Dimension_Y'] = 0

            attribute = self.__get_attribute(layer, "strides")
            if attribute:
                op_info['Stride_Dimension_X'] = list(attribute.ints)[0]
                op_info['Stride_Dimension_Y'] = list(attribute.ints)[1]
            else:
                op_info['Stride_Dimension_X'] = 1
                op_info['Stride_Dimension_Y'] = 1

            attribute = self.__get_attribute(layer, "dilations")
            if attribute:
                op_info['Dilate_Dimension_X'] = list(attribute.ints)[0]
                op_info['Dilate_Dimension_Y'] = list(attribute.ints)[1]
            else:
                op_info['Dilate_Dimension_X'] = 1
                op_info['Dilate_Dimension_Y'] = 1

            attribute = self.__get_attribute(layer, "auto_pad")
            if attribute:
                op_info['Pad_Mode'] = attribute.s
            else:
                op_info['Pad_Mode'] = "NOTSET"

            attribute = self.__get_attribute(layer, "group")
            if attribute:
                op_info['Group'] = attribute.i
            else:
                op_info['Group'] = 1

            # unit: MFLOPs
            op_info["FLOPs"] = op_info["Output_Dimension_X"][0] * op_info["Output_Dimension_Y"][0] * \
                op_info["Output_Dimension_C"][0] * op_info['Input_Dimension_C'][0] * op_info[
                'Kernel_Dimension_X'] * op_info[
                'Kernel_Dimension_Y'] / 1024 / 1024 if op_info["Input_Dimension_C"] and op_info["Output_Dimension_C"] else 0
        elif layer_Type in ["MaxPool", "AveragePool"]:
            op_info['Pooling_Mode'] = layer_Type

            attribute = self.__get_attribute(layer, "kernel_shape")
            if attribute:
                op_info['Kernel_Dimension_X'] = list(attribute.ints)[0]
                op_info['Kernel_Dimension_Y'] = list(attribute.ints)[1]
            else:
                op_info['Kernel_Dimension_X'] = ""
                op_info['Kernel_Dimension_Y'] = ""

            attribute = self.__get_attribute(layer, "pads")
            if attribute:
                op_info['Pad_Dimension_X'] = list(attribute.ints)[0]
                op_info['Pad_Dimension_Y'] = list(attribute.ints)[1]
            else:
                op_info['Pad_Dimension_X'] = 0
                op_info['Pad_Dimension_Y'] = 0

            attribute = self.__get_attribute(layer, "strides")
            if attribute:
                op_info['Stride_Dimension_X'] = list(attribute.ints)[0]
                op_info['Stride_Dimension_Y'] = list(attribute.ints)[1]
            else:
                op_info['Stride_Dimension_X'] = 1
                op_info['Stride_Dimension_Y'] = 1

        elif layer_Type in ["Split", "Concat"]:
            pass
        elif layer_Type in ["Relu", "PRelu", "LeakyRelu"]:
            op_info["Activate_Mode"] = layer_Type
        return op_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True,
                        help="input model, the onnx model file to be parsed")
    parser.add_argument("-d",
                        "--debug",
                        action='store_true',
                        default=False,
                        help="debug mode, 0: disabled, 1: enabled, default: 0")
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=5678,
        help="set the port number of the debugger, default: 5678")
    parser.add_argument(
        "-p",
        "--print",
        action='store_true',
        default=False,
        help="print model infomation, 0: disabled, 1: enabled, default: 0")
    parser.add_argument(
        "-gi",
        "--get_inputs",
        action='store_true',
        default=False,
        help=
        "get the names and shapes of inputs, 0: disabled, 1: enabled, default: 0"
    )
    parser.add_argument(
        "-go",
        "--get_outputs",
        action='store_true',
        default=False,
        help=
        "get the names and shapes of outputs, 0: disabled, 1: enabled, default: 0"
    )
    parser.add_argument("-ga",
                        "--get_attributes",
                        type=str,
                        required=False,
                        help="get the attributes of the specified layer")
    parser.add_argument("-gs",
                        "--get_shape",
                        type=str,
                        required=False,
                        help="get the shape of the specified tensor")
    parser.add_argument(
        "-gap",
        "--get_available_passes",
        action='store_true',
        default=False,
        help="get all available passes, 0: disabled, 1: enabled, default: 0")
    parser.add_argument(
        "-gfep",
        "--get_fuse_and_elimination_passes",
        action='store_true',
        default=False,
        help=
        "get fuse and elimination passes, 0: disabled, 1: enabled, default: 0")
    parser.add_argument(
        "-om",
        "--optimize_model",
        action='store_true',
        default=False,
        help="optimize model, 0: disabled, 1: enabled, default: 0")
    parser.add_argument(
        "-ol",
        "--optimization_level",
        type=int,
        required=False,
        choices=[-1, 0, 1, 2, 3],
        default=-1,
        help=
        "set optimization level, -1: disabled, 0,1,2,3: enabled, default: 1")
    parser.add_argument(
        "-si",
        "--shape_inference",
        action='store_true',
        default=False,
        help=
        "calculate the shape information of each layer and store to the output model, 0: disabled, 1: enabled, default: 0"
    )
    parser.add_argument(
        "-ir",
        "--input_reshape",
        type=str,
        required=False,
        help=
        "modify input shape of input layer, input shape must be specified using a list with 3 dimensions like [3,224,224]"
    )
    args = parser.parse_args()
    print("input arguments: ", args)

    model_parser = OnnxModelParser(args.input)

    if args.debug:
        import debugpy
        print("port: ", args.port)
        debugpy.listen(("0.0.0.0", args.port))
        print("Waiting for client to attach...")
        debugpy.wait_for_client()

    if args.print:
        print("print model infomation...")
        model_parser.print_model()
        print("print model infomation, succeeded")

    if args.get_inputs:
        print("get the names and shapes of inputs...")
        names, shapes = model_parser.get_inputs()
        for i in range(len(names)):
            print("input", [i], ", name: " + names[i], ", shape: ", shapes[i])
        print("get the names and shapes of inputs, succeeded")

    if args.get_outputs:
        print("get the names and shapes of outputs...")
        names, shapes = model_parser.get_outputs()
        for i in range(len(names)):
            print("output", [i], ", name: " + names[i], ", shape: ", shapes[i])
        print("get the names and shapes of outputs, succeeded")

    if args.get_attributes:
        ret = model_parser.get_attributes(args.get_attributes)
        if ret is False:
            print("get the attributes of the specified layer..., failed!")
        else:
            print(args.get_attributes, ": ", ret)
            print("get the attributes of the specified layer..., succeeded")

    if args.get_shape:
        ret = model_parser.get_shape(args.get_shape)
        if ret is False:
            print("get the shape of the specified tensor..., failed!")
        else:
            print(args.get_shape, ": ", ret)
            print("get the shape of the specified tensor..., succeeded")

    if args.get_available_passes:
        print("get all available passes...")
        passes = model_parser.get_available_passes()
        for p in passes:
            print(p)
        print("get all available passes, succeeded")

    if args.get_fuse_and_elimination_passes:
        print("get the fuse and elimination passes...")
        passes = model_parser.get_fuse_and_elimination_passes()
        for p in passes:
            print(p)
        print("get the fuse and elimination passes, succeeded")

    if args.optimize_model:
        print("optimize model...")
        ret = model_parser.optimize_model(
            optimization_level=args.optimization_level)
        if ret is False:
            print("optimize model, failed!")
        else:
            print("optimize model, succeeded")

    if args.shape_inference:
        print(
            "calculate the shape information of each layer and store to the output..."
        )
        ret = model_parser.shape_inference()
        if ret is False:
            print(
                "calculate the shape information of each layer and store to the output, failed!"
            )
        else:
            print(
                "calculate the shape information of each layer and store to the output, succeeded"
            )

    if args.input_reshape:
        input_reshape = eval(args.input_reshape)
        valid = True
        if not isinstance(input_reshape, (list)):
            print(
                "error: input shape must be specified using a list like [3,224,224]!"
            )
            valid = False
        if len(input_reshape) != 3:
            print(
                "error: input shape must be specified using a list with 3 dimensions!"
            )
            valid = False
        for v in input_reshape:
            if not isinstance(v, (int)):
                print("error: input shape must be integer!")
                valid = False
            if v <= 0:
                print("error: input shape must be positive integer!")
                valid = False

        if valid:
            input_reshape = [1] + input_reshape
        else:
            print("please check the input shape!")

        print("new input shape: ", input_reshape)

        _, old_shape = model_parser.get_inputs()
        print("old input shape: ", *old_shape)

        ret = model_parser.input_reshape(input_reshape)
        if ret is None:
            print("modify input shape of input layer, failed!")

        else:
            print("modify input shape of input layer, succeeded!")
