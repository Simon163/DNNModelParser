import os
import sys
import torch
import torchvision
import torchview
import torchinfo
import graphviz

from IModelParser import IModelParser
from demo import myCustomPytorchModel


class PytorchModelParser(IModelParser):
    '''
    Parse the pytorch model, for example, get the model summary or visualize the model, etc.
    '''

    def __init__(self, input_model):
        super(PytorchModelParser, self).__init__()

        torch.backends.nnpack.enabled = False

        if isinstance(input_model, str):
            self.input_model = input_model
            print("input model: ", self.input_model)
            if not os.path.exists(self.input_model):
                print("error: the input model {} doesn't exists!".format(
                    self.input_model))
                sys.exit(1)

            if not self.input_model.endswith('.pt'):
                print("error: the input model is not a valid Pytorch model!".
                      format(self.input_model))
                sys.exit(1)

            model_path, model_name = os.path.split(self.input_model)
            model_name, _ = os.path.splitext(model_name)
            self.model_name = model_name
            self.model_path = model_path
            self.model = torch.load(self.input_model)
        elif isinstance(input_model, torch.nn.Module):
            self.model = input_model
            self.model_name = input_model.__class__.__name__.lower()
            self.model_path = os.path.dirname(myCustomPytorchModel.__file__)
            print(
                f"custom, model name: {self.model_name}, model path: {self.model_path}"
            )

        self.model.eval()

    def print_model_details(self):
        '''
        Print the details of the Pytorch model.
        '''
        print("Pytorch model details: ")
        print(self.model)

    def visualize_model(self, input_size=(1, 3, 224, 224), save_graph=True):
        '''
        Visualize the Pytorch model using torchview.
        '''

        torchview.draw_graph(self.model,
                             input_size=input_size,
                             save_graph=save_graph,
                             hide_inner_tensors=False,
                             hide_module_functions=False,
                             filename=self.model_name,
                             directory=self.model_path)

    def export_to_onnx(self, dummy_input):
        '''
        Export the model to ONNX format.
        '''
        input_names = ['input']
        output_names = ['output']
        torch.onnx.export(self.model,
                          dummy_input,
                          os.path.join(self.model_path,
                                       self.model_name + ".onnx"),
                          input_names=input_names,
                          output_names=output_names)

    def get_torch_hub_list(self, repo_name="pytorch/vision"):
        '''
        List models available in a PyTorch Hub repository.
        '''
        try:
            models = torch.hub.list(repo_name, force_reload=True)
            print(f"models available in '{repo_name}': ")
            # print(*models, sep="\n")
            for idx, model in enumerate(models):
                print(f"{idx: <3} : {model : <20}")
        except Exception as e:
            print(f"error listing models from '{repo_name}': {e}")

    def get_torch_hub_model(self,
                            repo_name="pytorch/vision",
                            model_name="resnet18",
                            save=None):
        '''
        Load a model from a PyTorch Hub repository.
        '''
        try:
            model = torch.hub.load(repo_name, model_name)
            model.eval()
            print(
                f"loaded '{model_name}' model from '{repo_name}' repository.")

            if save:
                torch.save(model, f"{model_name}.pt")
                print(f"saved '{model_name}' model to '{model_name}.pt'")
            return model
        except Exception as e:
            print(
                f"error loading '{model_name}' model from '{repo_name}': {e}")
            self.get_torch_hub_list(repo_name)
            return None

    def load_model_from_file(self, model_file):
        '''
        Load a model from a file.
        '''
        model = torch.load(model_file).eval()
        print(f"loaded model from '{model_file}'")
        return model

    def predict(self, input_data):
        '''
        Make predictions on input data.
        '''
        with torch.no_grad():
            output = self.model.forward(input_data)
            return output

    def print_model_summary(self, input_size=(3, 224, 224)):
        '''
        Print and save a summary of the Pytorch model.
        '''
        summary_str = str(
            torchinfo.summary(model=self.model,
                              input_size=input_size,
                              batch_dim=0,
                              col_names=("input_size", "output_size",
                                         "kernel_size", "num_params",
                                         "params_percent", "mult_adds",
                                         "trainable"),
                              mode="eval",
                              verbose=0))
        print(summary_str)

        with open(
                os.path.join(self.model_path,
                             self.model_name + "_summary.txt"), "w") as f:
            f.write(summary_str)


if __name__ == "__main__":
    # Initialize model parser with a PyTorch model file
    model_parser = PytorchModelParser("resnet18.pt")
    # model_parser = PytorchModelParser("yolov5s.pt")

    # print model details
    model_parser.print_model_details()

    # convert pytorch model to onnx model
    input_data = torch.randn(1, 3, 224, 224)
    model_parser.export_to_onnx(input_data)

    # visualize model and save the model graph in .png file
    model_parser.visualize_model()

    model_parser.get_torch_hub_list()
    # model_parser.get_torch_hub_list(repo_name="pytorch/vision")
    model_parser.get_torch_hub_list(repo_name="ultralytics/yolov5")
    model = model_parser.get_torch_hub_model(repo_name="ultralytics/yolov5",
                                             model_name="yolov5s",
                                             save=True)

    # predict
    input_data = torch.randn((2, 3, 224, 224))
    output = model_parser.predict(input_data)
    print(output.size())

    # print and save model summary
    input_size = (3, 224, 224)
    model_parser.print_model_summary(input_size)

    # Initialize model parser with a custom PyTorch model
    my_model = myCustomPytorchModel.MyCustomModel()
    model_parser = PytorchModelParser(my_model)
    # print model details
    model_parser.print_model_details()

    # convert pytorch model to onnx model
    input_data = torch.randn(1, 100)
    model_parser.export_to_onnx(input_data)

    # visualize model and save the model graph in .png file
    model_parser.visualize_model(input_size=(1, 100))

    # predict
    input_data = torch.randn((1, 100))
    output = model_parser.predict(input_data)
    print(output.size())

    # print and save model summary
    input_size = (1, 100)
    model_parser.print_model_summary(input_size)
