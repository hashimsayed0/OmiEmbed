import torch
from .vae_basic_model import VaeBasicModel
from . import networks
from . import losses
from torch.nn import functional as F
import random


class VaeClassifierModel(VaeBasicModel):
    """
    This class implements the VAE classifier model, using the VAE framework with the classification downstream task.
    """

    @staticmethod
    def modify_commandline_parameters(parser, is_train=True):
        # changing the default values of parameters to match the vae regression model
        parser.add_argument('--class_num', type=int, default=0,
                            help='the number of classes for the classification task')
        return parser

    def __init__(self, param):
        """
        Initialize the VAE_classifier class.
        """
        VaeBasicModel.__init__(self, param)
        # specify the training losses you want to print out.
        self.loss_names.append('classifier')
        # specify the metrics you want to print out.
        self.metric_names = ['accuracy']
        # input tensor
        self.label = None
        # output tensor
        self.y_out = None
        self.y_out_subset = []

        if param.use_subset_features:
            if param.use_subset_identity:
                param.latent_space_dim = param.latent_space_dim + param.subset_num
            elif param.agg_method == 'concat':
                # if param.use_subset_identity:
                #     param.latent_space_dim = (param.latent_space_dim + param.subset_num) * param.subset_num
                # else:
                param.latent_space_dim = param.latent_space_dim * param.subset_num
            


        # define the network
        self.netDown = networks.define_down(param, param.net_down, param.norm_type, param.leaky_slope, param.dropout_p,
                                            param.latent_space_dim, param.class_num, None, None, param.init_type,
                                            param.init_gain, self.gpu_ids)
        # define the classification loss
        self.lossFuncClass = losses.get_loss_func('CE', param.reduction)
        self.loss_classifier = None
        self.metric_accuracy = None

        if self.isTrain:
            # Set the optimizer
            self.optimizer_Down = torch.optim.Adam(self.netDown.parameters(), lr=param.lr, betas=(param.beta1, 0.999), weight_decay=param.weight_decay)
            # optimizer list was already defined in BaseModel
            self.optimizers.append(self.optimizer_Down)

    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader

        Parameters:
            input_dict (dict): include the data tensor and its index.
        """
        VaeBasicModel.set_input(self, input_dict)
        self.label = input_dict['label'].to(self.device)

    def forward(self):

        # if self.param.use_subset_features:
        #     self.latent_subset = []
        #     self.recon_omics_subset = []
        #     self.latent_identity = F.one_hot(torch.arange(0,self.param.subset_num).to(self.device))
        #     for subset in range(self.param.subset_num):
        #         self.subset = subset
        #         VaeBasicModel.forward(self)
        #         if self.param.use_subset_identity:
        #             self.latent_subset.append(torch.cat([self.latent, self.latent_identity[subset].repeat(self.latent.shape[0], 1)], dim=1))
        #         else:
        #             self.latent_subset.append(self.latent)
        #         self.recon_omics_subset.append(self.recon_omics)
        #     if self.param.agg_method == 'mean':
        #         self.latent = torch.mean(torch.stack(self.latent_subset), axis=0)
        #     elif self.param.agg_method == 'max':
        #         self.latent = torch.max(torch.stack(self.latent_subset), axis=0)[0]
        #     elif self.param.agg_method == 'min':
        #         self.latent = torch.min(torch.stack(self.latent_subset), axis=0)[0]
        #     elif self.param.agg_method == 'sum':
        #         self.latent = torch.sum(torch.stack(self.latent_subset), axis=0)
        #     elif self.param.agg_method == 'concat':
        #         self.latent = torch.cat(self.latent_subset, axis=1)
        if self.param.use_subset_features:
            self.latent_subset = []
            self.recon_omics_subset = []
            self.y_out_subset = []
            self.latent_identity = F.one_hot(torch.arange(0,self.param.subset_num).to(self.device))
            for subset in range(self.param.subset_num):
                self.subset = subset
                VaeBasicModel.forward(self)
                if self.param.use_subset_identity:
                    self.latent = torch.cat([self.latent, self.latent_identity[subset].repeat(self.latent.shape[0], 1)], dim=1)
                    self.y_out = self.netDown(self.latent)
                    self.y_out_subset.append(self.y_out)
                else:
                    self.latent_subset.append(self.latent)
                self.recon_omics_subset.append(self.recon_omics)
            if self.param.use_subset_identity:
                # self.y_out = torch.mean(torch.stack(self.y_out_subset), axis=0)
                if self.param.agg_method == 'mean':
                    self.y_out = torch.mean(torch.stack(self.y_out_subset), axis=0)
                elif self.param.agg_method == 'max':
                    self.y_out = torch.max(torch.stack(self.y_out_subset), axis=0)[0]
                elif self.param.agg_method == 'min':
                    self.y_out = torch.min(torch.stack(self.y_out_subset), axis=0)[0]
                elif self.param.agg_method == 'sum':
                    self.y_out = torch.sum(torch.stack(self.y_out_subset), axis=0)
                elif self.param.agg_method == 'concat':
                    self.y_out = torch.cat(self.y_out_subset, axis=1)
                elif self.param.agg_method == 'random':
                    self.y_out = self.y_out_subset[random.randrange(0, self.param.subset_num)]
            else:
                if self.param.agg_method == 'mean':
                    self.latent = torch.mean(torch.stack(self.latent_subset), axis=0)
                elif self.param.agg_method == 'max':
                    self.latent = torch.max(torch.stack(self.latent_subset), axis=0)[0]
                elif self.param.agg_method == 'min':
                    self.latent = torch.min(torch.stack(self.latent_subset), axis=0)[0]
                elif self.param.agg_method == 'sum':
                    self.latent = torch.sum(torch.stack(self.latent_subset), axis=0)
                elif self.param.agg_method == 'concat':
                    self.latent = torch.cat(self.latent_subset, axis=1)
                elif self.param.agg_method == 'random':
                    self.latent = self.latent_subset[random.randrange(0, self.param.subset_num)]
                # Get the output tensor
                self.y_out = self.netDown(self.latent)
        else:
            VaeBasicModel.forward(self)
            # Get the output tensor
            self.y_out = self.netDown(self.latent)


        # if self.param.use_subset_features:
        #     VaeBasicModel.forward(self)
        #     # Get the output tensor
        #     self.y_out = self.netDown(self.latent)
        #     # if self.isTrain:
        #     #     VaeBasicModel.forward(self)
        #     #     # Get the output tensor
        #     #     self.y_out = self.netDown(self.latent)
        #     # else:
        #     #     self.y_out_subset = []
        #     #     for subset in range(self.param.subset_num):
        #     #         self.subset = subset
        #     #         VaeBasicModel.forward(self)
        #     #         # Get the output tensor
        #     #         self.y_out_subset.append(self.netDown(self.latent))
        #     #     self.y_out = torch.mean(torch.stack(self.y_out_subset), axis=0)
        # else:
        #     VaeBasicModel.forward(self)
        #     # Get the output tensor
        #     self.y_out = self.netDown(self.latent)
        

    def cal_losses(self):
        """Calculate losses"""
        
        if self.param.use_subset_features:
            self.loss_embed_subset = []
            for subset in range(self.param.subset_num):
                self.recon_omics = self.recon_omics_subset[subset]
                VaeBasicModel.cal_losses(self)
                self.loss_embed_subset.append(self.loss_embed)
            self.loss_embed = sum(self.loss_embed_subset)
        else:
            VaeBasicModel.cal_losses(self)

        # Calculate the classification loss (downstream loss)
        self.loss_classifier = self.lossFuncClass(self.y_out, self.label)
        # LOSS DOWN
        self.loss_down = self.loss_classifier

        self.loss_All = self.param.k_embed * self.loss_embed + self.loss_down

        # VaeBasicModel.cal_losses(self)
        # # Calculate the classification loss (downstream loss)
        # self.loss_classifier = self.lossFuncClass(self.y_out, self.label)
        # # LOSS DOWN
        # self.loss_down = self.loss_classifier

        # self.loss_All = self.param.k_embed * self.loss_embed + self.loss_down

    def update(self):
        VaeBasicModel.update(self)

    def get_down_output(self):
        """
        Get output from downstream task
        """
        with torch.no_grad():
            y_prob = F.softmax(self.y_out, dim=1)
            _, y_pred = torch.max(y_prob, 1)

            index = self.data_index
            y_true = self.label

            return {'index': index, 'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}

    def calculate_current_metrics(self, output_dict):
        """
        Calculate current metrics
        """
        self.metric_accuracy = (output_dict['y_true'] == output_dict['y_pred']).sum().item() / len(output_dict['y_true'])
