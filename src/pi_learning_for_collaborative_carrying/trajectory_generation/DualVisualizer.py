# Authors: Cheng Guo, Evelyn D'Elia
from idyntree.visualize import MeshcatVisualizer

class DualVisualizer:
    # initialize the huamn model visualizer
    def __init__(self, ml1, ml2, model1_name, model2_name,
                 robot_base_link="root_link", human_base_link="Pelvis",
                 force_scale_factor = 0.001):
        self.subject_name = model1_name
        self.pred_name = model2_name
        self.robot_base_link = robot_base_link
        self.human_base_link = human_base_link

        self.ml1 = ml1
        self.ml2 = ml2

        self.f_c_subject = None
        self.f_c_pred = None

        self.idyntree_visualizer = MeshcatVisualizer()
        super().__init__

        self.force_scale_factor = force_scale_factor


    def load_model(self, model1_color=None, model2_color=None):

        # workaround: force the default base frame to be the desired one
        # Set robot base link
        robot_desired_linkIndex = self.ml1.model().getLinkIndex(self.robot_base_link)
        self.ml1.model().setDefaultBaseLink(robot_desired_linkIndex)

        # Set human base link
        human_desired_linkIndex = self.ml2.model().getLinkIndex(self.human_base_link)
        self.ml2.model().setDefaultBaseLink(human_desired_linkIndex)

        # check the deafult base frames
        robotLinkIndex = self.ml1.model().getDefaultBaseLink()
        robotLinkName = self.ml1.model().getLinkName(robotLinkIndex)
        print("[INFO] Robot base frame is: {}".format(robotLinkName))

        humanLinkIndex = self.ml2.model().getDefaultBaseLink()
        humanLinkName = self.ml2.model().getLinkName(humanLinkIndex)
        print("[INFO] Human base frame is: {}".format(humanLinkName))

        self.idyntree_visualizer.load_model(self.ml1.model(), self.subject_name)
        self.idyntree_visualizer.load_model(self.ml2.model(), self.pred_name)

    # set model configuration
    # given s, and H_B representing joint configuration and base transform as numpy objects
    def update_models(self, s1, s2, H_B1, H_B2):

        # update the model configuration for each model
        self.idyntree_visualizer.set_multibody_system_state(H_B1[:3,3], H_B1[:3,:3], s1, self.subject_name)
        self.idyntree_visualizer.set_multibody_system_state(H_B2[:3,3], H_B2[:3,:3], s2, self.pred_name)