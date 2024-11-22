# Authors: Cheng Guo, Evelyn D'Elia
import idyntree.bindings as idynbind

class URDFVisualizer:
    # initialize the huamn model visualizer
    def __init__(self, path, model_name, base_link="Pelvis",
                 color_palette = "meshcat", force_scale_factor = 0.001):
        self.subject_name = model_name
        self.base_link = base_link

        self.urdf_path = path
        self.f_c_subject = None
        self.f_c_pred = None

        self.idyntree_visualizer = idynbind.Visualizer()
        super().__init__

        visualizer_options = idynbind.VisualizerOptions()
        self.idyntree_visualizer.init(visualizer_options)
        self.idyntree_visualizer.setColorPalette(color_palette)
        self.force_scale_factor = force_scale_factor


    def load_model(self, model_color=None):
        model_Loader_init = idynbind.ModelLoader()
        model_Loader_init.loadModelFromFile(self.urdf_path, "urdf")

        # manually set the base frame to be the desired one
        desired_base_link_idx = model_Loader_init.model().getLinkIndex(self.base_link)
        model_Loader_init.model().setDefaultBaseLink(desired_base_link_idx)

        # check the base frame
        base_link_idx = model_Loader_init.model().getDefaultBaseLink()
        base_link_name = model_Loader_init.model().getLinkName(base_link_idx)
        print("[INFO] Base frame is: {}".format(base_link_name))

        self.idyntree_visualizer.addModel(model_Loader_init.model(), self.subject_name)

        # when requiring a different model color
        """ if not model_color is None:
            self.idyntree_visualizer.modelViz(self.subject_name).setModelColor(idynbind.ColorViz(idynbind.Vector4(model_color))) """


    # set model configuration
    # given s, and H_B representing joint configuration and base transform as numpy objects
    def update_model(self, s, H_B):
        s_idyntree_opti = idynbind.VectorDynSize(s)

        T_b_opti = idynbind.Transform()
        T_b_opti.fromHomogeneousTransform(idynbind.Matrix4x4(H_B))

        # update the model configuration using 'setPositions'
        self.idyntree_visualizer.modelViz(self.subject_name).setPositions(T_b_opti, s_idyntree_opti)

    # run the visualizer while reading data from yarp ports
    def run(self):
        #self.idyntree_visualizer.camera().animator().enableMouseControl()
        #self.idyntree_visualizer.run()
        self.idyntree_visualizer.draw()




class DualVisualizer:
    # initialize the huamn model visualizer
    def __init__(self, ml1, ml2, model1_name, model2_name,
                 robot_base_link="root_link", human_base_link="Pelvis",
                 color_palette = "meshcat", force_scale_factor = 0.001):
        self.subject_name = model1_name
        self.pred_name = model2_name
        self.robot_base_link = robot_base_link
        self.human_base_link = human_base_link

        self.ml1 = ml1
        self.ml2 = ml2

        self.f_c_subject = None
        self.f_c_pred = None

        self.idyntree_visualizer = idynbind.Visualizer()
        super().__init__

        visualizer_options = idynbind.VisualizerOptions()
        self.idyntree_visualizer.init(visualizer_options)
        self.idyntree_visualizer.setColorPalette(color_palette)
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

        self.idyntree_visualizer.addModel(self.ml1.model(), self.subject_name)
        self.idyntree_visualizer.addModel(self.ml2.model(), self.pred_name)

        # when requiring a different model color
        if not model1_color is None:
            self.idyntree_visualizer.modelViz(self.subject_name).setModelColor(idynbind.ColorViz(idynbind.Vector4(model1_color)))

        if not model2_color is None:
            self.idyntree_visualizer.modelViz(self.pred_name).setModelColor(idynbind.ColorViz(idynbind.Vector4(model2_color)))

    # set model configuration
    # given s, and H_B representing joint configuration and base transform as numpy objects
    def update_model(self, s1, s2, H_B1, H_B2):
        s_idyntree_opti_sub = idynbind.VectorDynSize(s1)
        s_idyntree_opti_pred = idynbind.VectorDynSize(s2)

        T_b_opti1 = idynbind.Transform()
        T_b_opti1.fromHomogeneousTransform(idynbind.Matrix4x4(H_B1))

        T_b_opti2 = idynbind.Transform()
        T_b_opti2.fromHomogeneousTransform(idynbind.Matrix4x4(H_B2))

        # update the model configuration using 'setPositions'
        self.idyntree_visualizer.modelViz(self.subject_name).setPositions(T_b_opti1, s_idyntree_opti_sub)
        self.idyntree_visualizer.modelViz(self.pred_name).setPositions(T_b_opti2, s_idyntree_opti_pred)

    # run the visualizer while reading data from yarp ports
    def run(self):
        self.idyntree_visualizer.draw()



class MultiVisualizer:
    # initialize the huamn model visualizer
    def __init__(self, path,
                 model1_name, model2_name, model3_name,
                 base_link="Pelvis",
                 color_palette = "meshcat", force_scale_factor = 0.001):
        self.subject_name = model1_name
        self.pred_name = model2_name
        self.pred2_name = model3_name

        self.base_link = base_link

        self.urdf_path = path
        self.f_c_subject = None
        self.f_c_pred = None
        self.f_c_pred2 = None

        self.idyntree_visualizer = idynbind.Visualizer()
        super().__init__

        visualizer_options = idynbind.VisualizerOptions()
        self.idyntree_visualizer.init(visualizer_options)
        self.idyntree_visualizer.setColorPalette(color_palette)
        self.force_scale_factor = force_scale_factor


    def load_model(self, model1_color=None, model2_color=None, model3_color=None):
        model_subject_Loader_init = idynbind.ModelLoader()
        model_pred_Loader_init = idynbind.ModelLoader()
        model_pred2_Loader_init = idynbind.ModelLoader()

        model_subject_Loader_init.loadModelFromFile(self.urdf_path, "urdf")
        model_pred_Loader_init.loadModelFromFile(self.urdf_path, "urdf")
        model_pred2_Loader_init.loadModelFromFile(self.urdf_path, "urdf")

        # workaround: force the default base frame to be the desired one
        desired_linkIndex_sub = model_subject_Loader_init.model().getLinkIndex(self.base_link)
        model_subject_Loader_init.model().setDefaultBaseLink(desired_linkIndex_sub)

        desired_linkIndex_pred = model_pred_Loader_init.model().getLinkIndex(self.base_link)
        model_pred_Loader_init.model().setDefaultBaseLink(desired_linkIndex_pred)

        desired_linkIndex_pred2 = model_pred2_Loader_init.model().getLinkIndex(self.base_link)
        model_pred2_Loader_init.model().setDefaultBaseLink(desired_linkIndex_pred2)

        # check the deafult base frame
        linkIndex = model_subject_Loader_init.model().getDefaultBaseLink()
        linkName = model_subject_Loader_init.model().getLinkName(linkIndex)
        print("[INFO] Base frame is: {}".format(linkName))

        self.idyntree_visualizer.addModel(model_subject_Loader_init.model(), self.subject_name)
        self.idyntree_visualizer.addModel(model_pred_Loader_init.model(), self.pred_name)
        self.idyntree_visualizer.addModel(model_pred2_Loader_init.model(), self.pred2_name)

        # when requiring a different model color
        if not model1_color is None:
            self.idyntree_visualizer.modelViz(self.subject_name).setModelColor(idynbind.ColorViz(idynbind.Vector4(model1_color)))

        if not model2_color is None:
            self.idyntree_visualizer.modelViz(self.pred_name).setModelColor(idynbind.ColorViz(idynbind.Vector4(model2_color)))

        if not model3_color is None:
            self.idyntree_visualizer.modelViz(self.pred2_name).setModelColor(idynbind.ColorViz(idynbind.Vector4(model3_color)))

    # set model configuration
    # given s, and H_B representing joint configuration and base transform as numpy objects
    def update_model(self, s1, s2, s3, H_B1, H_B2, H_B3):
        s_idyntree_opti_sub = idynbind.VectorDynSize(s1)
        s_idyntree_opti_pred = idynbind.VectorDynSize(s2)
        s_idyntree_opti_pred2 = idynbind.VectorDynSize(s3)

        T_b_opti1 = idynbind.Transform()
        T_b_opti1.fromHomogeneousTransform(idynbind.Matrix4x4(H_B1))

        T_b_opti2 = idynbind.Transform()
        T_b_opti2.fromHomogeneousTransform(idynbind.Matrix4x4(H_B2))

        T_b_opti3 = idynbind.Transform()
        T_b_opti3.fromHomogeneousTransform(idynbind.Matrix4x4(H_B3))

        # update the model configuration using 'setPositions'
        self.idyntree_visualizer.modelViz(self.subject_name).setPositions(T_b_opti1, s_idyntree_opti_sub)
        self.idyntree_visualizer.modelViz(self.pred_name).setPositions(T_b_opti2, s_idyntree_opti_pred)
        self.idyntree_visualizer.modelViz(self.pred2_name).setPositions(T_b_opti3, s_idyntree_opti_pred2)

    # run the visualizer while reading data from yarp ports
    def run(self):
        self.idyntree_visualizer.draw()