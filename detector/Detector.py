from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, img):
        predictions = self.predictor(img)

        viz = Visualizer(img[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))

        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        return (output.get_image()[:,:,::-1], predictions)

