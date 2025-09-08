import os
from IORT.canvas import Scene, View
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from IORT.ui.MainWindow import Ui_MainWindow
from IORT.configs import STATUSMode
from IORT.anno import Annotation
from PIL import Image
from LaMaProject.bin.object_removal import load_lama_model, inpaint
from IORT.segment_any.segment_any import SegAny, SegAnyVideo
from IORT.widgets.model_manager_dialog import ModelManagerDialog
from IORT.configs import CHECKPOINT_PATH, CONTOURMode
from IORT.segment_any.gpu_resource import GPUResource_Thread
from IORT.anno import Object
from skimage.draw.draw import polygon
from IORT.widgets.polygon import Polygon
import cv2
import functools
import torch
import orjson
import requests
import numpy as np

def calculate_area(points):
    area = 0
    num_points = len(points)
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        d = p1[0] * p2[1] - p2[0] * p1[1]
        area += d
    return abs(area) / 2


class SegAnyThread(QThread):
    tag = pyqtSignal(int, int, str)

    def __init__(self, mainwindow):
        super(SegAnyThread, self).__init__()
        self.mainwindow = mainwindow
        self.results_dict = {}
        self.index = None

    @torch.no_grad()
    def sam_encoder(self, image: np.ndarray):
        torch.cuda.empty_cache()
        with torch.inference_mode(), torch.autocast(self.mainwindow.segany.device,
                                                    dtype=self.mainwindow.segany.model_dtype,
                                                    enabled=torch.cuda.is_available()):

            # sam2 函数命名等发生很大改变，为了适应后续基于sam2的各类模型，这里分开处理sam1和sam2模型
            if 'sam2' in self.mainwindow.segany.model_type:
                _orig_hw = tuple([image.shape[:2]])
                input_image = self.mainwindow.segany.predictor_with_point_prompt._transforms(image)
                input_image = input_image[None, ...].to(self.mainwindow.segany.predictor_with_point_prompt.device)
                backbone_out = self.mainwindow.segany.predictor_with_point_prompt.model.forward_image(input_image)
                _, vision_feats, _, _ = self.mainwindow.segany.predictor_with_point_prompt.model._prepare_backbone_features(backbone_out)
                if self.mainwindow.segany.predictor_with_point_prompt.model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + self.mainwindow.segany.predictor_with_point_prompt.model.no_mem_embed
                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self.mainwindow.segany.predictor_with_point_prompt._bb_feat_sizes[::-1])
                ][::-1]
                _features = {"image_embed": feats[-1], "high_res_feats": tuple(feats[:-1])}
                return _features, _orig_hw, _orig_hw
            else:
                input_image = self.mainwindow.segany.predictor_with_point_prompt.transform.apply_image(image)
                input_image_torch = torch.as_tensor(input_image, device=self.mainwindow.segany.predictor_with_point_prompt.device)
                input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

                original_size = image.shape[:2]
                input_size = tuple(input_image_torch.shape[-2:])

                input_image = self.mainwindow.segany.predictor_with_point_prompt.model.preprocess(input_image_torch)
                features = self.mainwindow.segany.predictor_with_point_prompt.model.image_encoder(input_image)
                return features, original_size, input_size

    def run(self):
        if self.index is not None:

            # 需要缓存特征的图像索引，可以自行更改缓存策略
            indexs = []
            for i in range(-1, 2):
                cache_index = self.index + i
                if cache_index < 0 or cache_index >= len(self.mainwindow.files_list):
                    continue
                indexs.append(cache_index)

            # 先删除不需要的旧特征
            features_ks = list(self.results_dict.keys())
            for k in features_ks:
                if k not in indexs:
                    try:
                        del self.results_dict[k]
                        self.tag.emit(k, 0, '')  # 删除
                    except:
                        pass

            for index in indexs:
                if index not in self.results_dict:
                    self.tag.emit(index, 2, '')    # 进行

                    image_path = os.path.join(self.mainwindow.image_root, self.mainwindow.files_list[index])

                    image_data = np.array(Image.open(image_path).convert('RGB'))
                    try:
                        features, original_size, input_size = self.sam_encoder(image_data)
                    except Exception as e:
                        self.tag.emit(index, 3, '{}'.format(e))  # error
                        continue

                    self.results_dict[index] = {}
                    self.results_dict[index]['features'] = features
                    self.results_dict[index]['original_size'] = original_size
                    self.results_dict[index]['input_size'] = input_size

                    self.tag.emit(index, 1, '')    # 完成

                    torch.cuda.empty_cache()
                else:
                    self.tag.emit(index, 1, '')


class SegAnyVideoThread(QThread):
    tag = pyqtSignal(int, int, bool, bool, str)    # current, total, finished, is_error, message

    def __init__(self, mainwindow):
        super(SegAnyVideoThread, self).__init__()
        self.mainwindow = mainwindow
        self.start_frame_idx = 0
        self.max_frame_num_to_track = None

    def run(self):
        print('self.start_frame_idx: ', self.start_frame_idx)
        print('self.max_frame_num_to_track: ', self.max_frame_num_to_track)

        if self.max_frame_num_to_track is not None:
            total = self.max_frame_num_to_track
        else:
            total = len(self.mainwindow.files_list) - self.start_frame_idx + 1

        with torch.inference_mode(), torch.autocast(self.mainwindow.segany_video.device,
                                                    dtype=self.mainwindow.segany_video.model_dtype,
                                                    enabled=torch.cuda.is_available()):

            # if not self.mainwindow.use_segment_anything_video:
            #     self.mainwindow.actionVideo_segment.setEnabled(False)
            #     self.mainwindow.actionVideo_segment_once.setEnabled(False)
            #     self.mainwindow.actionVideo_segment_five_times.setEnabled(False)
            #     return

            if self.mainwindow.segany_video.inference_state == {}:
                self.mainwindow.segany_video.init_state(self.mainwindow.image_root, self.mainwindow.files_list)
            self.mainwindow.segany_video.reset_state()

            current_file = self.mainwindow.files_list[self.start_frame_idx]
            current_file_path = os.path.join(self.mainwindow.image_root, current_file)
            current_label_path = os.path.join(self.mainwindow.label_root, '.'.join(current_file.split('.')[:-1]) + '.json')
            current_label = Annotation(current_file_path, current_label_path)

            current_label.load_annotation()

            group_object_dict = {}

            for object in current_label.objects:
                group = int(object.group)
                segmentation = [(int(p[1]), int(p[0])) for p in object.segmentation]
                category = object.category
                is_crowd = object.iscrowd
                layer = object.layer
                note = object.note

                # fill mask
                mask = np.zeros(shape=(current_label.height, current_label.width), dtype=np.uint8)
                xs = [x for x, y in segmentation]
                ys = [y for x, y in segmentation]
                rr, cc = polygon(xs, ys, mask.shape)
                mask[rr, cc] = 1

                if group not in group_object_dict:
                    group_object_dict[group] = {}
                    group_object_dict[group]['mask'] = mask
                    group_object_dict[group]['category'] = category
                    group_object_dict[group]['is_crowd'] = is_crowd
                    group_object_dict[group]['layer'] = layer
                    group_object_dict[group]['note'] = note
                else:
                    group_object_dict[group]['mask'] = group_object_dict[group]['mask'] + mask

            if len(group_object_dict) < 1:
                self.tag.emit(0, total, True, True, 'Please label objects before video segment.')
                return
            try:
                for group, object_dict in group_object_dict.items():
                    mask = object_dict['mask']
                    self.mainwindow.segany_video.add_new_mask(self.start_frame_idx, group, mask)

                for index, (out_frame_idxs, out_obj_ids, out_mask_logits) in enumerate(self.mainwindow.segany_video.predictor.propagate_in_video(
                        self.mainwindow.segany_video.inference_state,
                        start_frame_idx=self.start_frame_idx,
                        max_frame_num_to_track=self.max_frame_num_to_track,
                        reverse=False,
                )):
                    if index == 0:  # 忽略当前图片
                        continue
                    file = self.mainwindow.files_list[out_frame_idxs]
                    file_path = os.path.join(self.mainwindow.image_root, file)
                    label_path = os.path.join(self.mainwindow.label_root, '.'.join(file.split('.')[:-1]) + '.json')
                    annotation = Annotation(file_path, label_path)

                    objects = []
                    for index_mask, out_obj_id in enumerate(out_obj_ids):

                        masks = out_mask_logits[index_mask]   # [1, h, w]
                        masks = masks > 0
                        masks = masks.cpu().numpy()

                        # mask to polygon
                        masks = masks.astype('uint8') * 255
                        h, w = masks.shape[-2:]
                        masks = masks.reshape(h, w)

                        if self.mainwindow.scene.contour_mode == CONTOURMode.SAVE_ALL:
                            # 当保留所有轮廓时，检测所有轮廓，并建立二层等级关系
                            contours, hierarchy = cv2.findContours(masks, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
                        else:
                            # 当只保留外轮廓或单个mask时，只检测外轮廓
                            contours, hierarchy = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

                        if self.mainwindow.scene.contour_mode == CONTOURMode.SAVE_MAX_ONLY and contours:
                            largest_contour = max(contours, key=cv2.contourArea)  # 只保留面积最大的轮廓
                            contours = [largest_contour]

                        for contour in contours:
                            # polydp
                            epsilon_factor = 0.001
                            epsilon = epsilon_factor * cv2.arcLength(contour, True)
                            contour = cv2.approxPolyDP(contour, epsilon, True)

                            if len(contour) < 3:
                                continue

                            segmentation = []
                            xmin, ymin, xmax, ymax = annotation.width, annotation.height, 0, 0
                            for point in contour:
                                x, y = point[0]
                                x, y = float(x), float(y)
                                xmin = min(x, xmin)
                                ymin = min(x, ymin)
                                xmax = max(y, xmax)
                                ymax = max(y, ymax)

                                segmentation.append((x, y))

                            area = calculate_area(segmentation)
                            # bbox = (xmin, ymin, xmax, ymax)
                            bbox = None
                            obj = Object(category=group_object_dict[out_obj_id]['category'],
                                         group=out_obj_id,
                                         segmentation=segmentation,
                                         area=area,
                                         layer=group_object_dict[out_obj_id]['layer'],
                                         bbox=bbox,
                                         iscrowd=group_object_dict[out_obj_id]['is_crowd'],
                                         note=group_object_dict[out_obj_id]['note'])
                            objects.append(obj)

                    annotation.objects = objects
                    annotation.save_annotation()
                    self.tag.emit(index, total, False, False, '')

                self.tag.emit(index, total, True, False, '')

            except Exception as e:
                self.tag.emit(index, total, True, True, '{}'.format(e))


class InitSegAnyThread(QThread):
    tag = pyqtSignal(bool, bool)

    def __init__(self, mainwindow):
        super(InitSegAnyThread, self).__init__()
        self.mainwindow = mainwindow
        self.model_path: str = None

    def run(self):
        sam_tag = False
        sam_video_tag = False
        if self.model_path is not None:
            try:
                self.mainwindow.segany = SegAny(self.model_path, False)
                sam_tag = True
            except Exception as e:
                print('Init SAM Error: ', e)
                sam_tag = False
            if 'sam2' in self.model_path:
                try:
                    self.mainwindow.segany_video = SegAnyVideo(self.model_path, False)
                    sam_video_tag = True
                except Exception as e:
                    print('Init SAM2 video Error: ', e)
                    sam_video_tag = False

        self.tag.emit(sam_tag, sam_video_tag)

class InpaintWorker(QThread):
    finished = pyqtSignal(object) 

    def __init__(self, image, mask, model, refinement=False):
        super().__init__()
        self.image = image
        self.mask = mask
        self.model = model
        self.refinement = refinement

    def run(self):
        result = inpaint(self.image, self.mask, self.model, self.refinement)
        self.finished.emit(result)

class Mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Mainwindow, self).__init__()
        self.setupUi(self)
        
        self.image_root = ''
        self.result_root = ''
        self.files_list = []
        self.results_files_list = []
        self.current_label = '__mask__'
        self.current_group = 1
        self.current_index = None
        self.show_paint = False
        self.saved = True
        self.load_finished = False
        self.polygons: list = []
        self.scene = Scene(self)
        self.view = View(self)
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)
        
        self.labelCoordinates = QtWidgets.QLabel('')
        self.status_bar.addPermanentWidget(self.labelCoordinates)
        
        self.current_label:Annotation = None
        self.lama = None
        self.can_be_inpainted = True
        self.use_segment_anything = False
        self.use_segment_anything_video = False
        self.gpu_resource_thread = None
        # sam初始化线程，大模型加载较慢
        self.init_segany_thread = InitSegAnyThread(self)
        self.init_segany_thread.tag.connect(self.init_sam_finish)
        self.use_remote_sam = False
        
        self.init_gui()
        self.connect()
    
    def init_gui(self):
        self.labelPaint = QtWidgets.QLabel('')
        self.labelPaint.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelPaint.setFixedWidth(300)
        self.statusbar.addPermanentWidget(self.labelPaint)
        
        self.labelGPUResource = QtWidgets.QLabel('')
        self.labelGPUResource.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelGPUResource.setFixedWidth(50)
        self.statusbar.addPermanentWidget(self.labelGPUResource)
        
        self.labelCoord = QtWidgets.QLabel('')
        self.labelCoord.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelCoord.setFixedWidth(130)
        self.statusbar.addPermanentWidget(self.labelCoord)
        
        self.labelData = QtWidgets.QLabel('')
        self.labelData.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelData.setFixedWidth(130)
        self.statusbar.addPermanentWidget(self.labelData)
        
        self.model_manager_dialog = ModelManagerDialog(self, self)
        
        self.lama, _ = load_lama_model('checkpoints/big-lama')
        print("load lama success!")
        self.setWindowTitle("IORTool")
        self.actionPrevious_image.setEnabled(False)
        self.actionNext_image.setEnabled(False)
        self.actionDelete.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionFitWindow.setEnabled(False)
        self.actionPolygon.setEnabled(False)
        self.actionVisible.setEnabled(False)
        self.actionObjectRemoval.setEnabled(False)
        self.actionChangeView.setEnabled(False)
        self.actionSegment_anything_point.setEnabled(False)
        self.actionDelete.setEnabled(False)
        self.actionCancel.setEnabled(False)
        self.actionFinish.setEnabled(False)
    
    def init_segment_anything(self, model_path=None, checked=None):
        if checked is not None and not checked:
            return

        if model_path is None:
            model_name = model_path
        else:
            model_name = os.path.basename(model_path)

        if not self.saved:
            result = QtWidgets.QMessageBox.question(self, 'Warning', 'Proceed without saved?', QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                if isinstance(self.sender(), QtWidgets.QAction):
                    self.sender().setChecked(False)
                return
        if model_name is None:
            if self.use_segment_anything:
                model_name = os.path.split(self.segany.checkpoint)[-1]
            else:
                return
        # 等待sam线程完成
        self.actionSegment_anything_point.setEnabled(False)
        # self.actionSegment_anything_box.setEnabled(False)
        try:
            self.seganythread.wait()
            self.seganythread.results_dict.clear()
        except:
            if isinstance(self.sender(), QtWidgets.QAction):
                self.sender().setChecked(False)

        if model_name == '':
            self.use_segment_anything = False
            self.model_manager_dialog.update_ui()
            return

        if not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, 'Warning',
                                          'The checkpoint of [Segment anything] not existed. If you want use quick annotate, please download from {}'.format(
                                              'https://github.com/facebookresearch/segment-anything#model-checkpoints'))
            self.use_segment_anything = False
            self.model_manager_dialog.update_ui()
            return

        self.init_segany_thread.model_path = model_path
        self.init_segany_thread.start()
        self.setEnabled(False)
    
    def init_sam_finish(self, sam_tag:bool, sam_video_tag:bool):

        print('sam_tag:', f'\033[32m{sam_tag}\033[0m' if sam_tag else f'\033[31m{sam_tag}\033[0m',
              'sam_video_tag: ', f'\033[32m{sam_video_tag}\033[0m' if sam_video_tag else f'\033[31m{sam_video_tag}\033[0m')
        self.setEnabled(True)
        if sam_video_tag:
            self.use_segment_anything_video = True
            if self.files_list:
                self.segany_video.init_state(self.image_root, self.files_list)

            self.segany_video_thread = SegAnyVideoThread(self)
            self.segany_video_thread.tag.connect(self.seg_video_finish)

            # sam2 建议使用bfloat16
            if self.segany.model_dtype == torch.float32:
                if 'en' == 'zh':
                    QtWidgets.QMessageBox.warning(self,
                                                  'warning',
                                                  """建议使用bfloat16模式进行视频分割\n在[设置]界面打开该功能""")
                else:
                    QtWidgets.QMessageBox.warning(self,
                                                  'warning',
                                                  """Suggest Use bfloat16 mode to segment video.\nYou can open it in [Setting].""")

        else:
            self.segany_video_thread = None
            self.use_segment_anything_video = False
            torch.cuda.empty_cache()

        # self.actionVideo_segment.setEnabled(self.use_segment_anything_video)
        # self.actionVideo_segment_once.setEnabled(self.use_segment_anything_video)
        # self.actionVideo_segment_five_times.setEnabled(self.use_segment_anything_video)

        if sam_tag:
            self.use_segment_anything = True
            if self.use_segment_anything:
                if self.segany.device != 'cpu':
                    if self.gpu_resource_thread is None:
                        self.gpu_resource_thread = GPUResource_Thread()
                        self.gpu_resource_thread.message.connect(self.labelGPUResource.setText)
                        self.gpu_resource_thread.start()
                else:
                    self.labelGPUResource.setText('cpu')
            else:
                self.labelGPUResource.setText('segment anything unused.')
            tooltip = 'model: {}'.format(os.path.split(self.segany.checkpoint)[-1])
            tooltip += '\ndtype: {}'.format(self.segany.model_dtype)
            tooltip += '\ntorch: {}'.format(torch.__version__)
            if self.segany.device == 'cuda':
                try:
                    tooltip += '\ncuda : {}'.format(torch.version.cuda)
                except: 
                    pass
            self.labelGPUResource.setToolTip(tooltip)

            self.seganythread = SegAnyThread(self)
            self.seganythread.tag.connect(self.sam_encoder_finish)
            self.seganythread.start()

            if self.current_index is not None:
                self.show_image(self.current_index)

            checkpoint_name = os.path.split(self.segany.checkpoint)[-1]
            self.statusbar.showMessage('Use the checkpoint named {}.'.format(checkpoint_name), 3000)
        else:
            self.use_segment_anything = False

        self.model_manager_dialog.update_ui()
    
    def sam_encoder_finish(self, index:int, state:int, message:str):
        # 图片识别状态刷新
        if state == 1: color = '#00FF00'
        elif state == 0: color = '#999999'
        elif state == 2: color = '#FFFF00'
        elif state == 3:
            color = '#999999'
            if index == self.current_index:
                QtWidgets.QMessageBox.warning(self, 'warning','SAM not support the image: {}\nError: {}'.format(self.files_list[index], message))

        else: color = '#999999'

        # if index == self.current_index:
        #     self.files_dock_widget.label_current_state.setStyleSheet("background-color: {};".format(color))
        # elif index == self.current_index - 1:
        #     self.files_dock_widget.label_prev_state.setStyleSheet("background-color: {};".format(color))
        # elif index == self.current_index + 1:
        #     self.files_dock_widget.label_next_state.setStyleSheet("background-color: {};".format(color))
        # else:
        #     pass

        # item = self.files_dock_widget.listWidget.item(index)
        # widget = self.files_dock_widget.listWidget.itemWidget(item)
        # if widget is not None:
        #     state_color = widget.findChild(QtWidgets.QLabel, 'state_color')
        #     state_color.setStyleSheet("background-color: {};".format(color))

        if state == 1:  # 识别完
            # 如果当前图片刚识别完，需刷新segany状态
            if self.current_index == index:
                self.SeganyEnabled()
                # self.plugin_manager_dialog.trigger_after_sam_encode_finished(index)

    def SeganyEnabled(self):
        """
        segany激活
        判断当前图片是否缓存特征图，如果存在特征图，设置segany参数，并开放半自动标注
        :return:
        """
        if not self.use_segment_anything:
            self.actionSegment_anything_point.setEnabled(False)
            # self.actionSegment_anything_box.setEnabled(False)
            return

        results = self.seganythread.results_dict.get(self.current_index, {})
        features = results.get('features', None)
        original_size = results.get('original_size', None)
        input_size = results.get('input_size', None)

        if features is not None and original_size is not None and input_size is not None:
            if self.segany.model_source == 'sam_hq':
                features, interm_features = features
                self.segany.predictor_with_point_prompt.interm_features = interm_features
            self.segany.predictor_with_point_prompt.features = features
            self.segany.predictor_with_point_prompt.original_size = original_size
            self.segany.predictor_with_point_prompt.input_size = input_size
            self.segany.predictor_with_point_prompt.is_image_set = True
            # sam2
            self.segany.predictor_with_point_prompt._orig_hw = list(original_size)
            self.segany.predictor_with_point_prompt._features = features
            self.segany.predictor_with_point_prompt._is_image_set = True

            self.actionSegment_anything_point.setEnabled(True)
            # self.actionSegment_anything_box.setEnabled(True)
        else:
            self.segany.predictor_with_point_prompt.reset_image()
            self.actionSegment_anything_point.setEnabled(False)
            # self.actionSegment_anything_box.setEnabled(False)

    def seg_video_start(self, max_frame_num_to_track=None):
        if self.current_index == None:
            return

        if not self.saved:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Current annotation has not been saved!')
            return

        self.setEnabled(False)
        self.statusbar_change_status(is_message=False)
        self.segany_video_thread.start_frame_idx = self.current_index
        self.segany_video_thread.max_frame_num_to_track=max_frame_num_to_track
        self.segany_video_thread.start()

    def seg_video_finish(self, current, total, finished, is_error, message):
        if is_error:
            QtWidgets.QMessageBox.warning(self, 'warning', message)

        self.progressbar.setMaximum(total)
        self.progressbar.setValue(current)
        if finished:
            self.statusbar_change_status(is_message=True)
            self.progressbar.setValue(0)
            self.setEnabled(True)
    
    def save_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not dir:
            return
        self.result_root = dir
        self.actionOpen_save_dir.setStatusTip("Result root: {}".format(self.result_root))
        if self.current_index is not None:
            self.show_image(self.current_index)
        
        self.listWidget_results.clear()
        self.update_widget(self.listWidget_results)

        files = []
        suffixs = tuple(['{}'.format(fmt.data().decode('ascii').lower()) for fmt in QtGui.QImageReader.supportedImageFormats()])
        for f in os.listdir(dir):
            if f.lower().endswith(suffixs):
                # f = os.path.join(dir, f)
                files.append(f)
        files = sorted(files)
        self.results_files_list = files
    
    def update_widget(self, widget):
        widget.clear()
        if self.files_list is None:
            return

        for idx, file_name in enumerate(self.files_list):
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 30))

            item.setText(f'[{idx + 1}] {file_name}')
            widget.addItem(item)
        
    def open_image_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not dir:
            return

        self.files_list.clear()
        self.listWidgetFiles.clear()

        files = []
        suffixs = tuple(['{}'.format(fmt.data().decode('ascii').lower()) for fmt in QtGui.QImageReader.supportedImageFormats()])
        for f in os.listdir(dir):
            if f.lower().endswith(suffixs):
                # f = os.path.join(dir, f)
                files.append(f)
        files = sorted(files)
        self.files_list = files
        
        self.update_widget(self.listWidgetFiles)

        self.current_index = 0

        self.image_root = dir
        if self.result_root == '':
            self.result_root = os.path.join(os.path.dirname(self.image_root), 'outputs')
            
        self.actionOpen_image_dir.setStatusTip("Image root: {}".format(self.image_root))

        self.saved = True

        self.show_image(self.current_index)
    
    def show_image(self, index:int, show_paint=False, zoomfit:bool=True):
        if not self.saved:
            result = QtWidgets.QMessageBox.question(self, 'Warning', 'Proceed without saved?', QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if result == QtWidgets.QMessageBox.StandardButton.No:
                return

        self.reset_action()
        self.scene.cancel_draw()
        self.scene.unload_image()

        self.load_finished = False
        self.saved = True
        if not -1 < index < len(self.files_list):
            return
        try:
            self.current_index = index
            file_path = os.path.join(self.image_root, self.files_list[index])
            
            _, name = os.path.split(file_path)
            label_path = os.path.join(self.result_root, '.'.join(name.split('.')[:-1]) + '.json')
            result_path = os.path.join(self.result_root, '.'.join(name.split('.')[:-1]) + '.png')
            self.object_removal_path = os.path.join(self.result_root, '.'.join(name.split('.')[:-1]) + '_paint.png')

            image_data = Image.open(file_path)

            self.png_palette = image_data.getpalette()

            self.actionPolygon.setEnabled(True)
            self.actionCancel.setEnabled(True)
            self.actionVisible.setEnabled(False)
            self.actionFitWindow.setEnabled(True)
            self.actionFinish.setEnabled(True)
            self.actionSegment_anything_point.setEnabled(False)
            self.actionObjectRemoval.setEnabled(False)
            self.actionSave.setEnabled(False)
            
            paint_exists = os.path.exists(self.object_removal_path)

            if paint_exists:
                self.actionChangeView.setEnabled(True)

            if paint_exists and show_paint:
                img_path = self.object_removal_path
            else:
                img_path = file_path

            self.scene.load_image(img_path)
            self.listWidgetFiles.setCurrentRow(self.current_index)

            if zoomfit:
                self.view.zoom_fit()

            # 判断图像是否旋转
            exif_info = image_data.getexif()
            if exif_info and exif_info.get(274, 1) != 1:
                warning_info = '这幅图像包含EXIF元数据，且图像的方向已被旋转.\n建议去除EXIF信息后再进行标注\n你可以使用[菜单栏]-[工具]-[处理exif标签]功能处理图像的旋转问题。'\
                    if 'en' == 'zh' \
                    else 'This image has EXIF metadata, and the image orientation is rotated.\nSuggest labeling after removing the EXIF metadata.\nYou can use the function of [Process EXIF tag] in [Tools] in [Menu bar] to deal with the problem of images.'
                QtWidgets.QMessageBox.warning(self, 'Warning', warning_info, QtWidgets.QMessageBox.Ok)
            
            if self.use_segment_anything and self.can_be_inpainted:
                self.segany.reset_image()
                self.seganythread.index = index
                self.seganythread.start()
                self.SeganyEnabled()
            
            self.current_group = 1
            self.current_label = Annotation(file_path, result_path, label_path)
            # self.current_label.load_annotation()
            
            if self.current_label is not None:
                self.setWindowTitle('{}'.format(self.current_label.label_path))
            else:
                self.setWindowTitle('{}'.format(file_path))

            self.load_finished = True

        except Exception as e:
            print(e)
        finally:
            if self.current_index > 0:
                self.actionPrevious_image.setEnabled(True)
            else:
                self.actionPrevious_image.setEnabled(False)

            if self.current_index < len(self.files_list) - 1:
                self.actionNext_image.setEnabled(True)
            else:
                self.actionNext_image.setEnabled(False)
    
    def reset_action(self):
        self.actionPrevious_image.setEnabled(False)
        self.actionNext_image.setEnabled(False)
        self.actionDelete.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionFitWindow.setEnabled(False)
        self.actionPolygon.setEnabled(False)
        self.actionVisible.setEnabled(False)
        self.actionObjectRemoval.setEnabled(False)
        self.actionChangeView.setEnabled(False)
        self.actionSegment_anything_point.setEnabled(False)
        self.actionDelete.setEnabled(False)
        self.actionCancel.setEnabled(False)
        self.actionFinish.setEnabled(False)
    
    def set_changed(self, changed:bool):
        if changed:
            self.is_changed =True
            self.setWindowTitle('*{}'.format(os.path.join(self.files_root, self.files_list[self.current_index])))
            self.actionSave.setEnabled(True)
        else:
            self.is_changed = False
            self.setWindowTitle('{}'.format(os.path.join(self.files_root, self.files_list[self.current_index])))
            self.actionSave.setEnabled(False)
    
    def previous_image(self):
        if self.scene.mode != STATUSMode.VIEW:
            return
        if self.current_index <= 0:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'It is the first iamge!', buttons=QtWidgets.QMessageBox.StandardButton.Yes)
            return
        self.current_index -= 1
        self.show_image(self.current_index)
    
    def next_image(self):
        if self.scene.mode != STATUSMode.VIEW:
            return
        if self.current_index >= len(self.files_list) - 1:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'It is the last iamge!', buttons=QtWidgets.QMessageBox.StandardButton.Yes)
            return
        self.current_index += 1
        self.show_image(self.current_index)
    
    def save(self):
        if self.current_label == None:
            return
        if self.result_root == '':
            os.makedirs(self.result_root, exist_ok=True)
        self.current_label.objects.clear()

        for polygon in self.polygons:
            objects = polygon.to_object()
            self.current_label.objects.append(objects)
        
        self.current_label.save_annotation()
        if self.reverse_mask.isChecked():
            self.current_label.save_mask(self.scene.draw_mode, reverse=True)
        else:
            self.current_label.save_mask(self.scene.draw_mode)
        self.labelPaint.setText("Save annotation and mask finished!")
        self.set_save_state(True)
        self.actionObjectRemoval.setEnabled(True)
    
    def object_removal(self):
        if not self.saved:
            return
        self.labelPaint.setText(f"start object removal, please wait...")
        self.actionPrevious_image.setEnabled(False)
        self.actionNext_image.setEnabled(False)
        self.actionObjectRemoval.setEnabled(False)
        img_path = os.path.join(self.image_root, self.files_list[self.current_index])
        mask_path = self.current_label.result_path

        image = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.refinement.isChecked():
            self.worker = InpaintWorker(image, mask, self.lama, refinement=True)
        else:
            self.worker = InpaintWorker(image, mask, self.lama)
        self.worker.finished.connect(self.show_result)
        self.worker.start()
    
    def show_result(self, result):
        paint_path = self.object_removal_path
        self.labelPaint.setText("object removal finished!")
        
        cv2.imwrite(paint_path, result[:, :, ::-1])
        self.set_save_state(True)
        self.change_view()
        
    def set_save_state(self, is_saved:bool):
        self.saved = is_saved
        if self.current_index > 0:
            self.actionPrevious_image.setEnabled(True)
        else:
            self.actionPrevious_image.setEnabled(False)

        if self.current_index < len(self.files_list) - 1:
            self.actionNext_image.setEnabled(True)
        else:
            self.actionNext_image.setEnabled(False)

        if self.files_list is not None and self.current_index is not None:

            if is_saved:
                self.setWindowTitle(self.current_label.label_path)
            else:
                self.setWindowTitle('*{}'.format(self.current_label.label_path))
    
    def change_view(self):
        if self.show_paint:
            self.show_paint = False
            paint_icon = QtGui.QIcon()
            paint_icon.addPixmap(QtGui.QPixmap("icons/照片_pic.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionChangeView.setIcon(paint_icon)
        else:
            self.show_paint = True
            paint_icon = QtGui.QIcon()
            paint_icon.addPixmap(QtGui.QPixmap("icons/instance.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.actionChangeView.setIcon(paint_icon)
        self.show_image(index=self.current_index, show_paint=self.show_paint)
        
    def model_manage(self):
        self.model_manager_dialog.show()
    
    def listwidgetfiles_doubleclick(self):
        row = self.listWidgetFiles.currentRow()
        self.show_image(row)
        
    def exit(self):
        
        self.close()
    
    def connect(self):
        self.actionOpen_image_dir.triggered.connect(self.open_image_dir)
        self.actionOpen_save_dir.triggered.connect(self.save_dir)
        self.actionPrevious_image.triggered.connect(self.previous_image)
        self.actionNext_image.triggered.connect(self.next_image)
        self.actionFitWindow.triggered.connect(self.view.zoom_fit)
        self.actionPolygon.triggered.connect(self.scene.start_draw_polygon)
        self.actionCancel.triggered.connect(self.scene.cancel_draw)
        self.actionFinish.triggered.connect(self.scene.finish_draw)
        self.actionSave.triggered.connect(self.save)
        self.actionExit.triggered.connect(self.exit)
        self.actionObjectRemoval.triggered.connect(self.object_removal)
        self.actionChangeView.triggered.connect(self.change_view)
        self.actionSegment_anything_point.triggered.connect(self.scene.start_segment_anything)
        # self.actionVisible.triggered.connect()
        self.actionModel_manage.triggered.connect(self.model_manage)
        self.actionDelete.triggered.connect(self.scene.delete_selected_graph)
        self.actionModel_manage.setStatusTip(CHECKPOINT_PATH)
        
        self.listWidgetFiles.clicked.connect(self.listwidgetfiles_doubleclick)
        
