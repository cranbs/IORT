import os
from IORT.canvas import Scene, View
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from IORT.ui.MainWindow import Ui_MainWindow
from IORT.configs import STATUSMode
from IORT.anno import Annotation
from PIL import Image
from LaMaProject.bin.object_removal import load_lama_model, inpaint
import cv2
import functools

class InpaintWorker(QThread):
    finished = pyqtSignal(object) 

    def __init__(self, image, mask, model):
        super().__init__()
        self.image = image
        self.mask = mask
        self.model = model

    def run(self):
        result = inpaint(self.image, self.mask, self.model)
        self.finished.emit(result)

class Mainwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Mainwindow, self).__init__()
        self.setupUi(self)
        
        self.image_root = ''
        self.result_root = ''
        self.files_list = []
        self.mask_files_list = []
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
        
        self.init_gui()
        self.connect()
    
    def init_gui(self):
        self.labelPaint = QtWidgets.QLabel('')
        self.labelPaint.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelPaint.setFixedWidth(300)
        self.statusbar.addPermanentWidget(self.labelPaint)
        
        self.labelCoord = QtWidgets.QLabel('')
        self.labelCoord.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelCoord.setFixedWidth(150)
        self.statusbar.addPermanentWidget(self.labelCoord)
        
        self.labelData = QtWidgets.QLabel('')
        self.labelData.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.labelData.setFixedWidth(150)
        self.statusbar.addPermanentWidget(self.labelData)
        
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
    
    def save_dir(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self)
        if not dir:
            return
        self.result_root = dir
        self.actionOpen_save_dir.setStatusTip("Result root: {}".format(self.result_root))
        if self.current_index is not None:
            self.show_image(self.current_index)
        
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

        for file_name in self.files_list:
            item = QtWidgets.QListWidgetItem()
            item.setText(file_name)
            self.listWidgetFiles.addItem(item)

        self.current_index = 0

        self.image_root = dir
        if self.result_root == '':
            self.result_root = os.path.join(os.path.dirname(self.image_root), 'outputs')
            os.makedirs(self.result_root, exist_ok=True)
            
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
            self.checkBox_visible.setChecked(True)
            self.actionVisible.setEnabled(True)
            self.actionFitWindow.setEnabled(True)
            self.actionFinish.setEnabled(True)
            self.actionSegment_anything_point.setEnabled(True)
            
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
                    if self.cfg['software']['language'] == 'zh' \
                    else 'This image has EXIF metadata, and the image orientation is rotated.\nSuggest labeling after removing the EXIF metadata.\nYou can use the function of [Process EXIF tag] in [Tools] in [Menu bar] to deal with the problem of images.'
                QtWidgets.QMessageBox.warning(self, 'Warning', warning_info, QtWidgets.QMessageBox.Ok)

            self.current_group = 1
            self.current_label = Annotation(file_path, result_path, label_path)
            self.current_label.load_annotation()

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
        self.current_label.save_mask()
        self.labelPaint.setText("Save annotation and mask finished!")
        self.set_save_state(True)
    
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
        
